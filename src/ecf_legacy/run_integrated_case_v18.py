"""Four source-native comparisons and the integrated recovery/direct case."""
from pathlib import Path
import sys,json,time,gc
ROOT=__import__("repository_paths").ROOT;pass # Dependencies are installed through requirements.txt.
import numpy as np
from threadpoolctl import threadpool_limits
from green_direct_v18 import make_case,recovery_metrics
from priced_library_v14 import priced_library
from dispatch_result_io import load,save
from branch_dispatch import evaluate
from audit_dispatch_feasibility import primal_audit
from native_industrial_v14 import parameters,run as native_run
import balanced_master_v16 as balanced
import parametric_master_v13 as economic

KEYS=['objective_actual','generation_emission_t','curtailment_mwh']

def search(net,prof,lib,out,name,starts,budget,refs=None,fixed=None):
    cached=load(out,name)
    if cached is not None:return cached
    value=lambda r:float(r[KEYS[0]] if refs is None else refs[0]*max(r[k]/v for k,v in zip(KEYS,refs)))
    violation=lambda r:float(np.maximum(r['allocation_t']-budget,0).sum())
    merit=lambda r:value(r)+2e4*violation(r)
    best=None;history=[];begin=time.perf_counter()
    for seed,initial in enumerate(starts):
        current=initial;trust=.10;buffer=1e-5;stalled=0
        for i in range(40):
            v=violation(current)
            if v<=1e-4:
                if best is None or value(current)<value(best):best=current.copy()
                buffer=max(2e-8,buffer*.25)
            row=dict(start=seed,iteration=i,objective=value(current),violation=v,**{k:current[k] for k in KEYS})
            history.append(row)
            if i%4==0:print(name,row,flush=True)
            args=dict(center=current,budgets=budget,trust=trust,budget_buffer=buffer,fixed_branch=fixed,time_limit=75,mip_gap=1e-7)
            trial=economic.master(net,prof,lib,**args) if refs is None else balanced.master(net,prof,lib,refs,**args)
            if 'pg' not in trial:trust*=.5;continue
            delta=merit(current)-merit(trial)
            if delta>1e-6:
                current=trial;trust=min(.20,trust*1.3);stalled=0
                if violation(current)<=1e-4 and delta<.005 and buffer<=2e-8:
                    if best is None or value(current)<value(best):best=current.copy()
                    break
            else:trust*=.5;stalled+=1
            if stalled>=9 or trust<1e-5:break
    if best is None:
        current.update(history=history,budgets_t=budget,budget_violation_t=violation(current),carbon_feasible=False)
        save(out,name+'_unresolved',current);raise RuntimeError(name+' has no verified budget-feasible point')
    best.update(history=history,budgets_t=budget,budget_violation_t=violation(best),carbon_feasible=True,total_runtime_s=time.perf_counter()-begin)
    if refs is not None:best['normalization_reference_totals']=refs
    save(out,name,best);return best

def run(date,mode):
    net,prof=make_case(date,mode);out=ROOT/f'results/integrated_v18_{mode}_{date}';out.mkdir(exist_ok=True)
    old=load(ROOT/f'results/comparison_v12_{date}','M1_constant');net.fixed_commitment=old['commitment']
    models,rows=parameters(net);p0=np.array([r['nominal_power_mw'] for r in rows])[:,None]*np.ones((6,48))
    base={'P':p0[:,None,:],'admissible':np.ones((6,1),bool),'metadata':{'version':'constant_production_reference_v18'}}
    constant=load(out,'constant_reference')
    if constant is None:
        constant=economic.master(net,prof,base,time_limit=90,mip_gap=1e-8);assert constant['primal_audit']['passed'];save(out,'constant_reference',constant)
    refs=np.array([max(constant[k],1. if k=='curtailment_mwh' else 1e-6) for k in KEYS]);budget=.97*constant['allocation_t']
    m1=search(net,prof,base,out,'M1_constant',[constant],budget,fixed=np.zeros(6,int))
    economic.CACHE.clear();gc.collect()
    m2=load(out,'M2_native')
    if m2 is None:
        m2=native_run(date,25.,12,240,True,False,net,prof,out);save(out,'M2_native',m2)
    m3=load(out,'M3_native')
    if m3 is None:
        from recent_methods_v12 import copf
        for row in rows:row.update(case_execution_bounds=True,terminal_recovery=False)
        linear=native_run(date,25.,None,120,True,False,net,prof,out)
        candidates=[]
        for seed,start in enumerate([linear,m1]):
            r=load(out,f'M3_start{seed}')
            if r is None:
                r=copf(net,prof,None,start,budget,native_rows=rows,source_price=25.,max_iter=1800,max_seconds=240,native_solver_profile='lowmemory')
                r['added_terminal_recovery']=False;save(out,f'M3_start{seed}',r)
            print('M3',seed,{k:r.get(k) for k in ['status','native_nlp_constraint_violation','budget_violation_t']+KEYS},flush=True)
            if r.get('carbon_feasible') and r.get('native_nlp_constraint_violation',1)<1e-5:candidates.append(r)
            gc.collect()
        if not candidates:raise RuntimeError('No primal-feasible original C-OPF result')
        m3=min(candidates,key=lambda r:r[KEYS[0]]+25*r[KEYS[1]]);save(out,'M3_native',m3)
    lib=priced_library(net,include_v15=True,include_v16=True)
    rec=prof['time_h']>=20;energyplus=np.maximum(lib['P']-p0[:,None,:],0)[:,:,rec].sum(axis=2)*prof['dt_h']
    recovery_cap=.10*p0[:,0]*4
    lib['admissible']&=energyplus<=recovery_cap[:,None]+1e-8
    raw=load(out,'M4_unconstrained')
    if raw is None:
        raw=balanced.master(net,prof,lib,refs,time_limit=90,mip_gap=1e-7);save(out,'M4_unconstrained',raw)
    constant_full=m1.copy();constant_full['weights']=np.zeros(lib['P'].shape[:2]);constant_full['weights'][:,0]=1
    m4=search(net,prof,lib,out,'M4_proposed',[raw,constant_full],budget,refs=refs)
    # Audit with recomputed fuel curves and exact carbon, never plot surrogates.
    audit=[]
    for name,r in [('M1',m1),('M2',m2),('M3',m3),('M4',m4)]:
        check=evaluate(net,prof,r['pg'],r['pr'],r['pl'])
        assert max(abs(check[k]-r[k]) for k in KEYS)<1e-5
        metrics=recovery_metrics(net,prof,r,constant);save(out,name+'_recovery',metrics)
        audit.append(dict(method=name,**{k:r[k] for k in KEYS},source_status=r['status'],carbon_balance_residual=check['carbon_balance_residual'],attribution_residual=check['attribution_conservation_residual']))
    improvements={name:{key:100*(r[key]-m4[key])/r[key] if abs(r[key])>1e-6 else None for key in KEYS} for name,r in [('M1',m1),('M2',m2),('M3',m3)]}
    protocol=dict(date=date,mode=mode,methods=audit,improvements=improvements,normalization='Common constant-production economic reference, fixed before method comparison',reference_totals=refs.tolist(),recovery_window_start_h=20.,recovery_cap_mwh=recovery_cap.tolist(),budget_ratio=.97,source_energy_preserved=True,added_baseline_terminal_controller=False,all_methods_share_proposed_library=False,formulation='lossless direct factory network; original native comparator interfaces',direct_case=prof.get('direct_connection'))
    (out/'audit.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
    print('COMPLETE',improvements,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):run(sys.argv[1] if len(sys.argv)>1 else '2020-11-29',sys.argv[2] if len(sys.argv)>2 else 'direct')
