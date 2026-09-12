"""Four methods including the proposed method, common continuous-checked data.

M1: constant industrial production, endogenous carbon-aware generation.
M2: Tan-2024 PVE + convex dispatch + nearest complete-trajectory recovery.
M3: Chen-2025 dual-flow C-OPF + nearest complete-trajectory recovery.
M4: integer trajectory / exact-carbon sequential coordination with adaptive
    feasibility buffer, two common starts and discrete neighborhood polishing.

All generation problems have the same native fuel curves and fixed commitment.
No-carbon economic dispatch is a lower-bound calculation, not a fifth method.
"""
from pathlib import Path
import sys,json,time,gc
ROOT=__import__("repository_paths").ROOT
import numpy as np
from threadpoolctl import threadpool_limits
from rts_network import make_rts_network,rts_profiles
from branch_dispatch import load_library
from parametric_master_v12 import master
from recent_methods_v12 import pve_library,copf

def save(out,name,r):
    np.savez_compressed(out/f'{name}.npz',**{k:v for k,v in r.items() if isinstance(v,np.ndarray)})
    (out/f'{name}.json').write_text(json.dumps({k:v for k,v in r.items() if not isinstance(v,np.ndarray)},indent=2),encoding='utf8')

def load(out,name):
    p=out/f'{name}.json'
    if not p.exists():return None
    return {**json.loads(p.read_text()),**dict(np.load(out/f'{name}.npz'))}

def continuous_library(net):
    lib=load_library(net,version='public_v3');lib['P']=lib['P'].copy()
    for k,b in zip(*np.where(lib['admissible'])):
        base=ROOT/f'results/execution/public_v12/site{k}_candidate{b}'
        report=json.loads(base.with_suffix('.json').read_text());assert report['thermal_passed'],(k,b)
        power=np.load(base.with_suffix('.npz'))['power_w'].mean(axis=0)
        lib['P'][k,b]=lib['cells_per_series'][k]*3*power/1e6
    lib['metadata']={**lib['metadata'],'version':'continuous_checked_v12'}
    return lib

def run(date,out_version='v12',library_factory=None):
    out=ROOT/f'results/comparison_{out_version}_{date}';out.mkdir(parents=True,exist_ok=True)
    net=make_rts_network();prof=rts_profiles(net,date);lib=(library_factory or continuous_library)(net)
    prev=ROOT/f'results/comparison_public_v3_{date}_0.970'
    net.fixed_commitment=np.load(prev/'common_commitment.npz')['commitment']
    np.savez_compressed(out/'library.npz',P=lib['P'],admissible=lib['admissible'],cells_per_series=lib['cells_per_series'])
    calls=json.loads((out/'master_calls.json').read_text()) if (out/'master_calls.json').exists() else []
    def solve(label,**kwargs):
        r=master(net,prof,lib,time_limit=45,mip_gap=2e-7,exact_heat_cut=True,**kwargs)
        calls.append(dict(label=label,status=r['status'],runtime_s=r['runtime_s'],objective=r.get('objective_actual')))
        (out/'master_calls.json').write_text(json.dumps(calls,indent=2))
        return r
    constant=load(out,'constant_unconstrained')
    if constant is None:
        constant=solve('constant_unconstrained',fixed_branch=np.zeros(6,dtype=int));save(out,'constant_unconstrained',constant)
    budget=.97*constant['allocation_t']
    lower=load(out,'economic_lower_bound')
    if lower is None:lower=solve('economic_lower_bound');save(out,'economic_lower_bound',lower)
    # Verify optimized derivative against original tensor implementation.
    ca=net.carbon(lower['pg'][:,12],lower['pr'][:,12],lower['pl'][:,12],prof['background_mw'][:,12],True)
    cb=net.carbon(lower['pg'][:,12],lower['pr'][:,12],lower['pl'][:,12],prof['background_mw'][:,12],'fast')
    err=float(abs(ca['dcarbon']-cb['dcarbon']).max());assert err<1e-9,err
    (out/'gradient_check.json').write_text(json.dumps(dict(maximum_difference=err)))
    def violation(r):return float(np.maximum(r['allocation_t']-budget,0).sum())
    def annotate(r):
        if 'pg' in r:
            r.update(budgets_t=budget,budget_violation_t=violation(r),carbon_feasible=violation(r)<=1e-4 and r['primal_audit']['passed'],
                global_finite_model_lower_bound=lower['relaxation_lower_bound'],
                finite_model_gap_percent=100*(r['objective_actual']-lower['relaxation_lower_bound'])/r['objective_actual'])
        return r
    def recover(name,current,relax=False,fixed=None,iterations=32):
        cached=load(out,name)
        if cached is not None and cached.get('carbon_feasible',False):return cached
        if cached is not None:
            save(out,name+'_attempt1',cached)
        resumed=load(out,name+'_checkpoint')
        if resumed is not None and 'pg' in resumed:current=resumed
        start=time.perf_counter();best=None;history=[];trust=.10;buffer=.001;stalled=0
        merit=lambda r:r['objective_actual']+2e4*violation(r)
        for it in range(iterations+1):
            if violation(current)<=1e-4:
                if best is None or current['objective_actual']<best['objective_actual']:best=current.copy()
                buffer=max(2e-8,buffer*.25)
            history.append(dict(iteration=it,cost=current['objective_actual'],violation_t=violation(current),trust=trust,buffer=buffer,elapsed_s=time.perf_counter()-start))
            checkpoint=current.copy();checkpoint['checkpoint_history']=history.copy()
            save(out,name+'_checkpoint',checkpoint)
            print(date,name,history[-1],flush=True)
            if it==iterations:break
            trial=solve(name+f'_{it}',center=current,budgets=budget,trust=trust,relax=relax,fixed_branch=fixed,budget_buffer=buffer)
            if 'pg' not in trial:trust*=.5;continue
            if merit(trial)<merit(current)-1e-6:
                improvement=merit(current)-merit(trial);current=trial;trust=min(.20,trust*1.3);stalled=0
                if violation(current)<=1e-4 and improvement<.005 and buffer<=2e-8:
                    if best is None or current['objective_actual']<best['objective_actual']:best=current.copy()
                    break
            else:
                trust*=.5;stalled+=1
                # Increase only the small numerical protection needed to
                # resolve a stalled near-feasible point, not a fixed 0.1% cut.
                if 0<violation(current)<.005:buffer=min(1e-6,buffer*2)
                if trust<1e-5 or stalled>=10:break
        result=best if best is not None else current
        result.update(history=history,total_runtime_s=time.perf_counter()-start,carbon_subproblem_calls=len(history)-1)
        annotate(result);save(out,name,result);return result
    def select(name,choices):
        valid=[r for r in choices if 'pg' in r and r.get('carbon_feasible',False)]
        r=min(valid,key=lambda x:x['objective_actual']).copy() if valid else min([r for r in choices if 'pg' in r],key=violation).copy()
        r['total_runtime_s']=sum(v.get('total_runtime_s',v.get('runtime_s',0)) for v in choices)
        annotate(r);save(out,name,r);return r
    # Order is declared in advance. M4 does not read M2 or M3 solutions.
    if load(out,'M4_proposed') is None:
        a=recover('M4_economic_start',lower.copy());b=recover('M4_constant_start',constant.copy())
        m4=select('M4_proposed',[a,b])
        # Two local branching radii allow a joint coordinated change of sites.
        if m4['carbon_feasible']:
            for radius in [1,2]:
                trial=solve(f'M4_neighborhood_{radius}',center=m4,budgets=budget,trust=.35,budget_buffer=2e-8,
                    branch_reference=m4['branches'],max_branch_changes=radius)
                if 'pg' in trial:
                    candidate=recover(f'M4_neighbor_recovery_{radius}',trial,iterations=12)
                    if candidate['carbon_feasible'] and candidate['objective_actual']<m4['objective_actual']:m4=candidate
            annotate(m4);save(out,'M4_proposed',m4)
    if load(out,'M4_joint_polish') is None:
        seed=load(out,'M4_proposed')
        # Resume any incomplete neighborhood stage before joint polishing.
        for radius in [1,2]:
            candidate=load(out,f'M4_neighbor_recovery_{radius}')
            if candidate is None:
                trial=solve(f'M4_resume_neighborhood_{radius}',center=seed,budgets=budget,trust=.35,budget_buffer=2e-8,
                    branch_reference=seed['branches'],max_branch_changes=radius)
                if 'pg' in trial:candidate=recover(f'M4_neighbor_recovery_{radius}',trial,iterations=12)
            if candidate is not None and candidate.get('carbon_feasible',False) and candidate['objective_actual']<seed['objective_actual']:seed=candidate
        polished=recover('M4_joint_polish',seed,iterations=32)
        if polished['carbon_feasible'] and polished['objective_actual']<seed['objective_actual']:
            polished['total_runtime_s']+=seed.get('total_runtime_s',0);save(out,'M4_proposed',polished)
    if load(out,'M1_constant') is None:
        m1=recover('M1_constant',constant.copy(),fixed=np.zeros(6,dtype=int),iterations=48)
    if load(out,'M2_EP2024') is None:
        ep,pvereport=pve_library(lib)
        (out/'PVE2024_projection.json').write_text(json.dumps(pvereport,indent=2))
        # Exact zero-tolerance projection equals the convex hull of admitted
        # trajectories. Its lifted simplex is retained for feasible disaggregation.
        root=solve('M2_convex_root',relax=True)
        raw=recover('M2_EP_convex',root,relax=True,iterations=32)
        dist=np.mean((lib['P']-raw['pl'][:,None,:])**2,axis=2);dist[~lib['admissible']]=np.inf
        nearest=np.argmin(dist,axis=1)
        point=solve('M2_nearest_root',fixed_branch=nearest)
        m2=recover('M2_EP2024',point,fixed=nearest,iterations=32)
        m2['projection_runtime_s']=sum(x['runtime_s'] for x in pvereport)
        m2['total_runtime_s']+=raw['total_runtime_s']+m2['projection_runtime_s'];save(out,'M2_EP2024',m2)
    if load(out,'M3_COPF2025') is None:
        from parametric_master_v12 import CACHE
        CACHE.clear();gc.collect()
        candidates=[]
        for label,seed in [('economic',lower),('common_feasible',load(out,'M1_constant'))]:
            raw=load(out,f'M3_COPF_raw_{label}')
            if raw is None or 'pg' not in raw:
                if raw is not None:save(out,f'M3_COPF_raw_{label}_failed_attempt',raw)
                raw=copf(net,prof,lib,seed,budget,max_seconds=240);annotate(raw);save(out,f'M3_COPF_raw_{label}',raw)
            if 'pg' not in raw:continue
            distance=np.mean((lib['P']-raw['pl'][:,None,:])**2,axis=2);distance[~lib['admissible']]=np.inf
            nearest=np.argmin(distance,axis=1);seed2=solve(f'M3_recovery_root_{label}',fixed_branch=nearest)
            CACHE.clear();gc.collect()
            r=load(out,f'M3_COPF_recovered_{label}')
            if r is None:r=copf(net,prof,lib,seed2,budget,fixed_branch=nearest,max_seconds=240)
            annotate(r);save(out,f'M3_COPF_recovered_{label}',r)
            # The same fixed-trajectory exact-carbon polishing used in EP
            # recovery prevents a solver time limit from becoming a weak straw
            # baseline. It never changes the branch selected by this method.
            root=r if r.get('carbon_feasible',False) else seed2
            polished=recover(f'M3_common_polish_{label}',root,fixed=nearest,iterations=32)
            polished['total_runtime_s']+=raw.get('runtime_s',0)+r.get('runtime_s',0)
            candidates.extend([r,polished])
            CACHE.clear();gc.collect()
        if candidates:select('M3_COPF2025',candidates)
    methods={name:{k:v for k,v in load(out,name).items() if not isinstance(v,np.ndarray)} for name in ['M1_constant','M2_EP2024','M3_COPF2025','M4_proposed'] if load(out,name) is not None}
    (out/'summary.json').write_text(json.dumps(methods,indent=2))
    print('COMPLETE',date,{k:(v.get('objective_actual'),v.get('carbon_feasible')) for k,v in methods.items()},flush=True)

if __name__=='__main__':
    # The check integrates in another process; no dispatch uses incomplete data.
    while not (ROOT/'results/execution/public_v12/summary.json').exists():time.sleep(2)
    with threadpool_limits(limits=1):
        for date in sys.argv[1:] or ['2020-04-07','2020-10-17','2020-11-29']:run(date)
