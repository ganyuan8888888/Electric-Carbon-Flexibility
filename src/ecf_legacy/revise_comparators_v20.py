"""Source-equation audit and fresh comparisons under the same physical case.

M2: Liu-2025 temperature-model day-ahead specialization, no added modal bound.
M3: Chen C-ED example (Eq.27): prescribed loads, generation-cost objective,
native endogenous carbon balances. No borrowed Liu thermal controller.
M1: fixed-industrial-control ablation of M4, same coordinating objective.
M4: verified dynamic library; normalization fixed by a common carbon-priced
constant-production plan before looking at any comparator result.
"""
from pathlib import Path
import sys,json,gc,time
ROOT=__import__("repository_paths").ROOT
import numpy as np
from threadpoolctl import threadpool_limits
from dispatch_result_io import load,save
from green_direct_v18 import make_case,recovery_metrics
from priced_library_v14 import priced_library
from proposed_library_v20 import library as proposed_library
from native_industrial_v14 import run as native_run
from run_integrated_case_v18 import search,KEYS
import parametric_master_v13 as economic
import balanced_master_v16 as balanced

DATES=['2020-04-07','2020-10-17','2020-11-29']
def clear():economic.CACHE.clear();balanced.CACHE.clear();gc.collect()
def setup(date):
    net,p=make_case(date);old=ROOT/f'results/integrated_v18_direct_{date}'
    out=ROOT/f'results/integrated_v20_direct_{date}';out.mkdir(exist_ok=True)
    original=load(old,'constant_reference');net.fixed_commitment=original['commitment']
    base={'P':original['pl'][:,None,:],'admissible':np.ones((6,1),bool),'metadata':{'version':'fixed-production-input-v20'}}
    ref=load(out,'constant_reference')
    if ref is None:
        ref=economic.master(net,p,base,emission_price=25.,time_limit=120,mip_gap=1e-8)
        assert ref['primal_audit']['passed'];save(out,'constant_reference',ref)
        save(out,'constant_economic_reference',original)
    refs=np.array([max(ref[k],1. if k=='curtailment_mwh' else 1e-6) for k in KEYS])
    budget=.97*original['allocation_t']
    lib=proposed_library(net)
    before=lib['admissible'].copy()
    plus=np.maximum(lib['P']-original['pl'][:,None,:],0)[:,:,p['time_h']>=20].sum(axis=2)*p['dt_h']
    lib['admissible'] &= plus<=(.1*original['pl'][:,0]*4)[:,None]+1e-8
    save(out,'library_geometry',dict(P=lib['P'],p0=original['pl'],admissible_before=before,admissible_after=lib['admissible'],recovery_plus=plus))
    return net,p,old,out,base,lib,refs,budget

def run(date,stage):
    net,p,old,out,base,lib,refs,budget=setup(date);clear()
    if stage in ['pilot','all','proposed']:
        raw=load(out,'M4_unconstrained')
        if raw is not None and raw['weights'].shape!=lib['P'].shape[:2]:
            save(out,'M4_unconstrained_original_library',raw);raw=None
        if raw is None:
            raw=balanced.master(net,p,lib,refs,time_limit=120,mip_gap=1e-7);save(out,'M4_unconstrained',raw)
        print(date,'normalization',refs,'raw',[(k,raw[k]) for k in KEYS],flush=True)
        if stage=='pilot':return
        m1=search(net,p,base,out,'M1_constant',[load(old,'M1_constant'),load(out,'constant_reference')],budget,refs=refs,fixed=np.zeros(6,int))
        clear()
        start=m1.copy();start['weights']=np.zeros(lib['P'].shape[:2]);start['weights'][:,0]=1
        m4=search(net,p,lib,out,'M4_proposed',[raw,start],budget,refs=refs)
        save(out,'M1_recovery',recovery_metrics(net,p,m1,load(out,'constant_reference')))
        save(out,'M4_recovery',recovery_metrics(net,p,m4,load(out,'constant_reference')))
        clear()
    if stage in ['native','all']:
        if load(out,'M2_native') is None:
            m2=native_run(date,25.,12,240,False,False,net,p,out)
            m2['scope']='Liu-2025 temperature-model day-ahead specialization in the common RTS case; no added modal envelope or terminal restoration.'
            save(out,'M2_native',m2)
        m2=load(out,'M2_native');print(date,'M2',[(k,m2[k]) for k in KEYS],flush=True)
        clear()
        if load(out,'M3_native') is None:
            from recent_methods_v12 import copf
            candidates=[]
            # Both starts are independently produced fixed-load schedules.
            for seed,start in enumerate([load(old,'M1_constant'),load(out,'constant_reference')]):
                r=load(out,f'M3_start{seed}')
                if r is None:
                    r=copf(net,p,base,start,budget,fixed_branch=np.zeros(6,int),source_price=0.,max_seconds=240,max_iter=1800,native_solver_profile='lowmemory')
                    r.update(added_terminal_recovery=False,added_industrial_thermal_model=False,added_modal_envelope=False,
                        scope='Chen C-ED Eq.27 fixed-load and generation-cost specialization; same RTS system; original cumulative carbon-cap form Eq.15.')
                    save(out,f'M3_start{seed}',r)
                if r.get('carbon_feasible') and r.get('native_nlp_constraint_violation',1)<1e-5:candidates.append(r)
                clear()
            if not candidates:raise RuntimeError('C-ED has no checked feasible iterate; keep failed records and resolve rather than rank it.')
            save(out,'M3_native',min(candidates,key=lambda x:x['objective_actual']))
        for name in ['M2','M3']:
            r=load(out,name+'_native');save(out,name+'_recovery',recovery_metrics(net,p,r,load(out,'constant_reference')))
        print(date,'NATIVE DONE',flush=True)
    clear()
if __name__=='__main__':
    with threadpool_limits(limits=1):
        for date in ([sys.argv[1]] if len(sys.argv)>1 and sys.argv[1]!='all' else DATES):run(date,sys.argv[2] if len(sys.argv)>2 else 'pilot')
