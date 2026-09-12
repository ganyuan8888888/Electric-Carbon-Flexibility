"""Command-count sensitivity with the original M2 result retained unchanged."""
from pathlib import Path
import sys,json,gc
ROOT=__import__("repository_paths").ROOT
import numpy as np
from threadpoolctl import threadpool_limits
from dispatch_result_io import load,save
from green_direct_v18 import make_case
import native_industrial_v14 as native
from audit_native_discretization_v21 import source_parameters

def run(date,scenarios=(0,),limits=(12,8,4)):
    old=ROOT/f'results/robust_v27/{date}'
    out=ROOT/f'results/m2_configuration_v28/{date}'
    common=load(old,'forecast_common_reference')
    net,_=make_case(date);net.fixed_commitment=common['commitment']
    native.parameters=source_parameters
    records=[]
    for sid in scenarios:
        p=load(ROOT/f'results/uncertainty_v26/{date}',f'scenario_{sid:02}')
        for limit in limits:
            dest=out/f'commands_{limit:02}/scenario_{sid:02}'
            dest.mkdir(parents=True,exist_ok=True)
            r=load(dest,'M2_native')
            if r is None:
                if limit==12:
                    r=load(old/f'baselines/scenario_{sid:02}','M2_native')
                    assert r['command_limit']==12
                else:
                    r=native.run(date,carbon_price=25.,change_limit=limit,time_limit=240,
                        net=net,prof=p,output=dest,use_provided_commitment=True)
                r=dict(r)
                r['configuration_note']=f'Original Multi-time-scale dispatch model and name retained. Daily command-count cap {limit}; reference cap 12 from Liu et al. Table 1 retained in the same sensitivity experiment. All other case inputs unchanged.'
                assert r['primal_audit']['passed'] and r['linear_constraint_violation']<1e-5
                save(dest,'M2_native',r)
            record={k:r[k] for k in ['status','objective_actual','generation_emission_t','curtailment_mwh','mip_gap','runtime_s','linear_constraint_violation']}
            record.update(date=date,scenario=sid,command_limit=limit,
                actual_changes=np.sum(abs(np.diff(np.c_[np.full(6,180.),r['current_ka']],axis=1))>1e-5,axis=1).tolist())
            records.append(record)
            print(record,flush=True);gc.collect()
    summary=out/'configuration_summary.json'
    prior=json.loads(summary.read_text()) if summary.exists() else []
    combined={(r['scenario'],r['command_limit']):r for r in prior+records}
    summary.write_text(json.dumps(list(combined.values()),indent=2),encoding='utf8')

if __name__=='__main__':
    with threadpool_limits(limits=1):
        run(sys.argv[1],range(5) if '--all-scenarios' in sys.argv else (0,),
            (4,) if '--four-only' in sys.argv else (12,8,4))
