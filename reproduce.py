"""Run documented reproduction tasks; write new observations only to reproduced/."""
from pathlib import Path
import argparse, csv, hashlib, json, os, platform, shutil, sys, time

REPO=Path(__file__).resolve().parent
sys.path.insert(0,str(REPO/'src/ecf_legacy'))
from repository_paths import ROOT
DATES=['2020-04-07','2020-10-17','2020-11-29']
OUTPUT=REPO/'reproduced'
def write(name,obj):
    import numpy as np
    path=OUTPUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x)),encoding='utf8')
    return path
def csvout(name,rows):
    p=OUTPUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def verify(args):
    """Check every archived byte and every numeric NPZ array, without solver imports."""
    import numpy as np
    count=arrays=0
    with (REPO/'data/REFERENCE_MANIFEST.csv').open(encoding='utf8') as f:
        for r in csv.DictReader(f):
            p=REPO/r['path'];assert p.is_file(),p
            assert p.stat().st_size==int(r['bytes']) and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],p
            if p.suffix=='.npz':
                with np.load(p,allow_pickle=False) as data:
                    for name in data.files:
                        try: a=data[name]
                        except ValueError as exc:
                            # Legacy metadata objects are not unpickled. Numerical arrays remain readable.
                            if 'Object arrays' not in str(exc):raise
                            continue
                        arrays+=1
            count+=1
    r={'passed':True,'verified_files':count,'readable_nonobject_arrays':arrays}
    write('verification.json',r);print(json.dumps(r))
def results(args):
    """Recompute nominal-day operating totals from saved dispatch arrays."""
    import numpy as np
    from green_direct_v18 import make_case
    from dispatch_result_io import load
    rows=[]
    for date in DATES:
        net,_=make_case(date);p=load(ROOT/f'results/uncertainty_v26/{date}','scenario_00')
        for method in ['M1_constant','M2_native','M3_native','M4_proposed']:
            branch='baselines/scenario_00' if method in ['M2_native','M3_native'] else 'scenario_00'
            r=load(ROOT/f'results/robust_v28/{date}/{branch}',method)
            curtail=float((p['available_renewable_mw']-r['pr']).sum()*p['dt_h'])
            cost=float(sum(net.generation_cost(r['pg'][:,t]).sum() for t in range(48))*p['dt_h']+150*curtail)
            emissions=float(sum(net.generation_emissions(r['pg'][:,t]).sum() for t in range(48))*p['dt_h'])
            assert abs(cost-r['objective_actual'])<1e-3 and abs(emissions-r['generation_emission_t'])<1e-5
            assert abs(curtail-r['curtailment_mwh'])<1e-6
            rows.append(dict(date=date,method=method,operating_cost_usd=cost,source_emissions_t=emissions,renewable_curtailment_mwh=curtail))
    gains=[]
    for date in DATES:
        m={r['method']:r for r in rows if r['date']==date}
        gains.append(dict(date=date,**{k+'_reduction_percent':100*(m['M2_native'][k]-m['M4_proposed'][k])/m['M2_native'][k]
                       for k in ['operating_cost_usd','source_emissions_t','renewable_curtailment_mwh']}))
    csvout('tables/nominal_operation.csv',rows);csvout('tables/nominal_reductions.csv',gains)
    # Recount dense-grid classifications from stored boundaries, not stored percentages.
    regions=[]
    for r in json.loads((ROOT/'revision_v37/experiments/region_dense_summary.json').read_text()):
        b=np.asarray(r['budget_axis'])[:,None];truth=b>=np.array(r['original_boundary'])[None,:]-1e-10
        accepted=b>=np.array(r['certified_boundary'])[None,:]-1e-10
        fp=int((accepted & ~truth).sum());n=int(accepted.sum());assert fp==r['false_feasible_points']==0
        assert n==r['certified_grid_points']
        regions.append(dict(date=r['date'],chart=r['chart'],original_grid_points=int(truth.sum()),certified_grid_points=n,false_feasible_points=fp,coverage=n/int(truth.sum())))
    csvout('tables/region_grid_accuracy.csv',regions)
    timing=[]
    for p in sorted((ROOT/'revision_v39/experiments/final_benchmark').glob('*.json')):
        r=json.loads(p.read_text());timing.append({k:r.get(k) for k in ['date','scenarios','method','status','objective','bound','elapsed_s','MILP_calls','LP_calls','implementation','measurement']})
    csvout('tables/archived_algorithm_measurements.csv',timing)
    write('results_summary.json',{'passed':True,'nominal_records':len(rows),'region_charts':len(regions),'archived_timing_records':len(timing),'high_renewable_reductions':gains[-1],
        'scope':'Operating metrics recomputed from saved dispatch; region grid counts reconstructed from saved boundaries. Historical timings are observations, not new runtimes.'})
    print(json.dumps(gains[-1]))
def process(args):
    """Rerun independent DOP853/RK4 propagation and nonlinear power reconstruction."""
    import validate_process as engine
    dest=OUTPUT/'process_work';target=dest/'results/process';target.mkdir(parents=True,exist_ok=True)
    for p in (ROOT/'results/process').glob('stability_*.npz'):shutil.copyfile(p,target/p.name)
    engine.ROOT=dest;engine.main()
    r=json.loads((target/'validation.json').read_text());write('process_validation.json',r)
def region(args):
    """Recompute original nonlinear carbon equations and interval bounds on one chart."""
    import numpy as np
    from evidence_core_v37 import make_case,load,carbon_eval,enclose
    records=json.loads((ROOT/'revision_v37/experiments/region_dense_summary.json').read_text())
    record=next(r for r in records if r['date']==args.date and r['chart']==0)
    net,_=make_case(args.date);base=ROOT/f'results/robust_v28/{args.date}';common=load(base,'common')
    p=load(ROOT/f'results/uncertainty_v26/{args.date}','scenario_00')
    a,b=[load(base/'scenario_00',n) for n in record['endpoints']]
    alpha=np.linspace(0,1,101);boundary=[];residual=0.;enclosed=0;excess=-np.inf
    for x in alpha:
        allocation,res=carbon_eval(net,p,a,b,x);boundary.append(float(max(allocation/common['budget'])));residual=max(residual,res)
    np.testing.assert_allclose(boundary,np.asarray(record['original_boundary'])[::20],rtol=1e-10,atol=1e-10)
    certified=np.full(len(alpha),np.inf);cells=2**args.depth
    for j in range(cells):
        l,u=j/cells,(j+1)/cells;c=enclose(net,p,a,b,l,u)
        if c is None:continue
        enclosed+=1
        for x in np.linspace(l,u,5):
            actual,_=carbon_eval(net,p,a,b,x);excess=max(excess,float((actual-c['upper']).max()))
        mask=(alpha>=l-1e-12)&(alpha<=u+1e-12)
        certified[mask]=np.minimum(certified[mask],max(c['upper']/common['budget']))
    assert enclosed>0 and excess<1e-5
    np.savez_compressed(OUTPUT/'region_recomputed.npz',alpha=alpha,original_boundary=boundary,certified_boundary=certified)
    r=dict(passed=True,date=args.date,chart=0,depth=args.depth,grid_points=len(alpha),intervals=cells,enclosed_intervals=enclosed,
           max_carbon_balance_residual=residual,max_sample_enclosure_excess_t=excess,
           scope='One fixed complete-trajectory chart; coverage is not a volume ratio for the unrestricted nonlinear feasible region.')
    write('region_validation.json',r);print(json.dumps(r))
def ac(args):
    """Independently reconstruct AC balances/limits from all 720 saved nonlinear solutions."""
    import audit_ac_v39 as engine
    engine.dump=lambda path,obj:write('ac_validation.json',obj)
    engine.audit()
def solve(args):
    """Fresh finite-model solves; compare bounds and objectives, never impose a timing ranking."""
    import numpy as np
    from benchmark_candidate_v39 import CandidateModel
    from benchmark_joint_v39 import JointModel,FinalClusterModel
    class Comparator(JointModel):
        nccg=FinalClusterModel.nccg
    rows=[]
    for method in (['full','no_reuse','reuse'] if args.method=='all' else [args.method]):
        cls=CandidateModel if method=='reuse' else Comparator
        started=time.perf_counter();m=cls(args.date,args.regions,args.scenarios,reuse=method=='reuse')
        m.time_limit=args.time_limit;m.deadline=m.start+args.time_limit
        try:
            r=m.solve(range(m.S)) if method=='full' else m.nccg()
            row={k:r[k] for k in ['status','value','bound','u']}
        except TimeoutError:
            row=dict(status='time_limited',value=m.latest_upper,bound=m.latest_lower,u=m.latest_u)
        row.update(method=method,date=args.date,regions=args.regions,scenarios=args.scenarios,
                   elapsed_s=time.perf_counter()-started,MILP_calls=m.mips,LP_calls=m.lps,
                   outer_trace=m.trace,inner_trace=m.inner_trace,solver_events=m.solver_events)
        rows.append(row);write(f'solve_{args.date}_{args.regions}_{args.scenarios}_{method}.json',row)
        print(json.dumps({k:v for k,v in row.items() if k not in ['u','outer_trace','inner_trace','solver_events']}),flush=True)
    if len(rows)==3 and all(r['status']=='optimal' for r in rows):
        values=[r['value'] for r in rows];assert np.ptp(values)<=5e-7*max(1.,max(abs(x) for x in values))
    write('solve_comparison.json',{'runs':rows,'timing_scope':'Fresh sequential local wall time; preprocessing included. Archived manuscript times are not overwritten.'})
def figures(args):
    """Regenerate three data-derived publication figures using original plotting functions."""
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import plot_revision_v28 as main
    import plot_inset_evidence_v40 as detail
    import plot_evidence_v40 as style
    import plot_final_v40 as final
    target=OUTPUT/'figures';target.mkdir(exist_ok=True)
    main.OUT=target;style.FIG=target;final.FIG=target;detail.FIG=target
    main.formatting()
    main.costs()
    detail.region()
    final.algorithm()
    plt.close('all')
    print('Figures saved to',target)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('task',choices=['verify','results','process','region','ac','solve','figures','quick'])
    p.add_argument('--date',choices=DATES,default=DATES[0]);p.add_argument('--depth',type=int,choices=range(1,10),default=5)
    p.add_argument('--regions',type=int,choices=[1,2,4,8],default=1)
    p.add_argument('--scenarios',type=int,choices=[5,10,20,40],default=5)
    p.add_argument('--method',choices=['full','no_reuse','reuse','all'],default='all')
    p.add_argument('--time-limit',type=float,default=600.)
    args=p.parse_args();OUTPUT.mkdir(exist_ok=True)
    from threadpoolctl import threadpool_limits
    start=time.perf_counter()
    with threadpool_limits(limits=1):
        for task in ['verify','results','process','region'] if args.task=='quick' else [args.task]:globals()[task](args)
    write(f'run_{args.task}.json',dict(task=args.task,elapsed_s=time.perf_counter()-start,python=sys.version,platform=platform.platform(),arguments=vars(args)))
if __name__=='__main__': main()
