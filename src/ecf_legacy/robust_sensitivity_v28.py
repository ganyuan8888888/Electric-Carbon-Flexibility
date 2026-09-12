"""Nested uncertainty-set study using the same frozen recourse cells."""
from pathlib import Path
import sys,json
ROOT=__import__("repository_paths").ROOT
from dispatch_result_io import load,save
from nccg_solver_v26 import FiniteRecourse

def run(date):
    model=FiniteRecourse(date,result_root=ROOT/'results/robust_v28')
    manifest=json.loads((ROOT/f'results/uncertainty_v26/{date}/manifest.json').read_text())
    order=sorted(range(5),key=lambda s:manifest['scenarios'][s]['normalized_error_budget'])
    rows=[]
    for count in [1,3,5]:
        S=order[:count];primary=model.solve(S)
        r=model.solve(S,robust_cap=primary['value']+1e-9*max(1,abs(primary['value'])))
        checks=[model.solve([s],r['u']) for s in range(5)]
        rows.append(dict(scenarios=S,scenario_count=count,gamma_fraction=max(manifest['scenarios'][s]['normalized_error_budget'] for s in S)/manifest['gamma'],
            objective_ratio=primary['value'],lower_bound_ratio=primary['bound'],q_t=r['u']*model.B,
            recourse_feasible_scenarios=sum(c['value']<float('inf') for c in checks),runtime_s=r['runtime_s']))
    save(model.path,'uncertainty_set_sensitivity',dict(rows=rows,
        scope='Nested subsets of the five frozen joint trajectories. Recourse coverage concerns only this certified finite model; not a statistical out-of-sample guarantee.'))
    fixed=FiniteRecourse(date,allowed={'constant'},result_root=ROOT/'results/robust_v28');r=fixed.solve([0])
    if 'u' in r:
        save(model.path,'constant_nominal',dict(q=r['u']*model.B,u=r['u'],scenarios={0:r},
            scope='Nominal-only constant-control solution, separately labelled when no five-scenario constant-control certificate is available.'))
    print(date,'nested uncertainty study',[(v['scenario_count'],v['recourse_feasible_scenarios']) for v in rows],flush=True)

if __name__=='__main__':run(sys.argv[1])
