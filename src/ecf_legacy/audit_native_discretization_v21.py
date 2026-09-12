"""Restore the explicit two-state difference scheme of Liu Eqs.11/14."""
from pathlib import Path
import sys,json
ROOT=__import__("repository_paths").ROOT
import numpy as np
from threadpoolctl import threadpool_limits
import native_industrial_v14 as native
from green_direct_v18 import make_case
from dispatch_result_io import load,save

original_parameters=native.parameters
def source_parameters(net):
    cells,rows=original_parameters(net)
    for row in rows:
        C,Cs,G,H=[row[k] for k in ['liquid_capacity_j_k','solid_capacity_j_k','liquid_solid_conductance_w_k','solid_ambient_conductance_w_k']]
        A=np.array([[-G/C,G/C],[G/Cs,-(G+H)/Cs]])
        b=np.array([row['net_heat_w']/180/C,0.])
        row['Ad']=(np.eye(2)+1800*A).tolist();row['Bd']=(1800*b).tolist()
        row['discretization']='Explicit two-state difference equations, Liu Eqs.11 and14, common 30-minute interval.'
        assert np.max(abs(np.linalg.eigvals(row['Ad'])))<1
    return cells,rows

def run(date):
    net,p=make_case(date);out=ROOT/f'results/integrated_v21_direct_{date}';out.mkdir(exist_ok=True)
    result=load(out,'M2_source_difference')
    if result is None:
        native.parameters=source_parameters
        result=native.run(date,25.,12,240,False,False,net,p,out)
        result.update(source_equations=['11','14','34','35','36','37','39','40'],
            source_difference_scheme=True,proposed_features_used=[],
            scope='Original explicit two-state thermal difference model in the shared case; no periodic excitation, modal constraint, full-trajectory library or terminal restoration controller.')
        save(out,'M2_source_difference',result)
    old=load(ROOT/f'results/integrated_v20_direct_{date}','M2_native')
    keys=['objective_actual','generation_emission_t','curtailment_mwh']
    report=dict(date=date,before={k:old[k] for k in keys},after={k:result[k] for k in keys},
        reason='Original Eqs.11/14 use an explicit finite difference; prior adaptation used a matrix exponential.',
        same_equipment_and_case=True,proposed_features_used=[])
    (out/'M2_discretization_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):
        for date in ['2020-11-29','2020-04-07','2020-10-17']:run(date)
