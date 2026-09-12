"""Independent primal checks from saved dispatch arrays, without solver status."""
from pathlib import Path
import json
import numpy as np
from rts_network import make_rts_network,rts_profiles
from branch_dispatch import load_library,evaluate
ROOT=__import__("repository_paths").ROOT

def primal_audit(net,prof,lib,r,require_integral=True,tolerance=1e-5):
    pg,pr,pl,w=[r[k] for k in ['pg','pr','pl','weights']]
    on=np.asarray(r['commitment'])
    inj=net.genmap@pg+net.renmap@pr-net.loadmap@pl-prof['background_mw']
    flow=net.ptdf@inj+net.flow_offset[:,None]
    reconstructed=np.einsum('kb,kbt->kt',w,lib['P'])
    pos=lambda x:float(np.maximum(0,np.asarray(x)).max(initial=0))
    checks={
      'total_power_balance_mw':float(abs(inj.sum(axis=0)).max()),
      'nodal_dc_balance_mw':float(abs(net.incidence@flow-inj).max()),
      'line_capacity_excess_mw':pos(abs(flow)-net.capacity[:,None]),
      'generation_lower_excess_mw':pos(net.pmin[:,None]*on[:,None]-pg),
      'generation_upper_excess_mw':pos(pg-net.pmax[:,None]*on[:,None]),
      'generation_ramp_excess_mw':pos(abs(np.diff(pg,axis=1))-net.ramp_mw_per_hour[:,None]*prof['dt_h']),
      'renewable_lower_excess_mw':pos(prof['minimum_renewable_mw']-pr),
      'renewable_upper_excess_mw':pos(pr-prof['available_renewable_mw']),
      'library_interface_error_mw':float(abs(pl-reconstructed).max()),
      'weight_sum_error':float(abs(w.sum(axis=1)-1).max()),
      'weight_bound_excess':max(pos(-w),pos(w-1)),
      'inadmissible_branch_weight':float(abs(w[~lib['admissible']]).max(initial=0)),
      'commitment_integrality_error':float(abs(on-np.rint(on)).max()),
    }
    fractionality=float(abs(w-np.rint(w)).max())
    if require_integral:checks['branch_integrality_error']=fractionality
    passed=all(np.isfinite(x) and x<=tolerance for x in checks.values())
    return dict(passed=bool(passed),tolerance=tolerance,checks=checks,
                integral_required=require_integral,weight_fractionality=fractionality)

def main():
    net=make_rts_network();lib=load_library(net,version='public_v3');reports=[]
    for date in ['2020-04-07','2020-10-17','2020-11-29']:
        directory=ROOT/f'results/comparison_public_v3_{date}_0.970';prof=rts_profiles(net,date)
        for name in ['constant','M1_no_carbon','M2_fixed_factor','M3_convex_original','M3_convex_recovered','M4_proposed']:
            meta=json.loads((directory/f'{name}.json').read_text(encoding='utf8'))
            r={**meta,**dict(np.load(directory/f'{name}.npz'))}
            if 'pg' not in r:
                reports.append(dict(date=date,method=name,status=r['status'],primal_available=False));continue
            result=primal_audit(net,prof,lib,r,require_integral=name!='M3_convex_original')
            ev=evaluate(net,prof,r['pg'],r['pr'],r['pl'])
            result.update(date=date,method=name,objective_recomputed=ev['objective_actual'],
                          objective_difference=abs(ev['objective_actual']-r['objective_actual']),
                          carbon_balance_residual=ev['carbon_balance_residual'],
                          attribution_conservation_residual=ev['attribution_conservation_residual'])
            if 'budgets_t' in r:
                result['budget_violation_t']=float(np.maximum(ev['allocation_t']-r['budgets_t'],0).sum())
                result['budget_passed']=result['budget_violation_t']<=1e-4
            reports.append(result)
            (directory/f'{name}_independent_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
            print(date,name,result['passed'],max(result['checks'].values()),result.get('budget_violation_t'),flush=True)
    (ROOT/'results/public_v3_independent_dispatch_audit.json').write_text(json.dumps(reports,indent=2),encoding='utf8')

if __name__=='__main__':main()
