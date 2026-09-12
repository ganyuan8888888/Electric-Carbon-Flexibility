"""Extend resolution to the existing production certification depth nine."""
from evidence_core_v37 import *

def run():
 rows=json.loads((RESULT/'region_summary.json').read_text())
 dump(RESULT/'region_resolution_extension.json',{'reason':'Compare the coarse-depth study with the existing chart_cells production limit max_depth=9.',
    'additional_depths':[7,8,9],'all_charts_retained':True,'interior_checks':'Five points in eight equally spaced cells per added depth, plus the independent 201-point original-constraint grid.'})
 for row in rows:
  d=row['date'];j=row['chart'];f=RESULT/'region_refined'/f'{d}_{j}.json'
  if f.exists():continue
  net,_=make_case(d);base=ROOT/'results/robust_v28'/d;B=load(base,'common')['budget'];p=load(ROOT/f'results/uncertainty_v26/{d}','scenario_00')
  a,b=[load(base/'scenario_00',x) for x in row['endpoints']];alpha=np.array(row['alpha']);budget=np.array(row['budget_axis']);truth=np.array(row['original_boundary']);actual=budget[:,None]>=truth[None,:]-1e-10
  for depth in [7,8,9]:
   boundary=np.full(len(alpha),np.inf);valid=0;extra=-np.inf;checkset=set(np.linspace(0,2**depth-1,8,dtype=int))
   for i in range(2**depth):
    l=i/2**depth;r=(i+1)/2**depth;c=enclose(net,p,a,b,l,r)
    if c is None:continue
    valid+=1;mask=(alpha>=l-1e-12)&(alpha<=r+1e-12);boundary[mask]=np.minimum(boundary[mask],max(c['upper']/B))
    if i in checkset:
     for x in [l,l+(r-l)*.2113248654,(l+r)/2,l+(r-l)*.7886751346,r]:
      v,_=carbon_eval(net,p,a,b,x);extra=max(extra,float(max(v-c['upper'])))
   accepted=budget[:,None]>=boundary[None,:]-1e-10;fp=int((accepted&~actual).sum());assert fp==0 and extra<1e-5
   gap=boundary-truth;finite=np.isfinite(gap)
   row['levels'].append({'depth':depth,'cells':2**depth,'enclosed_cells':valid,'enclosed_length':valid/2**depth,
     'coverage':float((accepted&actual).sum()/actual.sum()) if actual.any() else None,'false_feasible_points':fp,'certified_grid_points':int(accepted.sum()),
     'mean_budget_gap':float(gap[finite].mean()) if finite.any() else None,'max_budget_gap':float(gap[finite].max()) if finite.any() else None,
     'grid101_coverage':float((accepted&actual)[::2,::2].sum()/actual[::2,::2].sum()) if actual[::2,::2].any() else None,'boundary':boundary.tolist(),
     'max_extra_sample_excess_t':extra})
   print('REFINE',d,j,depth,row['levels'][-1]['coverage'],flush=True)
  dump(f,row)
 combined=[json.loads((RESULT/'region_refined'/f"{r['date']}_{r['chart']}.json").read_text()) for r in rows]
 dump(RESULT/'region_refined_summary.json',combined)

if __name__=='__main__':
 with threadpool_limits(limits=1):run()
