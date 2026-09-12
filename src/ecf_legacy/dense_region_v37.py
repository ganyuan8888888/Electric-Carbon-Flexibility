"""Refine the reference grid uniformly; retain the same seven charts and certificates."""
from evidence_core_v37 import *

def run():
 rows=json.loads((RESULT/'region_refined_summary.json').read_text());output=[]
 dump(RESULT/'dense_grid_protocol.json',{'reason':'Resolve thin feasible sections and assess grid sensitivity uniformly on all seven existing charts. No chart, carbon-budget axis, or certificate rule is changed.','alpha_points':2001,'budget_points':2001,'depth':9,'comparison_grid':1001})
 for row in rows:
  date,index=row['date'],row['chart'];path=RESULT/'region_dense'/f'{date}_{index}.json'
  if path.exists():output.append(json.loads(path.read_text()));continue
  start=time.perf_counter();net,_=make_case(date);base=ROOT/'results/robust_v28'/date;common=load(base,'common');B=common['budget']
  p=load(ROOT/f'results/uncertainty_v26/{date}','scenario_00');a,b=[load(base/'scenario_00',name) for name in row['endpoints']]
  alpha=np.linspace(0,1,2001);budget=np.linspace(.8,1.2,2001);truth=[];res=0.
  for x in alpha:
   v,r=carbon_eval(net,p,a,b,x);truth.append(max(v/B));res=max(res,r)
  certified=np.full(len(alpha),np.inf);nc=0
  for j in range(512):
   c=enclose(net,p,a,b,j/512,(j+1)/512)
   if c is None:continue
   nc+=1;mask=(alpha>=j/512-1e-12)&(alpha<=(j+1)/512+1e-12)
   certified[mask]=np.minimum(certified[mask],max(c['upper']/B))
  truth=np.array(truth);actual=budget[:,None]>=truth[None,:]-1e-10;accepted=budget[:,None]>=certified[None,:]-1e-10
  fp=int((accepted&~actual).sum());assert fp==0
  gap=certified-truth;gap=gap[np.isfinite(gap)]
  result={'date':date,'chart':index,'endpoints':row['endpoints'],'alpha':alpha,'budget_axis':budget,'original_boundary':truth,'certified_boundary':certified,
   'original_grid_points':int(actual.sum()),'certified_grid_points':int(accepted.sum()),'false_feasible_points':fp,
   'coverage':float(accepted.sum()/actual.sum()),'grid1001_coverage':float(accepted[::2,::2].sum()/actual[::2,::2].sum()),
   'mean_budget_gap':float(gap.mean()),'max_budget_gap':float(gap.max()),'enclosed_cells':nc,'max_carbon_balance_residual':res,'elapsed_s':time.perf_counter()-start}
  dump(path,result);output.append(result);print('DENSE REGION',date,index,result['coverage'],result['grid1001_coverage'],flush=True)
 dump(RESULT/'region_dense_summary.json',output)

if __name__=='__main__':
 with threadpool_limits(limits=1):run()
