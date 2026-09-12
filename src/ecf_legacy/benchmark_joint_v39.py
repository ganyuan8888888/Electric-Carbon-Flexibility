"""Nested distinct regional uncertainty combinations, with exact cut coverage.

Each regional source/load realization is one original complete-day trajectory;
different regions may simultaneously experience different realizations. The
prescribed combination sequence is fixed before timing or inspecting rankings.
"""
from benchmark_screened_v39 import replace_cells
from benchmark_final_v38 import *
import itertools

class JointModel(FinalClusterModel):
 def __init__(self,date,groups,count,reuse=True):
  super().__init__(date,groups,reuse)
  originals=[self.cells[s][0] for s in range(5)]
  first=[tuple((s+g)%5 for g in range(groups)) for s in range(5)]
  rest=[t for t in itertools.product(range(5),repeat=groups) if t not in first]
  rng=np.random.default_rng(2026091139);rng.shuffle(rest);self.combinations=(first+rest)[:count]
  assert len(self.combinations)==count;self.S=count
  replace_cells(self,[[originals[c[g]] for g in range(groups)] for c in self.combinations])

 def nccg(self):
  start=time.perf_counter();incumbent,upper=self.initial();lower=-np.inf;active=[0]
  self.latest_upper=upper;self.latest_u=incumbent.copy()
  for it in range(1,self.S+3):
   master=self.solve(active);lower=max(lower,master['bound']);self.latest_lower=lower
   if len(active)==self.S:
    if master['value']<upper:upper=master['value'];incumbent=master['u'].copy()
    self.latest_upper=upper;self.latest_u=incumbent.copy()
    return dict(value=upper,bound=lower,u=incumbent,status='optimal' if upper-lower<=2e-8*max(1.,abs(upper)) else 'time_limited')
   missing={}
   for s in range(self.S):
    pairs={(g,self.combinations[s][g]) for g,ids in enumerate(self.eligible(s,master['u'])) if len(ids)==0}
    if pairs:missing[s]=pairs
   if missing:
    assert not set(missing).intersection(active);uncovered=set.union(*missing.values());added=[]
    while uncovered:
     s=max(sorted(missing),key=lambda j:len(missing[j]&uncovered));covered=missing[s]&uncovered
     assert covered;added.append(s);uncovered-=covered
    active.extend(added)
    self.trace.append(dict(iteration=it,lower=lower,upper=upper,active=list(active),feasibility_scenarios=added,
     screened_missing_scenarios=len(missing),elapsed_s=time.perf_counter()-start));continue
   self.master_keys=list(master['key'].values());inner=self.inner(master['u'],it);assert inner['infeasible'] is None
   candidate=max(master['value'],inner['upper'])
   if candidate<upper:upper=candidate;incumbent=master['u'].copy()
   self.latest_upper=upper;self.latest_u=incumbent.copy()
   self.trace.append(dict(iteration=it,lower=lower,upper=upper,active=list(active),worst_scenario=inner['worst'],elapsed_s=time.perf_counter()-start))
   if upper-lower<=2e-8*max(1.,abs(upper)):return dict(value=upper,bound=lower,u=incumbent,status='optimal')
   s=inner['worst'];assert s not in active;active.append(s)
  raise RuntimeError('Outer iteration cap')

def protocol():
 obj=dict(dates=DATES,regions=4,scenarios=[5,10,20,40],seed=2026091139,
  interpretation='Distinct regional combinations of original 24-hour joint source-load realizations; 4 physical regions, unchanged per-region grid and industrial trajectories. These are designed uncertainty scenarios, not new observed days.',
  methods=['full','no_reuse','reuse'],threads=1,time_limit_s=600,mip_rel_gap=1e-8,mip_abs_gap=1e-9,
  scope='Same model, inputs, quota cuts, solver tolerances and timing scope. NCCGs add a deterministic greedy cover of missing local realizations instead of every redundant global feasibility scenario. Both NCCGs use this exact feasibility-screening improvement.',
  timing='One calibration run precedes the full preset suite. Formal runs rotate method order, retain all instances and all timeouts. No outcome-based date/size filtering.')
 path=ROOT/'revision_v39/experiments/joint_protocol.json'
 if path.exists():assert json.loads(path.read_text())==obj
 else:dump(path,obj)

def run(date,count,method,phase='calibration'):
 protocol();target=ROOT/f'revision_v39/experiments/joint_{phase}/{date}_{count}_{method}.json'
 if target.exists():print('EXISTS',target);return
 start=time.perf_counter();m=JointModel(date,4,count,reuse=method!='no_reuse')
 try:
  r=m.solve(range(m.S)) if method=='full' else m.nccg()
  row=dict(status=r['status'],objective=r['value'],bound=r['bound'],u=r['u'])
 except Exception as e:
  row=dict(status='unresolved',exception=type(e).__name__+': '+str(e))
  if np.isfinite(m.latest_upper):row.update(objective=m.latest_upper,bound=m.latest_lower,u=m.latest_u)
 row.update(date=date,regions=4,scenarios=count,method=method,elapsed_s=time.perf_counter()-start,combinations=m.combinations,
  MILP_calls=m.mips,LP_calls=m.lps,solver_events=m.solver_events,outer_trace=m.trace,inner_trace=m.inner_trace)
 dump(target,row);print('JOINT_RESULT',{k:v for k,v in row.items() if k not in ['u','combinations','solver_events','outer_trace','inner_trace']},flush=True)

if __name__=='__main__':
 with threadpool_limits(limits=1):run(sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4] if len(sys.argv)>4 else 'calibration')
