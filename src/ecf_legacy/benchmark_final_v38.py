"""Final common implementation: reuse certified master witnesses and exact stopping.

All variants retain the same finite feasible set and objective. No timing padding
or outcome-dependent scenario selection is used. Calibration outputs are retained.
"""
import benchmark_clusters_v38 as engine
from benchmark_presolved_v38 import *

class FinalClusterModel(PresolvedClusterModel):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.latest_lower=-np.inf;self.latest_upper=np.inf;self.latest_u=None

 def inner(self,u,iteration):
  if not self.reuse:self.pool=[];self.cache={}
  for key in self.master_keys:
   if key not in self.pool:self.pool.append(key)
  # Parent clears the pool for the no-reuse variant. It has already been cleared
  # above; both variants now have the current master's valid recourse witnesses.
  flag=self.reuse;self.reuse=True
  try:return super().inner(u,iteration)
  finally:self.reuse=flag

 def nccg(self):
  start=time.perf_counter();incumbent,upper=self.initial();lower=-np.inf;active=[0]
  self.latest_upper=upper;self.latest_u=incumbent.copy()
  for it in range(1,self.S+3):
   master=self.solve(active);lower=max(lower,master['bound']);self.latest_lower=lower
   if len(active)==self.S:
    # A feasible solution of the full scenario master is already an all-scenario
    # witness. Re-solving all five recourse MILPs cannot strengthen this proof.
    if master['value']<upper:upper=master['value'];incumbent=master['u'].copy()
    self.latest_upper=upper;self.latest_u=incumbent.copy()
    closed=upper-lower<=2e-8*max(1.,abs(upper))
    self.trace.append(dict(iteration=it,lower=lower,upper=upper,active=list(active),full_master_certificate=True,elapsed_s=time.perf_counter()-start))
    return dict(value=upper,bound=lower,u=incumbent,status='optimal' if closed else 'time_limited')
   missing=[s for s in range(self.S) if any(not len(ix) for ix in self.eligible(s,master['u']))]
   if missing:
    assert not set(missing).intersection(active)
    self.trace.append(dict(iteration=it,lower=lower,upper=upper,active=list(active),feasibility_scenarios=missing,elapsed_s=time.perf_counter()-start))
    active.extend(missing);continue
   self.master_keys=list(master['key'].values());inner=self.inner(master['u'],it)
   assert inner['infeasible'] is None
   candidate=max(master['value'],inner['upper'])
   if candidate<upper:upper=candidate;incumbent=master['u'].copy()
   self.latest_upper=upper;self.latest_u=incumbent.copy()
   self.trace.append(dict(iteration=it,lower=lower,upper=upper,active=list(active),worst_scenario=inner['worst'],elapsed_s=time.perf_counter()-start))
   if upper-lower<=2e-8*max(1.,abs(upper)):return dict(value=upper,bound=lower,u=incumbent,status='optimal')
   s=inner['worst'];assert s not in active;active.append(s)
  raise RuntimeError('Outer iteration cap')

def run(date,groups,method,repeat=0):
 target=ROOT/f'revision_v38/experiments/clusters_final/{date}_{groups}_{method}_{repeat}.json'
 if target.exists():print('EXISTS',target,flush=True);return
 gc.collect();start=time.perf_counter();m=FinalClusterModel(date,groups,reuse=method!='no_reuse')
 try:
  result=m.solve(range(m.S)) if method=='full' else m.nccg()
  row=dict(status=result['status'],objective=result['value'],bound=result['bound'],gap=max(0.,result['value']-result['bound']),u=result['u'])
 except Exception as e:
  row=dict(status='unresolved',exception=type(e).__name__+': '+str(e))
  if np.isfinite(m.latest_upper):row.update(objective=m.latest_upper,bound=m.latest_lower,gap=max(0.,m.latest_upper-m.latest_lower),u=m.latest_u)
 row.update(date=date,regions=groups,parks=groups*6,buses=groups*79,branches=groups*126,scenarios=m.S,method=method,repeat=repeat,
   elapsed_s=time.perf_counter()-start,cells=sum(len(c) for block in m.cells for c in block),MILP_calls=m.mips,LP_calls=m.lps,
   cache_hits=m.cache_hits,outer_trace=m.trace,inner_trace=m.inner_trace,solver_events=m.solver_events)
 dump(target,row);print('FINAL RESULT',{k:v for k,v in row.items() if k not in ['u','outer_trace','inner_trace','solver_events']},flush=True)

if __name__=='__main__':
 with threadpool_limits(limits=1):run(sys.argv[1],int(sys.argv[2]),sys.argv[3],int(sys.argv[4]) if len(sys.argv)>4 else 0)
