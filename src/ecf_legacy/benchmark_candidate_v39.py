"""NCCG primal initialization with a small candidate-only mixed-integer master."""
from benchmark_warm_v39 import *
from benchmark_screened_v39 import replace_cells

class CandidateModel(WarmModel):
 def initial(self):
  relaxation=self.scenario_relaxations;self.scenario_relaxations=lambda:None
  try:u,upper=super().initial()
  finally:self.scenario_relaxations=relaxation
  self.scenario_relaxations();original=self.cells
  if upper-max(self.relaxation_info['values'])<=2e-8*max(1.,abs(upper)):return u,upper
  chosen=[]
  for s in range(self.S):
   values=[self.fixed_lp(s,u,key) for key in self.pool];chosen.append(self.pool[int(np.argmin(values))])
  candidate=[]
  for s in range(self.S):
   block=[]
   for g in range(self.groups):
    weights=self.seed_weights[s][g];ids=list(np.argsort(-weights,kind='stable')[:8]);i=self.maps[s][g][chosen[s][g]]
    if i not in ids:ids.append(i)
    block.append([original[s][g][int(i)] for i in ids])
   candidate.append(block)
  start=time.perf_counter();replace_cells(self,candidate)
  # This master only improves a feasible upper bound. Its dual bound is never
  # used as a lower bound for the full cell library.
  saved_u=self.latest_u;self.latest_u=None;saved_limit=self.time_limit;self.time_limit=30.
  try:result=self.solve(range(self.S));self.solver_events[-1]['phase']='candidate_only_primal_initialization'
  finally:replace_cells(self,original);self.latest_u=saved_u;self.time_limit=saved_limit
  if result['value']<upper:
   u=result['u'].copy();upper=result['value'];self.pool=list(dict.fromkeys(self.pool+list(result['key'].values())))
  self.latest_upper=upper;self.latest_u=u.copy()
  self.initial_polish['candidate_master']=dict(upper=upper,elapsed_s=time.perf_counter()-start,cells=sum(len(c) for block in candidate for c in block))
  print('PRIMAL_CANDIDATE',self.initial_polish['candidate_master'],flush=True)
  return u,upper

 def scenario_relaxations(self):
  start=time.perf_counter();K=len(self.B);weights=[[np.zeros(len(self.cells[s][g])) for g in range(self.groups)] for s in range(self.S)]
  # Minima of separate emission/curtailment terms are valid lower bounds even
  # when their minimizing cells cannot be combined. They certify no upper bound.
  values=np.array([max(0.,max(sum(np.minimum(lo[:,k],(lo+d)[:,k]).min() for lo,d,up in self.arr[s]) for k in [1,2])) for s in range(self.S)])
  cheap=values.copy();visited=[]
  for s in np.argsort(-cheap,kind='stable'):
   if self.latest_upper-values.max()<=2e-8*max(1.,abs(self.latest_upper)):break
   s=int(s);u=cp.Variable(K);tr=cp.Variable(K);eta=cp.Variable(nonneg=True)
   cons=[u>=-1,u<=self.qbar,tr>=cp.multiply(25*self.B/self.refs[0],u),tr>=cp.multiply(20*self.B/self.refs[0],u)]
   A,b=self.quota_cuts;cons.append(A@u>=b);metric=cp.Constant(np.zeros(3))
   zvars=[]
   for g in range(self.groups):
    lo,d,up=self.arr[s][g];z=cp.Variable(len(lo),nonneg=True);a=cp.Variable(len(lo),nonneg=True)
    zvars.append(z)
    cons.extend([cp.sum(z)==1,a<=z,up.T@z<=1+u[6*g:6*(g+1)]])
    metric+=lo.T@z+d.T@a
   cons.extend([metric[0]+cp.sum(tr)<=eta,metric[1]<=eta,metric[2]<=eta]);prob=cp.Problem(cp.Minimize(eta),cons)
   prob.solve(solver='HIGHS',highs_options={'threads':1,'log_to_console':False,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
   assert prob.status=='optimal';values[s]=max(values[s],float(prob.value));weights[s]=[np.asarray(z.value).copy() for z in zvars];visited.append(s);prob._solver_cache.clear();self.lps+=1
  self.seed_weights=weights
  self.initial_scenario=int(values.argmax());self.relaxation_info=dict(values=values,cheap_metric_bounds=cheap,lp_scenarios=visited,selected=self.initial_scenario,elapsed_s=time.perf_counter()-start)
  print('LP_SEED',self.relaxation_info,flush=True)

 def nccg(self):
  start=time.perf_counter();incumbent,upper=self.initial();lower=float(max(self.relaxation_info['values']));active=[self.initial_scenario]
  self.latest_lower=lower
  if upper-lower<=2e-8*max(1.,abs(upper)):
   self.trace.append(dict(iteration=0,lower=lower,upper=upper,active=[],initial_primal_dual_certificate=True,elapsed_s=time.perf_counter()-start))
   return dict(value=upper,bound=lower,u=incumbent,status='optimal')
  for it in range(1,self.S+3):
   master=self.solve(active);lower=max(lower,master['bound']);self.current_lower=lower;self.latest_lower=lower
   # A previously verified all-scenario incumbent and a new valid master bound
   # already certify optimality. Do not repeat redundant recourse optimizations.
   if upper-lower<=2e-8*max(1.,abs(upper)):
    self.trace.append(dict(iteration=it,lower=lower,upper=upper,active=list(active),existing_incumbent_certificate=True,elapsed_s=time.perf_counter()-start))
    return dict(value=upper,bound=lower,u=incumbent,status='optimal')
   if len(active)==self.S:
    if master['value']<upper:
     upper=master['value'];incumbent=master['u'].copy()
     self.pool=list(dict.fromkeys(self.pool+list(master['key'].values())))
    self.latest_upper=upper;self.latest_u=incumbent.copy()
    return dict(value=upper,bound=lower,u=incumbent,status='optimal' if upper-lower<=2e-8*max(1.,abs(upper)) else 'time_limited')
   missing={s:{(g,self.combinations[s][g]) for g,ids in enumerate(self.eligible(s,master['u'])) if len(ids)==0} for s in range(self.S)}
   missing={s:p for s,p in missing.items() if p}
   if missing:
    assert not set(missing).intersection(active);uncovered=set.union(*missing.values());added=[]
    while uncovered:
     s=max(sorted(missing),key=lambda j:len(missing[j]&uncovered));covered=missing[s]&uncovered;assert covered;added.append(s);uncovered-=covered
    active.extend(added);self.trace.append(dict(iteration=it,lower=lower,upper=upper,active=list(active),feasibility_scenarios=added));continue
   self.master_keys=list(master['key'].values());inner=self.inner(master['u'],it);candidate=max(master['value'],inner['upper'])
   if candidate<upper:upper=candidate;incumbent=master['u'].copy()
   self.latest_upper=upper;self.latest_u=incumbent.copy();self.trace.append(dict(iteration=it,lower=lower,upper=upper,active=list(active),elapsed_s=time.perf_counter()-start))
   if upper-lower<=2e-8*max(1.,abs(upper)):return dict(value=upper,bound=lower,u=incumbent,status='optimal')
   s=inner['worst'];assert s not in active;active.append(s)
  raise RuntimeError('Outer cap')
