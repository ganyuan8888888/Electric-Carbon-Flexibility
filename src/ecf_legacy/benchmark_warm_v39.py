"""Proposed NCCG with a verified primal MIP start and exact incumbent cutoff."""
from benchmark_final_v39 import *
from highs_start_v39 import solve_with_start

class WarmModel(OptimizedModel):
 def set_start(self,S,variables,u,tr,eta):
  q=self.latest_u.copy();u.value=q;tr.value=np.maximum(25*self.B*q,20*self.B*q)/self.refs[0];eta.value=self.latest_upper+1e-10
  for s in S:
   values=[self.fixed_lp(s,q,key) for key in self.pool];key=self.pool[int(np.argmin(values))]
   assert min(values)<=self.latest_upper+2e-8
   constant=np.zeros(3);diff=[]
   for g,k in enumerate(key):
    i=self.maps[s][g][k];lo,d,up=self.arr[s][g];constant+=lo[i];diff.append(d[i])
   constant[0]+=self.trade(q)
   lp=linprog([0.]*self.groups+[1.],A_ub=np.column_stack([np.array(diff).T,-np.ones(3)]),b_ub=-constant,
     bounds=[(0.,1.)]*self.groups+[(0.,None)],method='highs',options={'threads':1});self.lps+=1
   assert lp.success
   for g,(z,a,ix) in enumerate(variables[s]):
    j=int(np.flatnonzero(ix==self.maps[s][g][key[g]])[0]);zz=np.zeros(len(ix));aa=zz.copy();zz[j]=1.;aa[j]=np.clip(lp.x[g],0,1);z.value=zz;a.value=aa

 def solve(self,scenarios,u_fixed=None):
  if time.perf_counter()>self.deadline:raise TimeoutError('Common 600 s per-method budget reached.')
  start=time.perf_counter();S=list(scenarios);K=len(self.B)
  u=cp.Variable(K);tr=cp.Variable(K);eta=cp.Variable(nonneg=True)
  cons=[u>=-1,u<=self.qbar,tr>=cp.multiply(25*self.B/self.refs[0],u),tr>=cp.multiply(20*self.B/self.refs[0],u)]
  if self.quota_cuts is not None:
   A,b=self.quota_cuts;cons.append(A@u>=b)
  if u_fixed is not None:cons.append(u==u_fixed)
  warm = u_fixed is None and np.isfinite(self.latest_upper) and self.latest_u is not None
  if warm:cons.append(eta<=self.latest_upper+1e-9)
  variables={};retained=0
  for s in S:
   metric=cp.Constant(np.zeros(3));variables[s]=[]
   ids=self.eligible(s,u_fixed) if u_fixed is not None else [np.arange(len(self.cells[s][g])) for g in range(self.groups)]
   if any(len(ix)==0 for ix in ids):return dict(status='screening_infeasible',value=np.inf,bound=np.inf)
   for g,ix in enumerate(ids):
    lo,d,up=self.arr[s][g];J=len(ix);z=cp.Variable(J,boolean=True);a=cp.Variable(J,nonneg=True)
    cons += [cp.sum(z)==1,a<=z,up[ix].T@z<=1+u[6*g:6*(g+1)]]
    metric=metric+lo[ix].T@z+d[ix].T@a;variables[s].append((z,a,ix));retained+=J
   cons += [metric[0]+cp.sum(tr)<=eta,metric[1]<=eta,metric[2]<=eta]
  problem=cp.Problem(cp.Minimize(eta),cons);remaining=max(1.,min(self.time_limit,self.deadline-time.perf_counter()))
  if warm:self.set_start(S,variables,u,tr,eta)
  start_audit=solve_with_start(problem,{'threads':1,'time_limit':remaining,'mip_rel_gap':1e-8,'mip_abs_gap':1e-9,
    'mip_feasibility_tolerance':1e-8,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8,'log_to_console':False},warm)
  self.mips+=1;stats=problem.solver_stats.extra_stats
  if eta.value is None:raise RuntimeError('No incumbent: '+str(problem.status))
  bound=float(getattr(stats,'mip_dual_bound',problem.value))
  if self.reserve_tiebreak and u_fixed is None and len(S)<self.S and problem.status=='optimal':
   # A different optimum of the SAME restricted master can provide larger
   # allowance margins to unseen scenarios. The primary objective is fixed.
   cap=float(problem.value)+1e-10*max(1.,abs(float(problem.value)));saved=[(v,np.asarray(v.value).copy()) for v in problem.variables()]
   floor=np.concatenate([np.max([np.min(self.arr[s][g][2],axis=0)-1 for s in range(self.S)],axis=0) for g in range(self.groups)])
   floor=np.maximum(floor,-1);reserve=cp.Variable();extra=[eta<=cap,u>=floor+cp.multiply(self.qbar-floor,reserve),reserve>=0,reserve<=1]
   second=cp.Problem(cp.Maximize(reserve+1e-6*cp.sum(u)/K),cons+extra);began=time.perf_counter()
   try:
    second.solve(solver='HIGHS',highs_options={'threads':1,'time_limit':max(1.,min(120.,self.deadline-time.perf_counter())),
      'mip_rel_gap':1e-8,'mip_abs_gap':1e-9,'mip_feasibility_tolerance':1e-8,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8,'log_to_console':False})
    self.mips+=1
    if eta.value is None or max(float(np.max(c.violation())) for c in cons+extra)>1e-5:
     for variable,value in saved:variable.value=value
    else:self.solver_events.append({'phase':'primary_optimum_quota_margin_tiebreak','scenarios':S,'seconds':time.perf_counter()-began,'primary_cap':cap,'retained_value':float(eta.value),'reserve_fraction':float(reserve.value),'status':second.status})
   except Exception:
    for variable,value in saved:variable.value=value
   finally:second._solver_cache.clear()
  violation=max(float(np.max(c.violation())) for c in cons)
  if not np.isfinite(bound):assert problem.status=='optimal';bound=float(problem.value)
  selected={};key={}
  for s,rows in variables.items():
   selected[s]=[];keys=[]
   for g,(z,a,ix) in enumerate(rows):
    j=int(np.argmax(z.value));assert abs(z.value[j]-1)<1e-5
    keys.append(signature(self.cells[s][g][int(ix[j])]))
    selected[s].append({'cell_index':int(ix[j]),'alpha':float(a.value[j])})
   key[s]=tuple(keys)
  assert violation<1e-5,violation
  event=dict(scenarios=S,fixed_quota=u_fixed is not None,status=problem.status,value=float(problem.value),bound=bound,seconds=time.perf_counter()-start,binaries=retained,
    max_constraint_violation=violation,mip_nodes=int(getattr(stats,'mip_node_count',0)))
  event['primal_start']=start_audit
  self.solver_events.append(event);print('SOLVE',self.date,self.groups,'reuse',self.reuse,event,flush=True)
  problem._solver_cache.clear()
  return dict(**event,u=np.asarray(u.value),selected=selected,key=key)

 def nccg(self):
  start=time.perf_counter();incumbent,upper=self.initial();lower=-np.inf;active=[self.initial_scenario]
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
