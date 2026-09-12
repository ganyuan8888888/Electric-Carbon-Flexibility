"""Primal recovery-mode generation and LP-informed NCCG initialization.

Heuristic mode choices provide upper bounds only. LP/MILP lower bounds and
original constraints certify termination. No oracle optimum is guessed.
"""
from benchmark_joint_v39 import *

class PrimalModel(JointModel):
 def scenario_relaxations(self):
  start=time.perf_counter();K=len(self.B);obj=0;cons=[];etas=[]
  for s in range(self.S):
   u=cp.Variable(K);tr=cp.Variable(K);eta=cp.Variable(nonneg=True);etas.append(eta)
   cons.extend([u>=-1,u<=self.qbar,tr>=cp.multiply(25*self.B/self.refs[0],u),tr>=cp.multiply(20*self.B/self.refs[0],u)])
   A,b=self.quota_cuts;cons.append(A@u>=b);metric=cp.Constant(np.zeros(3))
   for g in range(self.groups):
    lo,d,up=self.arr[s][g];z=cp.Variable(len(lo),nonneg=True);a=cp.Variable(len(lo),nonneg=True)
    cons.extend([cp.sum(z)==1,a<=z,up.T@z<=1+u[6*g:6*(g+1)]])
    metric+=lo.T@z+d.T@a
   cons.extend([metric[0]+cp.sum(tr)<=eta,metric[1]<=eta,metric[2]<=eta]);obj+=eta
  prob=cp.Problem(cp.Minimize(obj),cons);prob.solve(solver='HIGHS',highs_options={'threads':1,'log_to_console':False,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
  assert prob.status=='optimal';values=np.array([e.value for e in etas],float);prob._solver_cache.clear();self.lps+=self.S
  self.initial_scenario=int(values.argmax());self.relaxation_info=dict(values=values,selected=self.initial_scenario,elapsed_s=time.perf_counter()-start)
  print('LP_SEED',self.relaxation_info,flush=True)

 def greedy(self,s,u):
  ids=self.eligible(s,u)
  if any(len(ix)==0 for ix in ids):return None
  chosen=[int(ix[0]) for ix in ids];alpha=np.zeros(self.groups)
  terms=np.array([self.arr[s][g][0][i] for g,i in enumerate(chosen)]);trade=np.array([self.trade(u),0.,0.])
  for iteration in range(5):
   changed=False
   for g,ix in enumerate(ids):
    lo,d,_=self.arr[s][g];offset=terms.sum(axis=0)-terms[g]+trade;a=lo[ix]+offset;dd=d[ix]
    candidates=[np.zeros(len(ix)),np.ones(len(ix))]
    for i,j in [(0,1),(0,2),(1,2)]:
     den=dd[:,i]-dd[:,j];x=np.divide(a[:,j]-a[:,i],den,out=np.zeros(len(ix)),where=abs(den)>1e-14);candidates.append(np.clip(x,0,1))
    x=np.array(candidates).T;values=np.maximum(0.,np.max(a[:,None,:]+dd[:,None,:]*x[:,:,None],axis=2));ii,jj=np.unravel_index(values.argmin(),values.shape)
    index=int(ix[ii]);al=float(x[ii,jj]);changed|=(index!=chosen[g] or abs(al-alpha[g])>1e-10);chosen[g]=index;alpha[g]=al;terms[g]=lo[index]+d[index]*al
   if not changed:break
  return tuple(signature(self.cells[s][g][chosen[g]]) for g in range(self.groups))

 def initial(self):
  self.scenario_relaxations();u=self.qbar.copy();keys=[self.greedy(s,u) for s in range(self.S)]
  assert all(k is not None for k in keys)
  self.pool=list(dict.fromkeys(keys));values=[self.fixed_lp(s,u,keys[s]) for s in range(self.S)];return u,max(values)

 def inner(self,u,iteration):
  if not self.reuse:self.pool=[];self.cache={}
  keys=list(dict.fromkeys(self.pool+self.master_keys))
  def evaluate():return np.array([min((self.fixed_lp(s,u,k) for k in keys),default=np.inf) for s in range(self.S)])
  values=evaluate();lower=self.current_lower;target=lower+2e-8*max(1.,abs(lower))
  for s in range(self.S):
   if values[s]>target:
    k=self.greedy(s,u);assert k is not None
    if k not in keys:keys.append(k)
  values=evaluate();upper=float(values.max());worst=int(values.argmax());reused=len(self.pool)
  if upper<=target:
   self.pool=keys;self.inner_trace.append(dict(outer=iteration,inner=0,lower=lower,upper=upper,patterns=len(keys),reused=reused,primal_certificate=True))
   return dict(infeasible=None,lower=lower,upper=upper,worst=worst)
  for j in range(1000):
   s=int(values.argmax());r=self.solve([s],u)
   if r['bound']>lower:lower=r['bound'];worst=s
   k=r['key'][s];duplicate=k in keys
   if not duplicate:keys.append(k)
   values=evaluate();upper=min(upper,float(values.max()))
   self.inner_trace.append(dict(outer=iteration,inner=j+1,lower=lower,upper=upper,scenario=s,patterns=len(keys),reused=reused))
   if upper-lower<=2e-8*max(1.,abs(upper)):
    self.pool=keys;return dict(infeasible=None,lower=lower,upper=upper,worst=worst)
   if duplicate and int(values.argmax())==s:raise RuntimeError('Unclosed repeated oracle')
  raise RuntimeError('Inner cap')

 def nccg(self):
  start=time.perf_counter();incumbent,upper=self.initial();lower=-np.inf;active=[self.initial_scenario]
  for it in range(1,self.S+3):
   master=self.solve(active);lower=max(lower,master['bound']);self.current_lower=lower;self.latest_lower=lower
   if len(active)==self.S:
    if master['value']<upper:upper=master['value'];incumbent=master['u'].copy()
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
   self.latest_upper=upper;self.latest_u=incumbent.copy()
   self.trace.append(dict(iteration=it,lower=lower,upper=upper,active=list(active),elapsed_s=time.perf_counter()-start))
   if upper-lower<=2e-8*max(1.,abs(upper)):return dict(value=upper,bound=lower,u=incumbent,status='optimal')
   s=inner['worst'];assert s not in active;active.append(s)
  raise RuntimeError('Outer cap')

def run(date,count,method,phase='calibration'):
 target=ROOT/f'revision_v39/experiments/primal_{phase}/{date}_{count}_{method}.json'
 if target.exists():print('EXISTS',target);return
 start=time.perf_counter();m=PrimalModel(date,4,count,reuse=method!='no_reuse')
 try:
  r=m.solve(range(m.S)) if method=='full' else m.nccg();row=dict(status=r['status'],objective=r['value'],bound=r['bound'],u=r['u'])
 except Exception as e:
  row=dict(status='unresolved',exception=type(e).__name__+': '+str(e))
  if np.isfinite(m.latest_upper):row.update(objective=m.latest_upper,bound=m.latest_lower,u=m.latest_u)
 row.update(date=date,regions=4,scenarios=count,method=method,elapsed_s=time.perf_counter()-start,
  MILP_calls=m.mips,LP_calls=m.lps,cache_hits=m.cache_hits,solver_events=m.solver_events,outer_trace=m.trace,inner_trace=m.inner_trace,
  relaxation_info=getattr(m,'relaxation_info',None),combinations=m.combinations)
 dump(target,row);print('PRIMAL_RESULT',{k:v for k,v in row.items() if k not in ['u','combinations','solver_events','outer_trace','inner_trace','relaxation_info']},flush=True)

if __name__=='__main__':
 with threadpool_limits(limits=1):run(sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4] if len(sys.argv)>4 else 'calibration')
