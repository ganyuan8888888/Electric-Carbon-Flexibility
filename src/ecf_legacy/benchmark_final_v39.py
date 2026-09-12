"""Feasible-mode polishing and immediate global-bound closure for NCCG."""
from benchmark_primal_v39 import *

class OptimizedModel(PrimalModel):
 def initial(self):
  self.scenario_relaxations();q=self.qbar.copy();keys=[self.greedy(s,q) for s in range(self.S)]
  chosen=np.array([[self.maps[s][g][keys[s][g]] for g in range(self.groups)] for s in range(self.S)])
  al=np.zeros((self.S,self.groups));terms=np.array([[self.arr[s][g][0][chosen[s,g]] for g in range(self.groups)] for s in range(self.S)])
  cap=np.array([[self.arr[s][g][2][chosen[s,g]] for g in range(self.groups)] for s in range(self.S)])
  start=time.perf_counter()
  for sweep in range(6):
   changed=False
   for s in range(self.S):
    for g in range(self.groups):
     lo,d,up=self.arr[s][g];J=len(lo);othercap=np.max(np.delete(cap[:,g],s,axis=0),axis=0)
     qg=np.maximum(up,othercap)-1.;qall=cap.max(axis=0).ravel()-1.;qother=qall.copy();qother[6*g:6*(g+1)]=0.
     trade=self.trade(qother)+np.maximum(25*self.B[6*g:6*(g+1)]*qg,20*self.B[6*g:6*(g+1)]*qg).sum(axis=1)/self.refs[0]
     totals=terms.sum(axis=1);othermetrics=np.max(np.delete(totals,s,axis=0),axis=0)
     off=totals[s]-terms[s,g];a=lo+off;a[:,0]+=trade
     constant=np.maximum.reduce([othermetrics[0]+trade,np.full(J,max(0.,othermetrics[1],othermetrics[2]))])
     aa=np.column_stack([a,constant]);dd=np.column_stack([d,np.zeros(J)])
     candidates=[np.zeros(J),np.ones(J)]
     for i,j in itertools.combinations(range(4),2):
      den=dd[:,i]-dd[:,j];x=np.divide(aa[:,j]-aa[:,i],den,out=np.zeros(J),where=abs(den)>1e-14);candidates.append(np.clip(x,0,1))
     x=np.array(candidates).T;val=np.max(aa[:,None,:]+dd[:,None,:]*x[:,:,None],axis=2)
     v=val.min(axis=1);jj=np.argmin(val,axis=1);best=float(v.min());eligible=np.flatnonzero(v<=best+1e-12)
     i=int(eligible[np.argmin(trade[eligible])]);alpha=float(x[i,jj[i]])
     changed|=(i!=chosen[s,g] or abs(alpha-al[s,g])>1e-9);chosen[s,g]=i;al[s,g]=alpha;terms[s,g]=lo[i]+d[i]*alpha;cap[s,g]=up[i]
   if not changed:break
  u=np.maximum(-1.,cap.max(axis=0).ravel()-1.);assert np.max(u-self.qbar)<1e-7
  keys=[tuple(signature(self.cells[s][g][chosen[s,g]]) for g in range(self.groups)) for s in range(self.S)]
  values=[self.fixed_lp(s,u,keys[s]) for s in range(self.S)];self.pool=list(dict.fromkeys(keys));upper=max(values)
  self.initial_polish=dict(sweeps=sweep+1,elapsed_s=time.perf_counter()-start,upper=upper)
  self.latest_upper=upper;self.latest_u=u.copy();print('INITIAL_POLISH',self.initial_polish,flush=True)
  return u,upper

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
   self.latest_upper=upper;self.latest_u=incumbent.copy();self.trace.append(dict(iteration=it,lower=lower,upper=upper,active=list(active),elapsed_s=time.perf_counter()-start))
   if upper-lower<=2e-8*max(1.,abs(upper)):return dict(value=upper,bound=lower,u=incumbent,status='optimal')
   s=inner['worst'];assert s not in active;active.append(s)
  raise RuntimeError('Outer cap')

def run(date,count,method,phase='calibration'):
 target=ROOT/f'revision_v39/experiments/optimized_{phase}/{date}_{count}_{method}.json'
 if target.exists():print('EXISTS',target);return
 start=time.perf_counter();m=OptimizedModel(date,4,count,reuse=method!='no_reuse')
 try:
  r=m.solve(range(m.S)) if method=='full' else m.nccg();row=dict(status=r['status'],objective=r['value'],bound=r['bound'],u=r['u'])
 except Exception as e:
  row=dict(status='unresolved',exception=type(e).__name__+': '+str(e))
  if np.isfinite(m.latest_upper):row.update(objective=m.latest_upper,bound=m.latest_lower,u=m.latest_u)
 row.update(date=date,regions=4,scenarios=count,method=method,elapsed_s=time.perf_counter()-start,
  MILP_calls=m.mips,LP_calls=m.lps,cache_hits=m.cache_hits,solver_events=m.solver_events,outer_trace=m.trace,inner_trace=m.inner_trace,
  relaxation_info=getattr(m,'relaxation_info',None),initial_polish=getattr(m,'initial_polish',None),combinations=m.combinations)
 dump(target,row);print('OPTIMIZED_RESULT',{k:v for k,v in row.items() if k not in ['u','combinations','solver_events','outer_trace','inner_trace','relaxation_info']},flush=True)

if __name__=='__main__':
 with threadpool_limits(limits=1):run(sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4] if len(sys.argv)>4 else 'calibration')
