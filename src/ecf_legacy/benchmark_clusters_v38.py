"""Larger coupled-objective regional systems on unchanged certified process charts.

Each region retains the original 79-bus/126-branch physics and six parks.
Regions have distinct joint-error permutations and predetermined allowance
endowments. One common worst-case objective couples their operating choices.
No constraints are duplicated to inflate solve times; each region has its
own first-stage allowances and physical dispatch selection.
"""
from evidence_core_v37 import *
from nccg_solver_v26 import signature
from scipy.optimize import linprog
import cvxpy as cp
import gc
OUT=ROOT/'revision_v38';RESULT=OUT/'experiments/clusters'

class ClusterModel:
 def __init__(self,date,groups,reuse=True):
  self.date,self.groups,self.reuse=date,groups,reuse;self.S=5;self.start=time.perf_counter()
  base=ROOT/f'results/robust_v28/{date}';c=load(base,'common')
  original=[json.loads((base/f'scenario_{s:02}/cells.json').read_text())['cells'] for s in range(5)]
  # Distinct regional endowments are fixed before timing and method comparison.
  factors=1+np.arange(groups)*.005;self.B=np.concatenate([c['budget']*f for f in factors]);self.refs=c['refs']*groups;self.qbar=np.tile(c['qbar']/c['budget'],groups)
  self.cells=[[original[(s+g)%5] for g in range(groups)] for s in range(5)]
  self.maps=[[{signature(x):i for i,x in enumerate(cells)} for cells in block] for block in self.cells]
  self.arr=[]
  for s in range(self.S):
   block=[]
   for g,cells in enumerate(self.cells[s]):
    lo=np.array([x['metrics_left'] for x in cells])/self.refs
    d=(np.array([x['metrics_right'] for x in cells])-np.array([x['metrics_left'] for x in cells]))/self.refs
    upper=np.array([x['allocation_upper'] for x in cells])/self.B[g*6:(g+1)*6]
    block.append((lo,d,upper))
   self.arr.append(block)
  self.pool=[];self.cache={};self.mips=0;self.lps=0;self.cache_hits=0;self.trace=[];self.inner_trace=[];self.solver_events=[]
  self.quota_cuts=None
  self.reserve_tiebreak=False
  self.scenario_initialization=False;self.initial_scenario=0;self.initial_master=None
  self.time_limit=600.;self.deadline=self.start+600.

 def trade(self,u):return float(np.maximum(25*self.B*u,20*self.B*u).sum()/self.refs[0])

 def eligible(self,s,u):
  return [np.flatnonzero(np.max(self.arr[s][g][2]-1-u[g*6:(g+1)*6],axis=1)<=1e-7) for g in range(self.groups)]

 def fixed_lp(self,s,u,key):
  token=(s,np.asarray(u,dtype=np.float64).tobytes(),key)
  if token in self.cache:self.cache_hits+=1;return self.cache[token]
  a=np.zeros(3);diff=[]
  for g,k in enumerate(key):
   if k not in self.maps[s][g]:return np.inf
   i=self.maps[s][g][k];lo,d,up=self.arr[s][g]
   if np.max(up[i]-1-u[6*g:6*(g+1)])>1e-7:return np.inf
   a+=lo[i];diff.append(d[i])
  a[0]+=self.trade(u);matrix=np.column_stack([np.array(diff).T,-np.ones(3)])
  lp=linprog([0.]*self.groups+[1.],A_ub=matrix,b_ub=-a,bounds=[(0.,1.)]*self.groups+[(0.,None)],method='highs',options={'threads':1})
  self.lps+=1
  if not lp.success:raise RuntimeError(lp.message)
  self.cache[token]=float(lp.fun);return float(lp.fun)

 def solve(self,scenarios,u_fixed=None):
  if time.perf_counter()>self.deadline:raise TimeoutError('Common 600 s per-method budget reached.')
  start=time.perf_counter();S=list(scenarios);K=len(self.B)
  u=cp.Variable(K);tr=cp.Variable(K);eta=cp.Variable(nonneg=True)
  cons=[u>=-1,u<=self.qbar,tr>=cp.multiply(25*self.B/self.refs[0],u),tr>=cp.multiply(20*self.B/self.refs[0],u)]
  if self.quota_cuts is not None:
   A,b=self.quota_cuts;cons.append(A@u>=b)
  if u_fixed is not None:cons.append(u==u_fixed)
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
  problem.solve(solver='HIGHS',highs_options={'threads':1,'time_limit':remaining,'mip_rel_gap':1e-8,'mip_abs_gap':1e-9,
    'mip_feasibility_tolerance':1e-8,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8,'log_to_console':False})
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
  self.solver_events.append(event);print('SOLVE',self.date,self.groups,'reuse',self.reuse,event,flush=True)
  problem._solver_cache.clear()
  return dict(**event,u=np.asarray(u.value),selected=selected,key=key)

 def initial(self):
  if self.scenario_initialization:
   seeds=[self.solve([s]) for s in range(self.S)]
   u=np.max([r['u'] for r in seeds],axis=0);values=[self.fixed_lp(s,u,seeds[s]['key'][s]) for s in range(self.S)]
   assert np.isfinite(values).all()
   self.initial_scenario=int(np.argmax([r['bound'] for r in seeds]));self.initial_master=seeds[self.initial_scenario]
   return u,max(values)
  u=self.qbar.copy();vals=[]
  for s in range(self.S):
   key=tuple(signature(self.cells[s][g][0]) for g in range(self.groups));vals.append(self.fixed_lp(s,u,key))
  assert np.isfinite(vals).all();return u,max(vals)

 def inner(self,u,iteration):
  if not self.reuse:self.pool=[];self.cache={}
  ids=[self.eligible(s,u) for s in range(self.S)]
  for s,groups in enumerate(ids):
   if any(not len(x) for x in groups):return {'infeasible':s}
  keys=list(self.pool);lower=-np.inf;worst=None
  def evaluate():return np.array([min((self.fixed_lp(s,u,k) for k in keys),default=np.inf) for s in range(self.S)])
  values=evaluate();upper=float(values.max());reused=len(keys)
  for j in range(1000):
   s=int(values.argmax());r=self.solve([s],u)
   if r['bound']>lower:lower=r['bound'];worst=s
   key=r['key'][s];duplicate=key in keys
   if not duplicate:keys.append(key)
   values=evaluate();upper=min(upper,float(values.max()))
   self.inner_trace.append(dict(outer=iteration,inner=j+1,lower=lower,upper=upper,scenario=s,patterns=len(keys),reused=reused))
   if np.isfinite(upper) and upper-lower<=2e-8*max(1,abs(upper)):
    self.pool=keys;return {'infeasible':None,'lower':lower,'upper':upper,'worst':worst}
   if duplicate and int(values.argmax())==s:raise RuntimeError('Unclosed oracle gap at repeated pattern; retained as unresolved.')
  raise RuntimeError('Inner iteration cap')

 def nccg(self):
  start=time.perf_counter();incumbent,upper=self.initial();lower=-np.inf;active=[self.initial_scenario]
  for it in range(1,self.S+3):
   master=self.initial_master if it==1 and self.initial_master is not None else self.solve(active)
   lower=max(lower,master['bound']);inner=self.inner(master['u'],it)
   if inner['infeasible'] is not None:
    s=inner['infeasible'];self.trace.append(dict(iteration=it,lower=lower,upper=upper,active=list(active),feasibility_scenario=s,elapsed_s=time.perf_counter()-start));assert s not in active;active.append(s);continue
   candidate=max(master['value'],inner['upper'])
   if candidate<upper:upper=candidate;incumbent=master['u'].copy()
   self.trace.append(dict(iteration=it,lower=lower,upper=upper,active=list(active),worst_scenario=inner['worst'],elapsed_s=time.perf_counter()-start))
   if upper-lower<=2e-8*max(1,abs(upper)):return dict(value=upper,bound=lower,u=incumbent,status='optimal')
   s=inner['worst'];assert s not in active;active.append(s)
  raise RuntimeError('Outer iteration cap')

def run(date,groups,method,repeat=0):
 RESULT.mkdir(parents=True,exist_ok=True);target=RESULT/f'{date}_{groups}_{method}_{repeat}.json'
 if target.exists():print('EXISTS',target);return
 gc.collect();start=time.perf_counter();m=ClusterModel(date,groups,reuse=method!='no_reuse')
 try:
  r=m.solve(range(m.S)) if method=='full' else m.nccg()
  row={'status':r['status'],'objective':r['value'],'bound':r['bound'],'gap':max(0.,r['value']-r['bound']),'u':r['u']}
 except Exception as e:row={'status':'unresolved','exception':type(e).__name__+': '+str(e)}
 row.update(date=date,regions=groups,parks=groups*6,buses=groups*79,branches=groups*126,scenarios=m.S,method=method,repeat=repeat,
   elapsed_s=time.perf_counter()-start,cells=sum(len(c) for block in m.cells for c in block),MILP_calls=m.mips,LP_calls=m.lps,
   cache_hits=m.cache_hits,outer_trace=m.trace,inner_trace=m.inner_trace,solver_events=m.solver_events)
 dump(target,row);print('CLUSTER RESULT',{k:v for k,v in row.items() if k not in ['u','outer_trace','inner_trace','solver_events']},flush=True)

if __name__=='__main__':
 with threadpool_limits(limits=1):run(sys.argv[1],int(sys.argv[2]),sys.argv[3],int(sys.argv[4]) if len(sys.argv)>4 else 0)
