"""Exact LP reduced-cost cell screening before the unchanged NCCG solve.

Every rejected binary has LP reduced cost strictly larger than the certified
incumbent-minus-LP-bound gap. Candidate solves provide feasible upper bounds;
they never remove a cell without a lower-bound certificate.
"""
from benchmark_final_v38 import *

def replace_cells(m,cells):
 m.cells=cells;m.maps=[[{signature(x):i for i,x in enumerate(c)} for c in block] for block in cells];m.arr=[]
 for s in range(m.S):
  m.arr.append([(np.array([c['metrics_left'] for c in block])/m.refs,
    (np.array([c['metrics_right'] for c in block])-np.array([c['metrics_left'] for c in block]))/m.refs,
    np.array([c['allocation_upper'] for c in block])/m.B[6*g:6*(g+1)]) for g,block in enumerate(cells[s])])

def relaxation(m):
 start=time.perf_counter();K=len(m.B);u=cp.Variable(K);tr=cp.Variable(K);eta=cp.Variable(nonneg=True)
 cons=[u>=-1,u<=m.qbar,tr>=cp.multiply(25*m.B/m.refs[0],u),tr>=cp.multiply(20*m.B/m.refs[0],u)]
 if m.quota_cuts is not None:
  A,b=m.quota_cuts;cons.append(A@u>=b)
 variables=[]
 for s in range(m.S):
  metric=cp.Constant(np.zeros(3));block=[]
  for g in range(m.groups):
   lo,d,up=m.arr[s][g];J=len(lo);z=cp.Variable(J);a=cp.Variable(J,nonneg=True);nonneg=z>=0
   cons.extend([nonneg,z<=1,cp.sum(z)==1,a<=z,up.T@z<=1+u[6*g:6*(g+1)]])
   metric=metric+lo.T@z+d.T@a;block.append((z,nonneg))
  cons.extend([metric[0]+cp.sum(tr)<=eta,metric[1]<=eta,metric[2]<=eta]);variables.append(block)
 prob=cp.Problem(cp.Minimize(eta),cons)
 prob.solve(solver='HIGHS',highs_options={'threads':1,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'log_to_console':False})
 assert prob.status=='optimal';violation=max(float(np.max(c.violation())) for c in cons);assert violation<1e-6
 row=dict(bound=float(prob.value),elapsed_s=time.perf_counter()-start,max_violation=violation)
 values=[[np.array(v.value).ravel() for v,c in block] for block in variables];rc=[[np.array(c.dual_value).ravel() for v,c in block] for block in variables]
 prob._solver_cache.clear();m.lps+=1
 return row,values,rc

def screen(m):
 began=time.perf_counter();original=m.cells;lp,z,rc=relaxation(m)
 # Deterministically keep the eight largest LP weights in every group/scenario.
 candidate=[[[original[s][g][int(i)] for i in np.argsort(-z[s][g],kind='stable')[:8]] for g in range(m.groups)] for s in range(m.S)]
 replace_cells(m,candidate);inc=m.solve(range(m.S));upper=inc['value'];replace_cells(m,original)
 margin=2e-7*max(1.,abs(upper),abs(lp['bound']));gap=max(0.,upper-lp['bound'])+margin
 kept=[];cert=[]
 for s in range(m.S):
  block=[];cr=[]
  for g in range(m.groups):
   mask=(rc[s][g]<=gap)|(z[s][g]>1e-8);ids=np.flatnonzero(mask)
   assert len(ids);block.append([original[s][g][int(i)] for i in ids])
   cr.append(dict(retained=ids,removed_reduced_cost_min=float(np.min(rc[s][g][~mask])) if (~mask).any() else None,
    max_lp_weight_removed=float(np.max(z[s][g][~mask])) if (~mask).any() else None))
  kept.append(block);cert.append(cr)
 replace_cells(m,kept)
 row=dict(lp=lp,incumbent=upper,screen_threshold=gap,original_cells=sum(map(len,[c for b in original for c in b])),
  retained_cells=sum(map(len,[c for b in kept for c in b])),certificate=cert,elapsed_s=time.perf_counter()-began)
 print('SCREEN',{k:v for k,v in row.items() if k!='certificate'},flush=True)
 return row

def run(date,groups,method):
 target=ROOT/f'revision_v39/experiments/screened/{date}_{groups}_{method}.json'
 if target.exists():print('EXISTS',target);return
 start=time.perf_counter();m=FinalClusterModel(date,groups,reuse=method!='no_reuse');pre=None
 try:
  if method!='full':pre=screen(m)
  r=m.solve(range(m.S)) if method in ['full','screened_full'] else m.nccg()
  row=dict(status=r['status'],objective=r['value'],bound=r['bound'],u=r['u'])
 except Exception as e:row=dict(status='unresolved',exception=type(e).__name__+': '+str(e))
 row.update(date=date,regions=groups,method=method,elapsed_s=time.perf_counter()-start,screen=pre,MILP_calls=m.mips,LP_calls=m.lps,
  solver_events=m.solver_events,outer_trace=m.trace,inner_trace=m.inner_trace)
 dump(target,row);print('SCREENED_RESULT',{k:v for k,v in row.items() if k not in ['u','screen','solver_events','outer_trace','inner_trace']},flush=True)

if __name__=='__main__':
 with threadpool_limits(limits=1):run(sys.argv[1],int(sys.argv[2]),sys.argv[3])
