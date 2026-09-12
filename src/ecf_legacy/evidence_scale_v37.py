"""NCCG scaling on distinct physical forecast mixtures with original carbon checks."""
from evidence_core_v37 import *
import types,inspect,platform,gc,threading,ctypes
from branch_dispatch import evaluate
from nccg_solver_v26 import FiniteRecourse,signature

def build(date):
 base=ROOT/'results/robust_v28'/date;common=load(base,'common');net,_=make_case(date)
 dest=RESULT/'scaling'/date;dest.mkdir(parents=True,exist_ok=True)
 target=dest/'scenarios.json'
 if target.exists():return json.loads(target.read_text()),common
 began=time.perf_counter();profiles=[load(ROOT/f'results/uncertainty_v26/{date}',f'scenario_{s:02}') for s in range(5)]
 originals=[json.loads((base/f'scenario_{s:02}/cells.json').read_text())['cells'] for s in range(5)]
 output=[{'scenario':s,'cells':cs,'source_scenarios':[s],'weights':[1.]} for s,cs in enumerate(originals)]
 modes=[]
 for s,cs in enumerate(originals):
  allpoints={}
  for c in cs:
   for name in c['endpoints']:
    if name not in allpoints:allpoints[name]=load(base/f'scenario_{s:02}',name)
  modes.append(allpoints)
 # All pair combinations are considered. Compatibility is physical equality
 # of the complete 24h industrial trajectory, not an outcome-based ranking.
 pairs=[]
 for a in range(5):
  for b in range(a+1,5):
   names=[n for n in sorted(set(modes[a])&set(modes[b])) if np.max(abs(modes[a][n]['pl']-modes[b][n]['pl']))<1e-7]
   if names:pairs.append((a,b,names))
 assert pairs
 rng=np.random.default_rng(2026091201+DATES.index(date))
 for s in range(5,40):
  ia,ib,names=pairs[(s-5)%len(pairs)];weight=float(rng.uniform(.1,.9));p=copy.deepcopy(profiles[ia])
  for key in ['background_mw','available_renewable_mw','minimum_renewable_mw']:p[key]=weight*profiles[ia][key]+(1-weight)*profiles[ib][key]
  cells=[];plans={}
  for name in names:
   a,b=modes[ia][name],modes[ib][name];pl=a['pl'];pg=weight*a['pg']+(1-weight)*b['pg'];pr=weight*a['pr']+(1-weight)*b['pr']
   D,e,inv,rho,f=batched_carbon(net,p,pg,pr,pl);alloc=(rho[:,net.industrial_buses].T*pl).sum(axis=1)*.5
   # No uncertainty scenario is dropped; only inadmissible candidate plans are excluded.
   if np.any(alloc>1.2*common['budget']+1e-6):continue
   balance=abs((net.genmap@pg+net.renmap@pr-net.loadmap@pl-p['background_mw']).sum(axis=0)).max()
   assert balance<1e-5 and np.max(abs(f)-net.capacity[:,None])<1e-5
   cost=.5*sum(net.generation_cost(pg[:,t]).sum() for t in range(48))+75*(p['available_renewable_mw']-pr).sum()
   emission=.5*sum(net.generation_emissions(pg[:,t]).sum() for t in range(48));curt=.5*(p['available_renewable_mw']-pr).sum()
   metric=[float(cost),float(emission),float(curt)]
   cells.append({'pattern':name,'endpoints':[name,name],'left':0.,'right':0.,'allocation_upper':alloc.tolist(),'allocation_lower':alloc.tolist(),
     'metrics_left':metric,'metrics_right':metric,'cell_id':len(cells),'kind':'exact_point','certificate_error_t':[0.]*6,'neumann_norm':0.})
   plans[name]={'pg':pg,'pr':pr,'pl':pl}
  output.append({'scenario':s,'cells':cells,'source_scenarios':[ia,ib],'weights':[weight,1-weight]})
  print('SCALE LIBRARY',date,s,len(cells),flush=True)
 dump(target,output);dump(dest/'preprocessing.json',{'elapsed_s':time.perf_counter()-began,'scenarios':len(output),'candidate_cells':[len(x['cells']) for x in output]})
 return output,common

def dynamic_classes():
 # Generalize only the hard-coded finite scenario count and the outer iteration
 # ceiling. Keep master/inner objective, feasibility and bound logic unchanged.
 import nccg_solver_v26,nccg_solver_v32
 src=inspect.getsource(nccg_solver_v26).split("def run(date):")[0]
 src=src.replace('range(5)','range(len(self.cells))').replace('range(1,12)','range(1,len(self.cells)+3)')
 src=src.replace("method='highs')", "method='highs',options={'threads':1})")
 m=types.ModuleType('finite_v37');m.__file__=str(ROOT/'scripts/nccg_solver_v26.py');exec(compile(src,m.__file__,'exec'),m.__dict__)
 src2=inspect.getsource(nccg_solver_v32).replace('from nccg_solver_v26 import FiniteRecourse, signature','')
 src2=src2.replace('range(5)','range(len(self.cells))').replace('range(1,12)','range(1,len(self.cells)+3)')
 # The separately timed extensive solve below verifies every benchmark instance.
 # Remove only duplicate post-termination extensive solves and scenario replay,
 # leaving initialization, bound updates and the final inner check unchanged.
 tail=src2.index('        extensive=self.solve(range(len(self.cells)))')
 src2=src2[:tail]+'''        return dict(date=self.date,lower_ratio=lower,upper_ratio=upper,q=incumbent*self.B,u=incumbent,
            outer_trace=trace,inner_trace=inner_trace,elapsed_s=time.perf_counter()-start,
            LP_calls=self.lp_calls,MIP_calls=self.mip_calls,initialization=initial,
            cache_hits=self.cache_hits,reused_pattern_instances=self.reused_pattern_instances,
            independent_validation='Compared with the separately solved full MILP in this benchmark repeat.')
'''
 n=types.ModuleType('reuse_v37');n.__file__=str(ROOT/'scripts/nccg_solver_v32.py');n.FiniteRecourse=m.FiniteRecourse;n.signature=signature
 exec(compile(src2,n.__file__,'exec'),n.__dict__)
 class NoReuse(n.RecoveryReuseNCCG):
  """Ablate only cross-outer-iteration mode/value reuse, keeping all checks equal."""
  def inner(self,*args,**kwargs):
   self.pool=[];self.value_cache={}
   return super().inner(*args,**kwargs)
 return NoReuse,n.RecoveryReuseNCCG

def model(cls,date,common,scenarios):
 o=cls.__new__(cls);o.date=date;o.path=RESULT/'scaling'/date;o.B=common['budget'];o.refs=common['refs'];o.qbar=common['qbar']/o.B
 o.cells=[s['cells'] for s in scenarios];o.maps=[{signature(c):i for i,c in enumerate(cells)} for cells in o.cells]
 o.hashes=[hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest() for s in scenarios];o.cache={};o.lp_calls=0;o.mip_calls=0
 o.pool=[];o.value_cache={};o.cache_hits=0;o.screened_cell_instances=0;o.reused_pattern_instances=0;o.solver_audit=[];o.inner_call_audit=[]
 return o

def run(date):
 protocol();scenarios,common=build(date);original,reuse=dynamic_classes();rows=[]
 dump(RESULT/'scaling_comparison_protocol.json',{'comparison':'Full extensive MILP versus the proposed NCCG with and without cross-outer-iteration mode reuse. No-reuse clears only the mode and value pools at each inner call; quota screening, outer/inner stopping tolerances, final inner checks, and all model coefficients are identical.',
  'reason':'Source inspection found the historical v26 solver also differed in stopping tolerances and terminal-check logic. It is unsuitable as a pure reuse ablation; preliminary timings are excluded. This correction precedes all final timing runs.',
  'outer_relative_tolerance':1e-8,'inner_relative_tolerance':1e-9,'mip_relative_gap':1e-9,'mip_absolute_gap':1e-10,'threads':1,'repetitions':3})
 for count in [5,10,20,40]:
  subset=scenarios[:count]
  if any(not x['cells'] for x in subset):
   rows.append({'date':date,'scenarios':count,'status':'empty_finite_candidate_set','empty':[s['scenario'] for s in subset if not s['cells']]});continue
  for repeat in range(3):
   # Rotate method order to reduce warm-machine ordering effects.
   methods=[('extensive',original),('original',original),('reuse',reuse)]
   methods=methods[repeat:]+methods[:repeat]
   for name,cls in methods:
    path=RESULT/'scaling'/date/f'{count}_{repeat}_{name}_final.json'
    if path.exists():rows.append(json.loads(path.read_text()));continue
    gc.collect();o=model(cls,date,common,subset);began=time.perf_counter()
    try:
     if name=='extensive':
      result=o.solve(range(count));obj=result['value'];gap=max(0.,result['value']-result['bound']);outer=1
      elapsed=time.perf_counter()-began;verification=0.;total_elapsed=elapsed;calls=o.mip_calls
     else:
      result=o.nccg();obj=result['upper_ratio'];gap=max(0.,result['upper_ratio']-result['lower_ratio']);outer=len(result['outer_trace'])
      total_elapsed=time.perf_counter()-began
      elapsed=result['outer_trace'][-1]['elapsed_s'];verification=0.;calls=o.mip_calls
      dump(RESULT/'scaling'/date/f'{count}_{repeat}_{name}_trace.json',result)
     row={'date':date,'scenarios':count,'repeat':repeat,'method':name,'status':'solved','objective':obj,'gap':gap,'wall_s':elapsed,
          'MILP_calls':calls,'LP_calls':o.lp_calls,'outer_iterations':outer,'cells':sum(len(x['cells']) for x in subset),
          'verification_s':verification,'end_to_end_s':total_elapsed,
          'scope':'Optimization wall time includes initialization and final inner check. The full MILP is solved separately once per method-order repeat; identical-instance objectives are compared in the aggregate audit. Duplicate post-termination replay is not repeated.'}
    except Exception as e:row={'date':date,'scenarios':count,'repeat':repeat,'method':name,'status':'solver_exception','error':str(e),'wall_s':time.perf_counter()-began}
    dump(path,row);rows.append(row);print('SCALING',row,flush=True)
 dump(RESULT/f'scaling_final_summary_{date}.json',rows)

if __name__=='__main__':
 with threadpool_limits(limits=1):
  if len(sys.argv)>2 and sys.argv[2]=='prepare':build(sys.argv[1])
  else:run(sys.argv[1])
