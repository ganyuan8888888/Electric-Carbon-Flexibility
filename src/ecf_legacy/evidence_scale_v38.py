"""Physical continuous-cell stress instances; preserve every calibration result."""
from evidence_core_v37 import *
from electric_carbon_region_v22 import certify
from evidence_scale_v37 import dynamic_classes,model
import gc
OUT=ROOT/'revision_v38';RESULT=OUT/'experiments'

def vdc(i):
 value=0.;factor=.5
 while i:
  value+=(i%2)*factor;i//=2;factor*=.5
 return value

def protocol38():
 OUT.mkdir(exist_ok=True);RESULT.mkdir(exist_ok=True)
 protocol()
 path=RESULT/'protocol.json'
 obj={'version':38,'baseline':'Unchanged v28 physical network, commitment, process trajectories, carbon budgets and objective; v37 results retained.',
 'scenarios':'Original five plus deterministic complete-day mixtures between the nominal and compatible error scenarios. Weights follow a van der Corput sequence; no scenario selected by algorithm ranking.',
 'cells':'Recompute interval carbon certificates on every compatible complete-process chart; retain continuous intervals, not only exact endpoints. All preset source trajectories remain in the instance.',
 'calibration':'Measure nested 10/20/40/80/160/320-scenario instances, subject to memory and 900 s solver limits. Increase genuine problem size based on runtime and memory only; retain all attempted sizes and statuses. Never pad timings or duplicate identical scenarios.',
 'comparison':'All solvers share identical finite cells, objective, tolerance, one thread and common implementation of scalar LP evaluation. No-reuse clears only the cross-outer mode/value pools.',
 'reporting':'Preserve original negative and positive results. Improvements require original-constraint witnesses; no imposed ranking.'}
 if path.exists():assert json.loads(path.read_text(encoding='utf8'))==obj
 else:dump(path,obj)

def metrics(net,p,pg,pr):
 return np.array([.5*sum(net.generation_cost(pg[:,t]).sum() for t in range(48))+75*(p['available_renewable_mw']-pr).sum(),
 .5*sum(net.generation_emissions(pg[:,t]).sum() for t in range(48)),.5*(p['available_renewable_mw']-pr).sum()])

def build(date,count):
 protocol38();base=ROOT/'results/robust_v28'/date;common=load(base,'common');net,_=make_case(date)
 dest=RESULT/'continuous'/date;dest.mkdir(parents=True,exist_ok=True)
 ps=[load(ROOT/f'results/uncertainty_v26/{date}',f'scenario_{s:02}') for s in range(5)]
 original=[json.loads((base/f'scenario_{s:02}/cells.json').read_text())['cells'] for s in range(5)]
 points=[];pairs=[];families=[]
 for s,cs in enumerate(original):
  points.append({n:load(base/f'scenario_{s:02}',n) for n in sorted({n for c in cs for n in c['endpoints']})})
  pairs.append(sorted({tuple(c['endpoints']) for c in cs if c['endpoints'][0]!=c['endpoints'][1]}))
 for s in range(1,5):
  names=sorted(set(points[0])&set(points[s]));names=[n for n in names if np.max(abs(points[0][n]['pl']-points[s][n]['pl']))<1e-7]
  charts=[pair for pair in sorted(set(pairs[0]+pairs[s])) if set(pair)<=set(names)]
  if charts:families.append((s,names,charts))
 assert families
 for j in range(count):
  target=dest/f'scenario_{j:03}.json'
  if target.exists():continue
  began=time.perf_counter()
  if j<5:
   dump(target,{'scenario':j,'cells':original[j],'source_scenarios':[j],'weights':[1.],'continuous':sum(c['left']<c['right'] for c in original[j]),'preprocessing_s':0.});continue
  s,names,charts=families[(j-5)%len(families)];weight=vdc((j-5)//len(families)+1)
  p=copy.deepcopy(ps[0])
  for key in ['background_mw','available_renewable_mw','minimum_renewable_mw']:p[key]=weight*ps[0][key]+(1-weight)*ps[s][key]
  pp={};pointcells=[]
  for n in names:
   a,b=points[0][n],points[s][n];pg=weight*a['pg']+(1-weight)*b['pg'];pr=weight*a['pr']+(1-weight)*b['pr'];pl=a['pl']
   pp[n]={'pg':pg,'pr':pr,'pl':pl};D,e,iv,rho,fl=batched_carbon(net,p,pg,pr,pl);allocation=(rho[:,net.industrial_buses].T*pl).sum(axis=1)*.5
   balance=float(abs((net.genmap@pg+net.renmap@pr-net.loadmap@pl-p['background_mw']).sum(axis=0)).max())
   assert balance<1e-5 and np.max(abs(fl)-net.capacity[:,None])<1e-5
   if np.all(allocation<=common['budget']*1.2+1e-7):
    m=metrics(net,p,pg,pr);pointcells.append(dict(pattern=n,endpoints=[n,n],left=0.,right=0.,allocation_upper=allocation,allocation_lower=allocation,metrics_left=m,metrics_right=m,kind='exact_point',certificate_error_t=np.zeros(6),neumann_norm=0.))
  cells=[];audits=[]
  for pair in charts:
   a,b=[pp[n] for n in pair];assert np.max(abs(a['pl']-b['pl']))<1e-7
   certs,unresolved=certify(net,p,a,b,common['budget'],.2,max_depth=10)
   excess=-np.inf
   for c in certs:
    mm=[]
    for x in [c['left'],c['right']]:
     pg=a['pg']+x*(b['pg']-a['pg']);pr=a['pr']+x*(b['pr']-a['pr']);mm.append(metrics(net,p,pg,pr))
    cells.append(dict(pattern=pair[0],endpoints=list(pair),left=c['left'],right=c['right'],allocation_upper=c['upper'],allocation_lower=c['lower'],metrics_left=mm[0],metrics_right=mm[1],kind='continuous_interval',certificate_error_t=c['error'],neumann_norm=c['q']))
   # Independently verify a deterministic spread of certified intervals.
   for k in np.unique(np.linspace(0,max(0,len(certs)-1),min(8,len(certs)),dtype=int)):
    c=certs[k]
    for f in [.2113248654,.5,.7886751346]:
     alloc,res=carbon_eval(net,p,a,b,c['left']+f*(c['right']-c['left']));excess=max(excess,float((alloc-c['upper']).max()));assert res<1e-5
   assert not certs or excess<1e-5
   audits.append({'endpoints':pair,'certified_intervals':len(certs),'unresolved_intervals':len(unresolved),'maximum_sample_excess_t':None if not certs else excess})
  cells+=pointcells
  for k,c in enumerate(cells):c['cell_id']=k
  row={'scenario':j,'cells':cells,'source_scenarios':[0,s],'weights':[weight,1-weight],'continuous':sum(c['left']<c['right'] for c in cells),
       'physical_profile_sha256':hashlib.sha256(np.r_[p['background_mw'].ravel(),p['available_renewable_mw'].ravel()].tobytes()).hexdigest(),
       'charts':audits,'preprocessing_s':time.perf_counter()-began,'status':'ready' if cells else 'empty_candidate_set'}
  dump(target,row);print('CONTINUOUS',date,j,len(cells),'intervals',row['continuous'],'seconds',round(row['preprocessing_s'],2),flush=True)
 return [json.loads((dest/f'scenario_{j:03}.json').read_text()) for j in range(count)],common

if __name__=='__main__':
 with threadpool_limits(limits=1):build(sys.argv[1],int(sys.argv[2]))
