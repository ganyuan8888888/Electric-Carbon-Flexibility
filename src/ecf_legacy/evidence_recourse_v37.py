"""Fixed-allowance out-of-sample recourse; every accepted plan has an original-model witness.

The candidate search is sequential MILP on frozen complete industrial modes.
It does not certify global nonlinear infeasibility when no witness is obtained.
"""
from evidence_core_v37 import *
import datetime,gc
import cvxpy as cp
from scipy import sparse
from uncertainty_scenarios_v26 import annual
from rts_network import make_rts_network

def pools(date):
 base=ROOT/'results/robust_v28'/date;patterns=[];names=[];sources=[]
 for s in range(5):
  data=json.loads((base/f'scenario_{s:02}/cells.json').read_text())
  for c in data['cells']:
   a=load(base/f'scenario_{s:02}',c['endpoints'][0]);v=a['pl']
   if not any(np.max(abs(x-v))<1e-6 for x in patterns):
    patterns.append(v);names.append(c['pattern']);sources.append(str((base/f'scenario_{s:02}'/c['endpoints'][0]).relative_to(ROOT)))
 assert patterns
 return np.array(patterns),names,sources

def scenario(net,nominal,datasets,date,donor,scale):
 p=copy.deepcopy(nominal);day=(datetime.date.fromisoformat(date)-datetime.date(2020,1,1)).days
 raw={k:d['forecast'][day]+scale*d['error'][donor] for k,d in datasets.items()};clip=0.
 vals=np.maximum(raw['Load'],0);clip+=float(abs(vals-raw['Load']).sum())
 for area in [1,2,3]:
  mask=net.busdata['Area'].to_numpy()==area;weights=net.busdata.loc[mask,'MW Load'].to_numpy()
  p['background_mw'][np.flatnonzero(mask)]=weights[:,None]/weights.sum()*vals[:,datasets['Load']['columns'].index(str(area))]
 origin=make_rts_network()
 for j,g in net.renewable.iterrows():
  kind=g['Unit Type']
  if kind not in ['WIND','PV']:continue
  old,share=net.direct_profile_mapping[j];uid=origin.renewable.iloc[old]['GEN UID']
  x=share*raw[kind][:,datasets[kind]['columns'].index(uid)];v=np.clip(x,0,g['PMax MW']);clip+=float(abs(x-v).sum())
  p['available_renewable_mw'][j]=v
 p['minimum_renewable_mw']=p['available_renewable_mw']*net.must_take[:,None]
 p.update(donor=donor,error_scale=scale,physical_clipping_mwh=clip*.5)
 return p

def gradient(net,p,pg,pr,pl):
 # Remove exactly inactive zero-throughput nodes, as prescribed by the model.
 flows=net.ptdf@(net.genmap@pg+net.renmap@pr-net.loadmap@pl-p['background_mw'])+net.flow_offset[:,None]
 T=pl.shape[1];n=len(net.bus);G=len(net.gen);R=len(net.renewable);K=6
 u=net.branch[:,0].astype(int);v=net.branch[:,1].astype(int);inv=np.zeros((T,n,n));rho=np.zeros((T,n))
 for t in range(T):
  sender=np.where(flows[:,t]>=0,u,v);receiver=np.where(flows[:,t]>=0,v,u)
  pin=net.genmap@pg[:,t]+net.renmap@pr[:,t];np.add.at(pin,receiver,abs(flows[:,t]))
  D=np.diag(pin);np.add.at(D,(receiver,sender),-abs(flows[:,t]));e=net.genmap@net.generation_emissions(pg[:,t])
  active=pin>1e-8;idx=np.ix_(active,active);inv[t][idx]=np.linalg.inv(D[idx]);rho[t]=inv[t]@e
  assert np.max(abs(D@rho[t]-e))<1e-5
 maps=np.c_[net.genmap,net.renmap,-net.loadmap];df=net.ptdf@maps
 direct=np.c_[net.genmap,net.renmap,np.zeros_like(net.loadmap)]
 u=net.branch[:,0].astype(int);v=net.branch[:,1].astype(int)
 grads=[]
 for t in range(T):
  rhs=np.c_[net.genmap*net.generation_emission_derivatives(pg[:,t])[None,:],np.zeros((n,R+K))]-direct*rho[t,:,None]
  sender=np.where(flows[:,t]>=0,u,v);receiver=np.where(flows[:,t]>=0,v,u)
  np.add.at(rhs,receiver,-np.sign(flows[:,t])[:,None]*df*(rho[t,receiver]-rho[t,sender])[:,None])
  dr=inv[t]@rhs;g=pl[:,t,None]*dr[net.industrial_buses]
  g[:,-K:]+=np.diag(rho[t,net.industrial_buses]);grads.append(g*.5)
 alloc=(rho[:,net.industrial_buses].T*pl).sum(axis=1)*.5
 return alloc,np.array(grads).transpose(1,2,0),rho.T,flows

class Probe:
 def __init__(self,net,patterns,common):
  self.net,self.patterns,self.common=net,patterns,common;self.initial_seed=None;J,K,T=patterns.shape;G=len(net.gen);R=len(net.renewable);N=len(net.bus)
  self.pg=cp.Variable((G,T));self.pr=cp.Variable((R,T));self.theta=cp.Variable((N,T));self.h=cp.Variable((G,T),nonneg=True)
  self.w=cp.Variable(J,boolean=J>1);self.pl=cp.vstack([self.w@patterns[:,k,:] for k in range(K)])
  self.bg=cp.Parameter((N,T));self.av=cp.Parameter((R,T));self.mn=cp.Parameter((R,T));self.cap=cp.Parameter(K)
  self.coef=cp.Parameter((K,(G+R)*T+J));self.rhs=cp.Parameter(K);self.slack=cp.Variable(nonneg=True)
  self.ganchor=cp.Parameter((G,T));self.ranchor=cp.Parameter((R,T));self.radius=cp.Parameter(nonneg=True)
  inc=sparse.csc_matrix(net.incidence);tap=np.where(net.branch[:,8]==0,1,net.branch[:,8]);sus=100/(net.branch[:,3]*tap)
  flow=sparse.diags(sus)@inc.T@self.theta
  inj=sparse.csc_matrix(net.genmap)@self.pg+sparse.csc_matrix(net.renmap)@self.pr-sparse.csc_matrix(net.loadmap)@self.pl-self.bg
  on=common['commitment'];self.cons=[cp.sum(self.w)==1,self.w>=0,self.pg>=net.pmin[:,None]*on[:,None],self.pg<=net.pmax[:,None]*on[:,None],
    self.pr>=self.mn,self.pr<=self.av,inc@flow==inj,self.theta[0]==0,flow<=net.capacity[:,None],flow>=-net.capacity[:,None],
    cp.abs(self.pg[:,1:]-self.pg[:,:-1])<=net.ramp_mw_per_hour[:,None]*.5,
    cp.abs(self.pg-self.ganchor)<=self.radius,cp.abs(self.pr-self.ranchor)<=self.radius]
  for g,(xp,hp) in enumerate(zip(net.fuel_power_points,net.fuel_heat_points)):
   for j in range(len(xp)-1):
    a=(hp[j+1]-hp[j])/(xp[j+1]-xp[j]);b=hp[j]-a*xp[j];self.cons.append(self.h[g]>=a*self.pg[g]+b*on[g])
  self.cost=.5*(cp.sum(cp.multiply(net.fuel_price[:,None],self.h)+cp.multiply(net.vom[:,None],self.pg))+150*cp.sum(self.av-self.pr))
  self.emission=.5*cp.sum(cp.multiply(net.fuel_carbon_t_mmbtu[:,None],self.h));self.curt=.5*cp.sum(self.av-self.pr)
  self.eta=cp.Variable(nonneg=True);refs=common['refs']
  self.cons +=[self.cost/refs[0]<=self.eta,self.emission/refs[1]<=self.eta,self.curt/refs[2]<=self.eta]
  vec=cp.hstack([cp.vec(self.pg,order='C'),cp.vec(self.pr,order='C'),self.w])
  self.cons.append(self.coef@vec<=self.rhs+self.slack)
  self.problem=cp.Problem(cp.Minimize(self.eta+1e5*self.slack+1e-8*self.cost/1e6),self.cons)

 def solve(self,p,cap,maxiter=5,return_plans=False,trust_radius=None):
  start=time.perf_counter();net=self.net;G=len(net.gen);R=len(net.renewable);J=len(self.patterns);B=self.common['budget']
  self.bg.value=p['background_mw'];self.av.value=p['available_renewable_mw'];self.mn.value=p['minimum_renewable_mw']
  self.coef.value=np.zeros(self.coef.shape);self.rhs.value=np.ones(6)*1e6;self.ganchor.value=np.zeros(self.pg.shape);self.ranchor.value=np.zeros(self.pr.shape);self.radius.value=1e5
  records=[];best=None;bestex=np.inf;plans=[]
  for it in range(-1,maxiter):
   if it==-1:
    if self.initial_seed is None:continue
    seed=self.initial_seed;j=int(np.argmin(np.max(abs(self.patterns-seed['pl']),axis=(1,2))))
    if np.max(abs(self.patterns[j]-seed['pl']))>1e-6:continue
    pg=seed['pg'].copy();pr=seed['pr'].copy();pl=self.patterns[j].copy()
   else:
    try:
     self.problem._solver_cache.clear()
     self.problem.solve(solver='HIGHS',warm_start=False,ignore_dpp=True,highs_options={'threads':1,'time_limit':25,'mip_rel_gap':1e-5,'log_to_console':False})
    except Exception as e:return {'status':'solver_error','error':str(e),'elapsed_s':time.perf_counter()-start}
    if self.pg.value is None:return {'status':self.problem.status,'witness_found':False,'elapsed_s':time.perf_counter()-start,'iterations':records}
    pg=self.pg.value.copy();pr=self.pr.value.copy();j=int(np.argmax(self.w.value));pl=self.patterns[j].copy()
    if np.max(abs(self.pl.value-pl))>1e-5:return {'status':'fractional_unresolved','witness_found':False,'elapsed_s':time.perf_counter()-start}
   try:alloc,grad,rho,flow=gradient(net,p,pg,pr,pl)
   except Exception as e:return {'status':'original_carbon_recheck_unresolved','witness_found':False,'error':str(e),'elapsed_s':time.perf_counter()-start,'iterations':records}
   excess=float(np.max(alloc-cap));exratio=float(np.max((alloc-cap)/B))
   pcost=.5*sum(net.generation_cost(pg[:,t]).sum() for t in range(48))+150*.5*(p['available_renewable_mw']-pr).sum()
   emis=.5*sum(net.generation_emissions(pg[:,t]).sum() for t in range(48));curt=.5*(p['available_renewable_mw']-pr).sum()
   balance=float(abs((net.genmap@pg+net.renmap@pr-net.loadmap@pl-p['background_mw']).sum(axis=0)).max())
   line=max(0.,float((abs(flow)-net.capacity[:,None]).max()));ramp=max(0.,float((abs(np.diff(pg,axis=1))-net.ramp_mw_per_hour[:,None]*.5).max()))
   on=self.common['commitment'];source=max(0.,float((pg-net.pmax[:,None]*on[:,None]).max()),float((net.pmin[:,None]*on[:,None]-pg).max()),float((pr-p['available_renewable_mw']).max()),float((p['minimum_renewable_mw']-pr).max()))
   row={'iteration':it,'mode':j,'allocation_t':alloc,'carbon_excess_t':max(0.,excess),'carbon_excess_ratio':max(0.,exratio),'cost_usd':float(pcost),
     'source_emission_t':float(emis),'curtailment_mwh':float(curt),'balance_residual_mw':balance,'line_excess_mw':line,'ramp_excess_mw':ramp,
     'source_limit_excess_mw':source,
     'objective':float(max(np.array([pcost,emis,curt])/self.common['refs']))}
   records.append(row);plan=dict(pg=pg,pr=pr,pl=pl,allocation_t=alloc,rho=rho,flow=flow,commitment=self.common['commitment'],**{k:row[k] for k in ['mode','cost_usd','source_emission_t','curtailment_mwh','objective']})
   if return_plans:plans.append(plan)
   if exratio<bestex:bestex=exratio;best=row.copy()
   if excess<=1e-4 and balance<1e-5 and line<1e-5 and ramp<1e-5 and source<1e-5:
    return dict(status='original_constraints_verified',witness_found=True,**row,elapsed_s=time.perf_counter()-start,iterations=records,plan=plan if return_plans else None,plans=plans if return_plans else None)
   # First-order cuts are only a candidate-search device. Acceptance always uses the original nonlinear carbon equations.
   gpg=grad[:,:G,:].reshape(6,-1);gpr=grad[:,G:G+R,:].reshape(6,-1);gw=np.einsum('ikt,jkt->ij',grad[:,G+R:,:],self.patterns)
   co=np.c_[gpg,gpr,gw];x=np.r_[pg.ravel(),pr.ravel(),np.eye(J)[j]]
   self.coef.value=co/B[:,None];self.rhs.value=(cap-alloc+co@x)/B
   self.ganchor.value=pg;self.ranchor.value=pr;self.radius.value=1e4 if it==-1 else (100. if it<2 else 40.)
   if trust_radius is not None:self.radius.value=min(self.radius.value,trust_radius)
  return dict(status='no_verified_witness_within_search_budget',witness_found=False,**best,elapsed_s=time.perf_counter()-start,iterations=records,plans=plans if return_plans else None)

def run(date,limit=None,shard=None,shards=1,retry_errors=False):
 protocol();base=ROOT/'results/robust_v28'/date;common=load(base,'common');net,_=make_case(date);net.fixed_commitment=common['commitment']
 nominal=load(ROOT/f'results/uncertainty_v26/{date}','scenario_00');patterns,names,sources=pools(date)
 dst=RESULT/'oos_final'/date;dst.mkdir(parents=True,exist_ok=True)
 dump(dst/'modes.json',{'names':names,'sources':sources,'count':len(patterns),'sha256':hashlib.sha256(patterns.tobytes()).hexdigest()})
 from nccg_solver_v26 import FiniteRecourse
 fm=FiniteRecourse(date,result_root=ROOT/'results/robust_v28');nom=fm.solve([0]);qnom=nom['u']*common['budget']
 qrob=load(base,'NCCG')['q'];constant=load(base,'constant_robust');qconst=constant.get('q',load(base,'constant_nominal')['q'])
 constantpl=load(base/'scenario_00','M1_constant')['pl'][None,:,:]
 probes={'M1':Probe(net,constantpl,common),'M4-N':Probe(net,patterns,common),'M4':Probe(net,patterns,common)}
 quotas={'M1':qconst,'M4-N':qnom,'M4':qrob};dump(dst/'first_stage.json',quotas)
 datasets,hashes=annual();manifest=json.loads((ROOT/f'results/uncertainty_v26/{date}/manifest.json').read_text())
 train=[(datetime.date.fromisoformat(d)-datetime.date(2020,1,1)).days for d in manifest['training_days']]
 rng=np.random.default_rng(2026091201+DATES.index(date));draws=[(int(rng.choice(train)),float(rng.lognormal(-.5*.15**2,.15))) for _ in range(200)]
 cases=[('in_sample',s,load(ROOT/f'results/uncertainty_v26/{date}',f'scenario_{s:02}')) for s in range(5)]
 cases += [('held_out',s-5,load(ROOT/f'results/uncertainty_v26/{date}',f'scenario_{s:02}')) for s in range(5,17)]
 cases += [('pressure',i,scenario(net,nominal,datasets,date,d,scale)) for i,(d,scale) in enumerate(draws)]
 dump(dst/'draws.json',{'pressure_draws':draws,'source_hashes':hashes})
 pressure_cache={}
 representatives=[load(ROOT/f'results/uncertainty_v26/{date}',f'scenario_{s:02}') for s in range(5)]
 representative_modes=[load(base/f'scenario_{s:02}','M4_proposed')['pl'] for s in range(5)]
 def reconstruct(solution,s):
  sel=solution['selected'][s];cs=fm.cells[s];c=next(c for c in cs if c['cell_id']==sel['cell_id'])
  a,b=[load(base/f'scenario_{s:02}',n) for n in c['endpoints']];alpha=c['left']+sel['alpha']*(c['right']-c['left'])
  return {'pl':a['pl'],'pg':a['pg']+alpha*(b['pg']-a['pg']),'pr':a['pr']+alpha*(b['pr']-a['pr'])}
 nominal_seed=reconstruct(nom,0)
 dump(dst/'search_protocol.json',{'historical':'Up to five sequential MILP candidates in the frozen full mode pool.',
   'pressure':'Nearest optimization trajectory in capacity-scaled joint load/renewable distance selects a complete certified recovery mode; five sequential LP candidates for network recourse. M4-N and M4 use the same selected mode and differ only in frozen first-stage quotas.',
   'change_reason':'Separate full-library historical validation and computationally bounded mode-policy stress evaluation; pilot showed repeated full-library MIPs dominate runtime. Applied before any generated pressure case is evaluated.'})
 for group,i,p in cases[:limit]:
  if shard is not None and ((not retry_errors and group!='pressure') or i%shards!=shard):continue
  f=dst/f'{group}_{i:03}.json'
  target=f;saved=None
  if retry_errors:
   if not f.exists():continue
   saved=json.loads(f.read_text())
   if not any(r['status']=='solver_error' for r in saved['results'].values()):continue
   target=RESULT/'oos_numerical_retry'/date/f.name
   if target.exists():continue
  elif f.exists():continue
  results={};current=probes
  distances=[float(np.mean(abs(p['background_mw']-r['background_mw'])/np.maximum(1.,nominal['background_mw']))+np.mean(abs(p['available_renewable_mw']-r['available_renewable_mw'])/np.maximum(1.,net.renewable['PMax MW'].to_numpy()[:,None]))) for r in representatives]
  nearest=int(np.argmin(distances))
  if group=='pressure':
   if nearest not in pressure_cache:pressure_cache[nearest]={'M1':probes['M1'],'M4-N':Probe(net,representative_modes[nearest][None,:,:],common),'M4':Probe(net,representative_modes[nearest][None,:,:],common)}
   current=pressure_cache[nearest]
  for method,probe in current.items():
   if retry_errors and saved['results'][method]['status']!='solver_error':
    results[method]=saved['results'][method];continue
   probe.initial_seed=load(base/f'scenario_{nearest:02}','M1_constant' if method=='M1' else 'M4_proposed')
   if probe.initial_seed is None:probe.initial_seed=load(base/'scenario_00','M1_constant' if method=='M1' else 'M4_proposed')
   if method=='M4-N' and group!='pressure':probe.initial_seed=nominal_seed
   r=probe.solve(p,common['budget']+quotas[method])
   if retry_errors:
    attempts=[r.copy()]
    if r['status']=='solver_error':r=probe.solve(p,common['budget']+quotas[method]);attempts.append(r.copy())
    r['numerical_retry_attempts']=attempts
   r['quota_t']=quotas[method];results[method]=r
  dump(target,{'group':group,'index':i,'date':date,'clipping_mwh':p.get('physical_clipping_mwh',0.),'results':results,'numerical_retry_only':retry_errors})
  print('OOS',date,group,i,{m:(x['status'],round(x['elapsed_s'],2)) for m,x in results.items()},flush=True)
  gc.collect()

if __name__=='__main__':
 with threadpool_limits(limits=1):
  if len(sys.argv)>2 and sys.argv[2]=='retry':run(sys.argv[1],shard=int(sys.argv[3]) if len(sys.argv)>3 else None,shards=int(sys.argv[4]) if len(sys.argv)>4 else 1,retry_errors=True)
  elif len(sys.argv)>2 and sys.argv[2]=='shard':run(sys.argv[1],None,int(sys.argv[3]),int(sys.argv[4]))
  else:run(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else None)
