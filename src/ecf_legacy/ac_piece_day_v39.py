"""Whole-day AC restoration with intertemporal ramps and original daily quotas."""
from ac_existing_units_v39 import *
from ac_piece_v39 import ACPieceModel
OUT=ROOT/'revision_v39/experiments/ac_piece'

def density(net,r):
 g,br=r['gen'],r['branch'];n=len(net.bus);D=np.zeros((n,n));pin=np.zeros(n);on=g[:,7]>0
 np.add.at(pin,g[on,0].astype(int),g[on,1])
 for f,t,pf,pt in br[:,[0,1,13,15]]:
  f,t=int(f),int(t)
  if pf>=0 and pt<0:pin[t]+=-pt;D[t,f]+=pt
  elif pt>=0 and pf<0:pin[f]+=-pf;D[f,t]+=pf
 D[np.arange(n),np.arange(n)]=pin;e=net.genmap@net.generation_emissions(np.maximum(g[:len(net.gen),1],0.))
 if np.min(pin)<=1e-8:
  empty=np.flatnonzero(pin<=1e-8)
  assert np.max(abs(e[empty]))<1e-7 and np.max(abs(D[empty]))<1e-7
  for i in empty:D[i,i]=1.;e[i]=0.
 return np.linalg.solve(D,e)

def run(date,s):
 target=OUT/f'day_{date}_{s}.json'
 if target.exists():print('EXISTS',target);return
 start=time.perf_counter();net,_=make_case(date);p=load(ROOT/f'results/uncertainty_v26/{date}',f'scenario_{s:02}')
 plan=load(ROOT/f'results/robust_v28/{date}/scenario_{s:02}','M4_proposed');common=load(ROOT/f'results/robust_v28/{date}','common');q=load(ROOT/f'results/robust_v28/{date}','NCCG')['q']
 warm=np.load(DEST/f'day_{date}_{s}.npz');c,_,ids=configure(net,p,plan,0,1);m=ACPieceModel(c,net)
 n,ng,nf=m.n,m.ng,m.nf;N=m.nx;X=ca.MX.sym('day',N,48);con=[];lbs=[];ubs=[];xl=[];xu=[];x0=[];caps=0;obj=0
 for t in range(48):
  c,original,ids=configure(net,p,plan,t,1);g=c['gen'];on=g[:,7]>0
  lb=np.r_[np.full(n,.95),np.full(n,-np.pi),g[:nf,9]*on[:nf]/100,g[:,4]*on/100,np.zeros(n),np.zeros(nf)]
  ub=np.r_[np.full(n,1.05),np.full(n,np.pi),g[:nf,8]*on[:nf]/100,g[:,3]*on/100,np.full(n,3.),net.generation_emissions(net.pmax)*on[:nf]/100.]
  ref=int(np.flatnonzero(c['bus'][:,1]==3)[0]);lb[n+ref]=ub[n+ref]=0.
  r={key:warm[key][t] for key in ['bus','gen','branch']};rho=density(net,r)
  init=np.r_[r['bus'][:,7],(r['bus'][:,8]-r['bus'][ref,8])*np.pi/180,r['gen'][:nf,1]/100,r['gen'][:,2]/100,np.maximum(rho,0.),net.generation_emissions(np.maximum(r['gen'][:nf,1],0.))/100.]
  par=np.r_[c['bus'][:,2:4].T.flatten()/100,original/100,(r['branch'][:,15]<-1e-8).astype(float),(r['branch'][:,13]<-1e-8).astype(float)]
  f,gg=m.problem_function(X[:,t],par);obj+=f/48.;con.append(gg[:-6]);lbs.extend(m.lbg[:-6]);ubs.extend(m.ubg[:-6]);caps+=gg[-6:]*p['dt_h']
  if t:
   con.append(X[2*n:2*n+nf,t]-X[2*n:2*n+nf,t-1]);ramp=net.ramp_mw_per_hour*p['dt_h']/100.;lbs.extend(-ramp);ubs.extend(ramp)
  xl.extend(lb);xu.extend(ub);x0.extend(init)
 con.append(caps);lbs.extend(np.full(6,-np.inf));ubs.extend((common['budget']+q-1e-6)/100.)
 problem=dict(x=ca.vec(X),f=obj,g=ca.vertcat(*con));solver=ca.nlpsol('whole_day','ipopt',problem,
  {'print_time':False,'ipopt.print_level':0,'ipopt.sb':'yes','ipopt.max_iter':500,'ipopt.max_cpu_time':300.,
   'ipopt.tol':1e-8,'ipopt.constr_viol_tol':1e-9,'ipopt.acceptable_tol':1e-7,'ipopt.bound_relax_factor':0.})
 print('AC_DAY_NLP_BUILT',date,s,'variables',N*48,'constraints',len(lbs),'seconds',time.perf_counter()-start,flush=True)
 sol=solver(x0=x0,lbx=xl,ubx=xu,lbg=lbs,ubg=ubs);stats=solver.stats();xx=np.array(sol['x']).reshape((N,48),order='F')
 rows=[];bs=[];gs=[];brs=[];allocation=np.zeros(6);newcost=0.;newem=0.;previous=None
 for t in range(48):
  c,original,_=configure(net,p,plan,t,1);x=xx[:,t];r=copy.deepcopy(c);r['bus'][:,7]=x[:n];r['bus'][:,8]=x[n:2*n]*180/np.pi
  r['gen'][:nf,1]=x[2*n:2*n+nf]*100;r['gen'][:,2]=x[2*n+nf:2*n+nf+ng]*100;r['gen'][:,5]=r['bus'][r['gen'][:,0].astype(int),7]
  r['branch']=np.pad(r['branch'],((0,0),(0,4)));r['branch'][:,13:17]=np.array([np.array(v).ravel() for v in m.flow(x)]).T
  r['success']=bool(stats['success']);r['f']=float(sol['f']);row=inspect(c,r,original);current=r['gen'][:nf,1]
  row.update(date=date,scenario=s,period=t,ramp_excess_mw=float(max(0.,np.max(abs(current-previous)-net.ramp_mw_per_hour*p['dt_h']))) if previous is not None else 0.)
  rho=density(net,r);allocation+=rho[net.industrial_buses]*plan['pl'][:,t]*p['dt_h']
  row['carbon_density_difference']=float(np.max(abs(rho-x[2*n+nf+ng:2*n+nf+ng+n])));previous=current;rows.append(row);bs.append(r['bus']);gs.append(r['gen']);brs.append(r['branch'])
  newcost+=net.generation_cost(np.maximum(current,0.)).sum()*p['dt_h'];newem+=net.generation_emissions(np.maximum(current,0.)).sum()*p['dt_h']
 excess=allocation-common['budget']-q
 record=dict(date=date,scenario=s,status=stats['return_status'],iterations=stats['iter_count'],elapsed_s=time.perf_counter()-start,
  passed=bool(all(r['limits_passed'] and r['ramp_excess_mw']<1e-5 for r in rows) and np.max(excess)<1e-5),rows=rows,
  original_quota_t=q,initial_budget_t=common['budget'],allocation_t=allocation,quota_excess_t=excess,
  restored_generation_cost_usd=float(newcost),restored_source_emissions_t=float(newem),commitment_added=[str(net.fossil.iloc[i]['GEN UID']) for i in ids if plan['commitment'][i]<.5])
 dump(target,record);np.savez_compressed(OUT/f'day_{date}_{s}.npz',bus=bs,gen=gs,branch=brs,pl=plan['pl'],pr=plan['pr'])
 print('AC_WHOLE_DAY',{k:v for k,v in record.items() if k not in ['rows','original_quota_t','initial_budget_t','allocation_t']},flush=True)

if __name__=='__main__':
 with threadpool_limits(limits=1):run(sys.argv[1],int(sys.argv[2]))
