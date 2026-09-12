"""Independent sparse polar AC restoration, verified against PYPOWER Y matrices."""
from ac_restore_v39 import *
_dll_handle=os.add_dll_directory(str(__import__("pathlib").Path(__import__("casadi").__file__).resolve().parent)) if os.name=='nt' else None
os.environ['PATH']=str(__import__("pathlib").Path(__import__("casadi").__file__).resolve().parent)+os.pathsep+os.environ.get('PATH','')
import casadi as ca
from pypower.makeYbus import makeYbus

class ACModel:
 def __init__(self,case,net,carbon_limited=False):
  self.case=case;self.net=net;self.carbon_limited=carbon_limited;n=len(case['bus']);ng=len(case['gen']);nf=len(net.gen)
  y,yf,yt=makeYbus(100.,case['bus'],case['branch'])
  vm=ca.SX.sym('vm',n);va=ca.SX.sym('va',n);pg=ca.SX.sym('pg',nf);qg=ca.SX.sym('qg',ng)
  params=ca.SX.sym('params',2*n+ng+(6 if carbon_limited else 0));pd=params[:n];qd=params[n:2*n];pref=params[2*n:2*n+ng]
  fullpg=ca.vertcat(pg,pref[nf:]);genmap=ca.DM(np.eye(n)[:,case['gen'][:,0].astype(int)])
  vr=vm*ca.cos(va);vi=vm*ca.sin(va)
  def power(y,vbr,vbi):
   ir=ca.DM(y.real.toarray())@vr-ca.DM(y.imag.toarray())@vi;ii=ca.DM(y.imag.toarray())@vr+ca.DM(y.real.toarray())@vi
   return vbr*ir+vbi*ii,vbi*ir-vbr*ii
  p,q=power(y,vr,vi);f=case['branch'][:,0].astype(int);t=case['branch'][:,1].astype(int)
  pf,qf=power(yf,vr[f],vi[f]);pt,qt=power(yt,vr[t],vi[t]);rates=ca.DM(case['branch'][:,5]/100.)
  balance=ca.vertcat(p-genmap@fullpg+pd,q-genmap@qg+qd)
  flows=ca.vertcat((pf**2+qf**2)/rates**2,(pt**2+qt**2)/rates**2)
  objective=ca.sumsqr((pg-pref[:nf])/ca.DM(np.maximum(net.pmax,10.)/100.))+1e-7*ca.sumsqr(vm-1.)
  x=ca.vertcat(vm,va,pg,qg);con=ca.vertcat(balance,flows)
  if carbon_limited:
   rho=ca.SX.sym('rho',n);em=[]
   for j in range(nf):
    if case['gen'][j,7]<.5:em.append(0.);continue
    xx=net.fuel_power_points[j];yy=net.fuel_heat_points[j];slopes=np.diff(yy)/np.diff(xx)
    heat=slopes[0]*(pg[j]*100-xx[0])+yy[0]
    for k in range(1,len(slopes)):heat=ca.fmax(heat,slopes[k]*(pg[j]*100-xx[k])+yy[k])
    em.append(heat*net.fuel_carbon_t_mmbtu[j]/100.)
   fm=ca.DM(np.eye(n)[:,f]);tm=ca.DM(np.eye(n)[:,t]);into_t=ca.fmax(-pt,0.);into_f=ca.fmax(-pf,0.)
   throughput=genmap@fullpg+tm@into_t+fm@into_f
   cbalance=throughput*rho-tm@(into_t*rho[f])-fm@(into_f*rho[t])-ca.DM(net.genmap)@ca.vertcat(*em)
   carbon_caps=rho[net.industrial_buses]*pd[net.industrial_buses]-params[-6:]
   x=ca.vertcat(x,rho);con=ca.vertcat(con,cbalance,carbon_caps)
  self.problem_function=ca.Function('ac_nlp',[x,params],[objective,con])
  self.solver=ca.nlpsol('restore','ipopt',dict(x=x,p=params,f=objective,g=con),
   {'print_time':False,'ipopt.print_level':0,'ipopt.sb':'yes','ipopt.max_iter':700,'ipopt.tol':1e-9,
    'ipopt.constr_viol_tol':1e-9,'ipopt.acceptable_tol':1e-8,'ipopt.bound_relax_factor':0.,'ipopt.max_cpu_time':60.})
  self.flow=ca.Function('flow',[x],[100*pf,100*qf,100*pt,100*qt])
  self.n=n;self.ng=ng;self.nf=nf;self.lbg=np.r_[np.zeros(2*n),np.full(2*len(f),-np.inf)];self.ubg=np.r_[np.zeros(2*n),np.ones(2*len(f))]
  if carbon_limited:self.lbg=np.r_[self.lbg,np.zeros(n),np.full(6,-np.inf)];self.ubg=np.r_[self.ubg,np.zeros(n+6)]
 def solve(self,c,original,carbon_caps=None):
  n,ng,nf=self.n,self.ng,self.nf;g=c['gen'];on=g[:,7]>0
  pmin=g[:nf,9]*on[:nf]/100;pmax=g[:nf,8]*on[:nf]/100
  qmin=g[:,4]*on/100;qmax=g[:,3]*on/100
  lb=np.r_[np.full(n,.95),np.full(n,-np.pi),pmin,qmin];ub=np.r_[np.full(n,1.05),np.full(n,np.pi),pmax,qmax]
  ref=int(np.flatnonzero(c['bus'][:,1]==3)[0]);lb[n+ref]=ub[n+ref]=0.
  x0=np.r_[c['bus'][:,7],np.zeros(n),np.clip(original[:nf]/100,pmin,pmax),np.clip(g[:,2]/100,qmin,qmax)]
  par=np.r_[c['bus'][:,2:4].T.flatten()/100,original/100]
  if self.carbon_limited:
   assert carbon_caps is not None
   lb=np.r_[lb,np.zeros(n)];ub=np.r_[ub,np.full(n,3.)];x0=np.r_[x0,np.full(n,.5)];par=np.r_[par,np.asarray(carbon_caps)/100]
  start=time.perf_counter();sol=self.solver(x0=x0,lbx=lb,ubx=ub,lbg=self.lbg,ubg=self.ubg,p=par)
  x=np.array(sol['x']).ravel();r=copy.deepcopy(c);r['bus'][:,7]=x[:n];r['bus'][:,8]=x[n:2*n]*180/np.pi
  r['gen'][:nf,1]=x[2*n:2*n+nf]*100;r['gen'][:,2]=x[2*n+nf:2*n+nf+ng]*100;r['gen'][:,5]=r['bus'][g[:,0].astype(int),7]
  r['branch']=np.pad(r['branch'],((0,0),(0,4)));r['branch'][:,13:17]=np.array([np.array(v).ravel() for v in self.flow(x)]).T
  stats=self.solver.stats();r['success']=bool(stats['success']);r['f']=float(sol['f']);row=inspect(c,r,original)
  row.update(elapsed_s=time.perf_counter()-start,solver_status=stats['return_status'],iterations=stats['iter_count'])
  return row,r

def probe():
 rows=[]
 for date in DATES:
  net,_=make_case(date);m=None
  for s,t in [(0,0),(0,24),(0,36),(4,24)]:
   p=load(ROOT/f'results/uncertainty_v26/{date}',f'scenario_{s:02}')
   plan=load(ROOT/f'results/robust_v28/{date}/scenario_{s:02}','M4_proposed');c,original=build(net,p,plan,t)
   if m is None:m=ACModel(c,net)
   row,r=m.solve(c,original);row.update(date=date,scenario=s,period=t)
   dump(DEST/f'ipopt_probe_{date}_{s}_{t}.json',dict(**row,bus=r['bus'],gen=r['gen'],branch=r['branch']))
   rows.append(row);print('AC_IPOPT',row,flush=True)
 dump(DEST/'ipopt_probe_summary.json',rows)

if __name__=='__main__':
 with threadpool_limits(limits=1):probe()
