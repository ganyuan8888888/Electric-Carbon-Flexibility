"""Smooth, conservative AC/carbon restoration inside the observed flow cell.

Actual receiving-end powers are used. Flow-direction inequalities make every
directed transfer exact. Fuel epigraphs upper-bound the original convex fuel
curves; accepted plans are independently checked with the original curves.
"""
from ac_existing_units_v39 import *

class ACPieceModel:
 def __init__(self,case,net):
  self.n=n=len(case['bus']);self.ng=ng=len(case['gen']);self.nf=nf=len(net.gen);L=len(case['branch']);self.nx=3*n+ng+2*nf
  y,yf,yt=makeYbus(100.,case['bus'],case['branch']);f=case['branch'][:,0].astype(int);t=case['branch'][:,1].astype(int)
  vm=ca.SX.sym('v',n);va=ca.SX.sym('a',n);pg=ca.SX.sym('p',nf);qg=ca.SX.sym('q',ng);rho=ca.SX.sym('rho',n);em=ca.SX.sym('em',nf)
  par=ca.SX.sym('par',2*n+ng+2*L);pd=par[:n];qd=par[n:2*n];pref=par[2*n:2*n+ng];mt=par[-2*L:-L];mf=par[-L:]
  fullpg=ca.vertcat(pg,pref[nf:]);gm=ca.DM(np.eye(n)[:,case['gen'][:,0].astype(int)]);vr=vm*ca.cos(va);vi=vm*ca.sin(va)
  def power(y,rr,ii):
   ir=ca.DM(y.real.toarray())@vr-ca.DM(y.imag.toarray())@vi;im=ca.DM(y.imag.toarray())@vr+ca.DM(y.real.toarray())@vi
   return rr*ir+ii*im,ii*ir-rr*im
  p,q=power(y,vr,vi);pf,qf=power(yf,vr[f],vi[f]);pt,qt=power(yt,vr[t],vi[t]);rates=ca.DM(case['branch'][:,5]/100.)
  electric=ca.vertcat(p-gm@fullpg+pd,q-gm@qg+qd);flows=ca.vertcat((pf**2+qf**2)/rates**2,(pt**2+qt**2)/rates**2)
  fm=ca.DM(np.eye(n)[:,f]);tm=ca.DM(np.eye(n)[:,t]);into_t=-pt*mt;into_f=-pf*mf
  pin=gm@fullpg+tm@into_t+fm@into_f
  cb=pin*rho-tm@(into_t*rho[f])-fm@(into_f*rho[t])-ca.DM(net.genmap)@em
  fuel=[]
  for j in range(nf):
   if case['gen'][j,7]<.5:continue
   xx,yy=net.fuel_power_points[j],net.fuel_heat_points[j]
   for k,slope in enumerate(np.diff(yy)/np.diff(xx)):
    fuel.append(em[j]-(slope*(pg[j]*100-xx[k])+yy[k])*net.fuel_carbon_t_mmbtu[j]/100.)
  direction=ca.vertcat((1-2*mf)*pf,(1-2*mt)*pt)
  allocated=rho[net.industrial_buses]*pd[net.industrial_buses]
  x=ca.vertcat(vm,va,pg,qg,rho,em);con=ca.vertcat(electric,flows,cb,ca.vertcat(*fuel),direction,allocated)
  obj=ca.sumsqr((pg-pref[:nf])/ca.DM(np.maximum(net.pmax,10.)/100.))+1e-7*ca.sumsqr(vm-1.)+1e-7*ca.sum1(em)
  self.problem_function=ca.Function('ac_piece',[x,par],[obj,con]);self.flow=ca.Function('piece_flow',[x],[100*pf,100*qf,100*pt,100*qt])
  self.lbg=np.r_[np.zeros(2*n),np.full(2*L,-np.inf),np.zeros(n+len(fuel)+2*L),np.full(6,-np.inf)]
  self.ubg=np.r_[np.zeros(2*n),np.ones(2*L),np.zeros(n),np.full(len(fuel)+2*L,np.inf),np.zeros(6)]

