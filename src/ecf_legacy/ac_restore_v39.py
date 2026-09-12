"""AC feasibility restoration with fixed commitment, renewable and industrial P.

This is a distinct, explicitly recorded corrective dispatch, not a relabeling of
the original lossless schedule as AC feasible. Original case files are immutable.
"""
from evidence_core_v37 import *
from pypower.runopf import runopf
from pypower.ppoption import ppoption

DEST=ROOT/'revision_v39/experiments/ac_restore'

def build(net,p,plan,t,previous=None):
 c,_=ac_case(net,p,plan,t);original=c['gen'][:,1].copy();n=len(net.gen)
 c['bus'][:,7]=np.clip(c['bus'][:,7],.96,1.04)
 c['gen'][:,5]=np.clip(c['gen'][:,5],.96,1.04)
 # Preserve every non-dispatchable source and the exact renewable schedule.
 c['gen'][n:,8]=original[n:];c['gen'][n:,9]=original[n:]
 if previous is not None:
  ramp=net.ramp_mw_per_hour*p['dt_h']
  c['gen'][:n,8]=np.minimum(c['gen'][:n,8],previous+ramp)
  c['gen'][:n,9]=np.maximum(c['gen'][:n,9],previous-ramp)
 # Minimize normalized squared synchronous redispatch, with original units MW.
 weight=1/np.maximum(net.pmax,10.)**2
 cost=np.zeros((len(original),7));cost[:,0]=2;cost[:,3]=3
 cost[:n,4]=weight;cost[:n,5]=-2*weight*original[:n];cost[:n,6]=weight*original[:n]**2
 c['gencost']=cost
 return c,original

def inspect(c,r,original):
 b,g,br=r['bus'],r['gen'],r['branch'];on=g[:,7]>0
 flow=np.maximum(np.hypot(br[:,13],br[:,14]),np.hypot(br[:,15],br[:,16]));loading=flow/br[:,5]
 pexc=max(0.,float(np.max(g[on,1]-c['gen'][on,8])),float(np.max(c['gen'][on,9]-g[on,1])))
 qexc=max(0.,float(np.max(g[on,2]-g[on,3])),float(np.max(g[on,4]-g[on,2])))
 from pypower.makeYbus import makeYbus
 y,_,_=makeYbus(c['baseMVA'],b,br);v=b[:,7]*np.exp(1j*np.pi*b[:,8]/180)
 s=v*np.conj(y@v)*c['baseMVA']
 injected=np.zeros(len(b),complex)
 np.add.at(injected,g[on,0].astype(int),g[on,1]+1j*g[on,2])
 residual=np.max(abs(s-injected+b[:,2]+1j*b[:,3]))
 passed=bool(r['success'] and b[:,7].min()>=.95-1e-6 and b[:,7].max()<=1.05+1e-6 and loading.max()<=1+1e-6 and max(pexc,qexc,residual)<1e-5)
 return dict(converged=bool(r['success']),limits_passed=passed,vmin=float(b[:,7].min()),vmax=float(b[:,7].max()),
  max_loading=float(loading.max()),max_p_excess_mw=pexc,max_q_excess_mvar=qexc,balance_residual_mva=float(residual),
  generation_change_mw=float((g[:,1]-original).sum()),max_abs_generator_change_mw=float(abs(g[:,1]-original).max()),
  loss_mw=float((br[:,13]+br[:,15]).sum()),objective=float(r['f']))

def solve(net,p,plan,t,previous=None):
 c,original=build(net,p,plan,t,previous);start=time.perf_counter()
 r=runopf(c,ppoption(VERBOSE=0,OUT_ALL=0,OPF_VIOLATION=1e-8,PDIPM_FEASTOL=1e-9,PDIPM_GRADTOL=1e-8,PDIPM_COMPTOL=1e-8,PDIPM_MAX_IT=200))
 row=inspect(c,r,original);row.update(period=t,elapsed_s=time.perf_counter()-start)
 return row,r

def probe():
 rows=[]
 for date in DATES:
  net,_=make_case(date)
  for s,t in [(0,0),(0,24),(0,36),(4,24)]:
   p=load(ROOT/f'results/uncertainty_v26/{date}',f'scenario_{s:02}')
   plan=load(ROOT/f'results/robust_v28/{date}/scenario_{s:02}','M4_proposed')
   row,r=solve(net,p,plan,t);row.update(date=date,scenario=s)
   dump(DEST/f'probe_{date}_{s}_{t}.json',dict(**row,bus=r['bus'],gen=r['gen'],branch=r['branch']))
   rows.append(row);print('AC_RESTORE',row,flush=True)
 dump(DEST/'probe_summary.json',rows)

if __name__=='__main__':
 with threadpool_limits(limits=1):probe()
