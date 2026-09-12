"""Independently reconstruct all final AC flows, balances and daily carbon totals."""
from evidence_core_v37 import *
from ac_existing_units_v39 import configure
from pypower.makeYbus import makeYbus

def audit():
 summary=json.loads((ROOT/'revision_v39/experiments/ac_complete_summary.json').read_text());assert summary['complete'] and summary['all_checks_passed']
 results=[];maxflow=0.;maxpower=0.;maxcarbon=0.;maxquota=-np.inf;maxramp=0.
 for date in DATES:
  net,_=make_case(date);nf=len(net.gen);common=load(ROOT/f'results/robust_v28/{date}','common');q=load(ROOT/f'results/robust_v28/{date}','NCCG')['q']
  for day in [d for d in summary['days'] if d['date']==date]:
   s=day['scenario'];source=ROOT/day['source'];arrays=ROOT/day['arrays']
   assert hashlib.sha256(source.read_bytes()).hexdigest()==day['source_sha256'];assert hashlib.sha256(arrays.read_bytes()).hexdigest()==day['arrays_sha256']
   saved=json.loads(source.read_text());a=np.load(arrays);p=load(ROOT/f'results/uncertainty_v26/{date}',f'scenario_{s:02}');plan=load(ROOT/f'results/robust_v28/{date}/scenario_{s:02}','M4_proposed')
   assert np.array_equal(a['pl'],plan['pl']) and np.array_equal(a['pr'],plan['pr']);assert np.array_equal(saved['original_quota_t'],q)
   assert np.array_equal(saved['initial_budget_t'],common['budget'])
   allocation=np.zeros(6);cost=0.;emissions=0.;previous=None;points=[]
   for t in range(48):
    c,original,_=configure(net,p,plan,t,1);b,g,br=(a[k][t] for k in ['bus','gen','branch']);on=g[:,7]>0
    assert np.array_equal(br[:,:13],c['branch'][:,:13]);assert np.array_equal(b[:,[0,2,3,4,5,9,10,11,12]],c['bus'][:,[0,2,3,4,5,9,10,11,12]])
    assert np.array_equal(g[:,7],c['gen'][:,7]) and np.max(abs(g[nf:,1]-original[nf:]))<1e-7
    Y,Yf,Yt=makeYbus(c['baseMVA'],b,br);V=b[:,7]*np.exp(1j*b[:,8]*np.pi/180);f=br[:,0].astype(int);to=br[:,1].astype(int)
    Sf=V[f]*np.conj(Yf@V)*c['baseMVA'];St=V[to]*np.conj(Yt@V)*c['baseMVA'];expected=np.column_stack([Sf.real,Sf.imag,St.real,St.imag])
    flow_error=float(abs(expected-br[:,13:17]).max());maxflow=max(maxflow,flow_error);assert flow_error<1e-5
    injected=np.zeros(len(b),complex);np.add.at(injected,g[on,0].astype(int),g[on,1]+1j*g[on,2])
    residual=float(abs(V*np.conj(Y@V)*c['baseMVA']-injected+b[:,2]+1j*b[:,3]).max());maxpower=max(maxpower,residual)
    loading=np.maximum(abs(Sf),abs(St))/br[:,5];assert b[:,7].min()>=.95-1e-6 and b[:,7].max()<=1.05+1e-6 and loading.max()<=1+1e-6 and residual<1e-5
    assert np.max(g[on,1]-c['gen'][on,8])<1e-5 and np.max(c['gen'][on,9]-g[on,1])<1e-5
    assert np.max(g[on,2]-g[on,3])<1e-5 and np.max(g[on,4]-g[on,2])<1e-5
    current=g[:nf,1]
    if previous is not None:maxramp=max(maxramp,float(np.max(abs(current-previous)-net.ramp_mw_per_hour*p['dt_h'])))
    previous=current;pin=np.zeros(len(b));np.add.at(pin,g[on,0].astype(int),g[on,1]);D=np.zeros((len(b),len(b)))
    for k,(i,j) in enumerate(zip(f,to)):
     pf,pt=Sf[k].real,St[k].real
     if pf>=0 and pt<0:pin[j]+=-pt;D[j,i]+=pt
     elif pt>=0 and pf<0:pin[i]+=-pf;D[i,j]+=pf
    D[np.arange(len(b)),np.arange(len(b))]=pin;e=net.genmap@net.generation_emissions(np.maximum(current,0.))
    for i in np.flatnonzero(pin<=1e-8):assert abs(e[i])<1e-7 and abs(D[i]).max()<1e-7;D[i,i]=1.;e[i]=0.
    rho=np.linalg.solve(D,e);assert rho.min()>-1e-7;carbon_res=float(abs(D@rho-e).max());maxcarbon=max(maxcarbon,carbon_res)
    allocation+=rho[net.industrial_buses]*plan['pl'][:,t]*p['dt_h'];cost+=net.generation_cost(np.maximum(current,0.)).sum()*p['dt_h'];emissions+=e.sum()*p['dt_h']
    points.append(dict(period=t,vmin=float(b[:,7].min()),vmax=float(b[:,7].max()),max_loading=float(loading.max()),loss_mw=float((Sf.real+St.real).sum()),power_residual_mva=residual,carbon_residual_tph=carbon_res))
   stored=np.array(saved.get('allocation_t',saved.get('restored_industrial_allocation_t')));assert abs(allocation-stored).max()<1e-5
   assert abs(cost-day['restored_generation_cost_usd'])<1e-4 and abs(emissions-day['restored_source_emissions_t'])<1e-4
   excess=float((allocation-common['budget']-q).max());assert excess<1e-5;maxquota=max(maxquota,excess)
   results.append(dict(date=date,scenario=s,quota_excess_t=excess,points=points,cost_usd=float(cost),emissions_t=float(emissions)))
 assert maxramp<1e-5
 out=dict(passed=True,days=15,operating_points=720,industrial_and_renewable_active_schedules_identical=True,first_stage_quota_identical=True,
  original_branch_parameters_and_ratings_identical=True,original_voltage_and_generator_limits_preserved=True,
  max_flow_reconstruction_residual_mva=maxflow,max_power_balance_residual_mva=maxpower,max_carbon_balance_residual_tph=maxcarbon,max_ramp_excess_mw=maxramp,max_daily_quota_excess_t=maxquota,results=results)
 dump(ROOT/'revision_v39/experiments/ac_independent_audit.json',out);print({k:v for k,v in out.items() if k!='results'});return out

if __name__=='__main__':
 with threadpool_limits(limits=1):audit()
