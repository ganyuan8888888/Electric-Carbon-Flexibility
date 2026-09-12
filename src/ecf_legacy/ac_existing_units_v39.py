"""Declared AC corrective commitment of existing bus-307 synchronous units.

No voltage/thermal limit, network branch or renewable/industrial active-power
schedule is altered. Changes of commitment and synchronous dispatch are retained.
This experiment cannot be reported as validation of the unchanged original plan.
"""
from ac_ipopt_v39 import *

def configure(net,p,plan,t,units=1,previous=None):
 c,original=build(net,p,plan,t)
 ids=np.flatnonzero(net.fossil['GEN UID'].isin(['307_CT_1','307_CT_2']).to_numpy())[:units]
 c['gen'][ids,7]=1
 if previous is not None:
  n=len(net.gen);ramp=net.ramp_mw_per_hour*p['dt_h'];on=c['gen'][:n,7]>0
  c['gen'][:n,8]=np.where(on,np.minimum(c['gen'][:n,8],previous+ramp),0.)
  c['gen'][:n,9]=np.where(on,np.maximum(c['gen'][:n,9],previous-ramp),0.)
 return c,original,ids

def probe(units):
 rows=[]
 for date in DATES:
  net,_=make_case(date);m=None
  for s,t in [(0,0),(0,24),(0,36),(4,24)]:
   p=load(ROOT/f'results/uncertainty_v26/{date}',f'scenario_{s:02}')
   plan=load(ROOT/f'results/robust_v28/{date}/scenario_{s:02}','M4_proposed');c,original,ids=configure(net,p,plan,t,units)
   if m is None:m=ACModel(c,net)
   row,r=m.solve(c,original);row.update(date=date,scenario=s,period=t,required_units=units,commitment_changed_ids=ids[plan['commitment'][ids]<.5])
   dump(DEST/f'existing_{units}_{date}_{s}_{t}.json',dict(**row,bus=r['bus'],gen=r['gen'],branch=r['branch']))
   rows.append(row);print('AC_EXISTING',row,flush=True)
 dump(DEST/f'existing_{units}_probe_summary.json',rows)

def carbon(net,r,pl):
 g=r['gen'];br=r['branch'];n=len(net.bus);D=np.zeros((n,n));pin=np.zeros(n);on=g[:,7]>0
 np.add.at(pin,g[on,0].astype(int),g[on,1])
 for f,t,pf,pt in br[:,[0,1,13,15]]:
  f,t=int(f),int(t)
  if pf>=0 and pt<0:pin[t]+=-pt;D[t,f]+=pt
  elif pt>=0 and pf<0:pin[f]+=-pf;D[f,t]+=pf
 D[np.arange(n),np.arange(n)]=pin
 assert pin.min()>1e-8
 e=net.genmap@net.generation_emissions(np.maximum(g[:len(net.gen),1],0.));rho=np.linalg.solve(D,e)
 assert rho.min()>-1e-7
 return rho[net.industrial_buses]*pl,float(abs(D@rho-e).max())

def full():
 rows=[];days=[]
 for date in DATES:
  net,_=make_case(date);m=None;common=load(ROOT/f'results/robust_v28/{date}','common')
  q=load(ROOT/f'results/robust_v28/{date}','NCCG')['q']
  for s in range(5):
   dest=DEST/f'day_{date}_{s}.json'
   if dest.exists():d=json.loads(dest.read_text());days.append(d);rows.extend(d['rows']);continue
   p=load(ROOT/f'results/uncertainty_v26/{date}',f'scenario_{s:02}')
   plan=load(ROOT/f'results/robust_v28/{date}/scenario_{s:02}','M4_proposed');day=[];pg=[];bus=[];gen=[];branch=[];alloc=[];oldcost=0.;newcost=0.;oldem=0.;newem=0.;previous=None
   for t in range(48):
    c,original,ids=configure(net,p,plan,t,1,previous)
    if m is None:m=ACModel(c,net)
    row,r=m.solve(c,original);row.update(date=date,scenario=s,period=t)
    nf=len(net.gen);current=r['gen'][:nf,1].copy()
    row['ramp_excess_mw']=float(max(0.,np.max(abs(current-previous)-net.ramp_mw_per_hour*p['dt_h']))) if previous is not None else 0.
    if row['limits_passed']:
     a,res=carbon(net,r,plan['pl'][:,t]);alloc.append(a*p['dt_h']);row['carbon_balance_residual_tph']=res
    else:alloc.append(np.full(6,np.nan))
    oldcost+=net.generation_cost(plan['pg'][:,t]).sum()*p['dt_h'];newcost+=net.generation_cost(np.maximum(current,0.)).sum()*p['dt_h']
    oldem+=net.generation_emissions(plan['pg'][:,t]).sum()*p['dt_h'];newem+=net.generation_emissions(np.maximum(current,0.)).sum()*p['dt_h']
    day.append(row);pg.append(current);bus.append(r['bus']);gen.append(r['gen']);branch.append(r['branch'])
    if row['limits_passed']:previous=current
   allocation=np.sum(alloc,axis=0)
   d=dict(date=date,scenario=s,rows=day,passed=sum(x['limits_passed'] for x in day),
    commitment_added=[str(net.fossil.iloc[i]['GEN UID']) for i in ids if plan['commitment'][i]<.5],
    original_generation_cost_usd=float(oldcost),restored_generation_cost_usd=float(newcost),
    original_source_emissions_t=float(oldem),restored_source_emissions_t=float(newem),
    original_quota_t=q,initial_budget_t=common['budget'],restored_industrial_allocation_t=allocation,
    original_quota_excess_t=allocation-common['budget']-q)
   dump(dest,d);np.savez_compressed(DEST/f'day_{date}_{s}.npz',bus=bus,gen=gen,branch=branch,pg=pg,pl=plan['pl'],pr=plan['pr'])
   rows.extend(day);days.append(d);print('AC_DAY',date,s,d['passed'],'cost_delta',newcost-oldcost,'carbon_cap_excess',d['original_quota_excess_t'],flush=True)
 dump(DEST/'full_summary.json',dict(rows=rows,days=[{k:v for k,v in d.items() if k!='rows'} for d in days],
  passed=sum(x['limits_passed'] and x['ramp_excess_mw']<=1e-5 for x in rows),total=len(rows),
  maximum_balance_residual=max(x['balance_residual_mva'] for x in rows)))

if __name__=='__main__':
 with threadpool_limits(limits=1):
  full() if sys.argv[1]=='full' else probe(int(sys.argv[1]))
