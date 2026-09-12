"""Reproducible validation on frozen v36 models and v28 dispatch records."""
from pathlib import Path
import sys,json,hashlib,time,copy,os
ROOT=__import__("repository_paths").ROOT
import numpy as np
from threadpoolctl import threadpool_limits
from dispatch_result_io import load,save
from green_direct_v18 import make_case
from electric_carbon_region_v22 import batched_carbon,enclose
OUT=ROOT/'revision_v37';RESULT=OUT/'experiments';DATES=['2020-04-07','2020-10-17','2020-11-29']

def dump(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name('.'+path.name+'.'+str(os.getpid())+'.tmp')
 temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x)),encoding='utf8')
 temp.replace(path)

def protocol():
 OUT.mkdir(exist_ok=True);RESULT.mkdir(exist_ok=True)
 p={'version':37,'baseline_version':36,'dates':DATES,'seed':2026091201,
 'frozen_sections':'Abstract, introduction, Sections II-IV, existing equation objects, existing numerical comparisons and references.',
 'holdout':'12 donor-day errors per target day; excluded from v26 medoid selection. Do not claim independence across overlapping seasonal windows.',
 'pressure':'200 complete-day source/load trajectories per target day. Draw one training donor jointly for all sources and 48 periods; multiply its entire error vector by an independent lognormal multiplier with mean 1 and log-sd 0.15. This is a model-based stress test, not 600 independent observed days.',
 'recourse':'Freeze existing first-stage allowances and commitment. Search original full-day industrial trajectories; validate original carbon balance, carbon caps and electrical limits. Solver failure is not a proof of nonlinear infeasibility.',
 'region':'Same complete 24h industrial trajectory, original RTS network and nonlinear carbon equations; two-dimensional chart-coordinate/carbon-budget section. Separate chart certification loss from trajectory-library coverage.',
 'region_grid':[101,201],'region_depths':[0,1,2,3,4,5,6],
 'ac':'All 15 original M4 plans x 48 periods. Original RTS R/X/B/taps and generator Q limits; declared 0.98 lagging industrial power factor, existing lossless PCC X=0.02. No load shedding or post-hoc network reinforcement.',
 'scaling':'Nested 5/10/20/40 scenario experiments; identical finite cells across solvers, one thread, common tolerances. Preprocessing separately measured.',
 'reporting':'Preserve all attempted cases and distinguish certified, rejected, unresolved and timed-out records; no outcome-driven filtering.'}
 f=RESULT/'protocol.json'
 if f.exists():assert json.loads(f.read_text(encoding='utf8'))==p
 else:dump(f,p)
 files=[ROOT/'revision_v36/manuscript_main_v36.md',ROOT/'revision_v36/supplement_v36.md']
 for d in DATES:
  base=ROOT/'results/robust_v28'/d
  files += [base/'NCCG.json',base/'common.json']+[base/f'scenario_{s:02}/cells.json' for s in range(5)]
 hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
 baseline=RESULT/'baseline_hashes.json'
 if baseline.exists():assert json.loads(baseline.read_text())==hashes,'Frozen source changed'
 else:dump(baseline,hashes)

def carbon_eval(net,p,a,b,alpha):
 pg=a['pg']+alpha*(b['pg']-a['pg']);pr=a['pr']+alpha*(b['pr']-a['pr'])
 D,e,inv,rho,flows=batched_carbon(net,p,pg,pr,a['pl'])
 alloc=(rho[:,net.industrial_buses].T*a['pl']).sum(axis=1)*p['dt_h']
 return alloc,float(abs(np.einsum('tij,tj->ti',D,rho)-e).max())

def region():
 protocol();rows=[]
 for date in DATES:
  net,_=make_case(date);base=ROOT/'results/robust_v28'/date;common=load(base,'common');B=common['budget']
  p=load(ROOT/f'results/uncertainty_v26/{date}','scenario_00');net.fixed_commitment=common['commitment']
  data=json.loads((base/'scenario_00/cells.json').read_text());pairs=[]
  for c in data['cells']:
   pair=tuple(c['endpoints'])
   if pair[0]!=pair[1] and pair not in pairs:pairs.append(pair)
  # Deterministic first three distinct charts, not selected for their accuracy.
  for index,pair in enumerate(pairs[:3]):
   dest=RESULT/'region'/f'{date}_{index}.json'
   if dest.exists():rows.append(json.loads(dest.read_text()));continue
   began=time.perf_counter();a,b=[load(base/'scenario_00',n) for n in pair]
   assert np.max(abs(a['pl']-b['pl']))<1e-7
   alpha=np.linspace(0,1,201);alloc=[];res=0.
   for x in alpha:
    v,r=carbon_eval(net,p,a,b,x);alloc.append(v);res=max(res,r)
   alloc=np.array(alloc);truth=(alloc/B).max(axis=1)
   # A fixed carbon budget axis (80--120% of the original allowance) is used for all charts.
   budget=np.linspace(.8,1.2,201);actual=budget[:,None]>=truth[None,:]-1e-10
   levels=[];max_excess=-np.inf
   for depth in range(7):
    bounds=[];valid_length=0.;unresolved=0
    for j in range(2**depth):
     l=j/2**depth;r=(j+1)/2**depth;c=enclose(net,p,a,b,l,r)
     if c is None:unresolved+=1;bounds.append(None);continue
     valid_length+=r-l;bounds.append(c)
     for x in [l,l+(r-l)*.2113248654,(l+r)/2,l+(r-l)*.7886751346,r]:
      v,_=carbon_eval(net,p,a,b,x);max_excess=max(max_excess,float((v-c['upper']).max()))
    certified=np.full(201,np.inf)
    for j,c in enumerate(bounds):
     if c is not None:
      mask=(alpha>=j/2**depth-1e-12)&(alpha<=(j+1)/2**depth+1e-12)
      certified[mask]=np.minimum(certified[mask],max(c['upper']/B))
    accepted=budget[:,None]>=certified[None,:]-1e-10
    fp=int(np.count_nonzero(accepted&~actual));assert fp==0,(date,index,depth,fp)
    intersect=int(np.count_nonzero(accepted&actual));den=int(actual.sum())
    gap=certified-truth;finite=np.isfinite(gap)
    levels.append({'depth':depth,'cells':2**depth,'enclosed_cells':2**depth-unresolved,'enclosed_length':valid_length,
      'coverage':intersect/den if den else None,'false_feasible_points':fp,'certified_grid_points':int(accepted.sum()),
      'mean_budget_gap':float(gap[finite].mean()) if finite.any() else None,'max_budget_gap':float(gap[finite].max()) if finite.any() else None,
      'grid101_coverage':int((accepted&actual)[::2,::2].sum())/int(actual[::2,::2].sum()) if actual[::2,::2].any() else None,
      'boundary':certified.tolist()})
    print('REGION',date,index,'depth',depth,'coverage',levels[-1]['coverage'],flush=True)
   row={'date':date,'chart':index,'endpoints':pair,'alpha':alpha,'budget_axis':budget,'original_boundary':truth,'levels':levels,
        'max_sample_enclosure_excess_t':max_excess,'max_carbon_balance_residual':res,'elapsed_s':time.perf_counter()-began}
   assert max_excess<1e-5
   dump(dest,row);rows.append(row)
 dump(RESULT/'region_summary.json',rows)

def ac_case(net,p,plan,t):
 from pypower.idx_bus import BUS_TYPE,PD,QD,VM,VA,VMAX,VMIN
 bus=net.bus.copy();bus[:,BUS_TYPE]=1;bus[:,VM]=1.;bus[:,VA]=0.;bus[:,VMAX]=1.05;bus[:,VMIN]=.95
 bus[:73,VM]=net.busdata['V Mag'];bus[:73,VA]=net.busdata['V Angle']
 bus[:73,4]=net.busdata['MW Shunt G'];bus[:73,5]=net.busdata['MVAR Shunt B']
 bus[:,PD]=p['background_mw'][:,t]+net.loadmap@plan['pl'][:,t]
 ratio=np.divide(net.busdata['MVAR Load'],net.busdata['MW Load'],out=np.zeros(73),where=net.busdata['MW Load'].to_numpy()!=0)
 bus[:73,QD]=p['background_mw'][:73,t]*ratio
 bus[net.industrial_buses,QD]=plan['pl'][:,t]*np.tan(np.arccos(.98))
 gen=[]
 for j,row in net.fossil.iterrows():
  g=np.zeros(21);g[0]=net.gen[j,0];g[1]=plan['pg'][j,t];g[3]=row['QMax MVAR'];g[4]=row['QMin MVAR'];g[5]=row['V Setpoint p.u.'];g[6]=100
  g[7]=plan['commitment'][j];g[8]=row['PMax MW'];g[9]=row['PMin MW'];gen.append(g)
 for j,row in net.renewable.iterrows():
  g=np.zeros(21);g[0]=net.ren_buses[j];g[1]=plan['pr'][j,t];g[3]=row['QMax MVAR'];g[4]=row['QMin MVAR'];g[5]=row['V Setpoint p.u.'];g[6]=100
  # PV split at bus 215 preserves the original source reactive capability as well as its active power.
  share=net.direct_profile_mapping[j][1];g[3:5]*=share
  g[7]=1 if g[1]>1e-8 else 0;g[8]=row['PMax MW'];g[9]=0;gen.append(g)
 # Retain the source system's synchronous condensers, which have no active
 # dispatch variables in the DC model but remain essential AC equipment.
 lookup={int(b):i for i,b in enumerate(net.busdata['Bus ID'])}
 for _,row in net.gendata[net.gendata['Unit Type']=='SYNC_COND'].iterrows():
  g=np.zeros(21);g[0]=lookup[int(row['Bus ID'])];g[3]=row['QMax MVAR'];g[4]=row['QMin MVAR'];g[5]=row['V Setpoint p.u.'];g[6]=100;g[7]=1
  gen.append(g)
 gen=np.asarray(gen);on=np.flatnonzero(gen[:,7]>0)
 for g in gen[on]:
  if g[3]>g[4]:bus[int(g[0]),BUS_TYPE]=2;bus[int(g[0]),VM]=g[5]
 # Co-located fixed-Q renewable injections follow the regulating bus setpoint;
 # they must not reset a synchronous machine's voltage to 1.0 p.u.
 for g in gen:g[5]=bus[int(g[0]),VM]
 # Largest online fossil headroom; deterministic before the AC solve.
 head=gen[:len(net.gen),8]-gen[:len(net.gen),1];head[gen[:len(net.gen),7]==0]=-np.inf
 slack=int(np.argmax(head));bus[int(gen[slack,0]),BUS_TYPE]=3
 branch=net.branch.copy();branch[:,11]=-360;branch[:,12]=360
 return {'version':'2','baseMVA':100.,'bus':bus,'gen':gen,'branch':branch},slack

def ac():
 protocol()
 from pypower.runpf import runpf
 from pypower.ppoption import ppoption
 from pypower.idx_bus import BUS_TYPE,VM,VA
 from pypower.idx_gen import GEN_BUS,PG,QG,QMAX,QMIN,GEN_STATUS
 rows=[]
 for date in DATES:
  dest=RESULT/'ac_final'/f'{date}.json'
  if dest.exists():rows.extend(json.loads(dest.read_text()));continue
  net,_=make_case(date);day=[]
  for s in range(5):
   p=load(ROOT/f'results/uncertainty_v26/{date}',f'scenario_{s:02}')
   plan=load(ROOT/f'results/robust_v28/{date}/scenario_{s:02}','M4_proposed')
   for t in range(48):
    case,slack=ac_case(net,p,plan,t);initialpg=case['gen'][:,PG].copy();clamped={};success=0;result=None
    for k in range(len(case['gen'])+1):
     result,success=runpf(case,ppoption(VERBOSE=0,OUT_ALL=0,PF_TOL=1e-8,PF_MAX_IT=30))
     if not success:break
     g=result['gen'];on=g[:,GEN_STATUS]>0;bad=np.flatnonzero(on&((g[:,QG]>g[:,QMAX]+1e-6)|(g[:,QG]<g[:,QMIN]-1e-6)))
     bad=[i for i in bad if i not in clamped]
     if not bad:break
     # Remove limited generators from voltage regulation, represent their fixed
     # P/Q as negative demand, and restore their outputs after convergence.
     # This prevents the next PF solve from redistributing Q to already-clamped units.
     for i in bad:
      qfixed=float(np.clip(g[i,QG],g[i,QMIN],g[i,QMAX]));pfixed=float(g[i,PG]);bidx=int(g[i,GEN_BUS])
      clamped[i]=(pfixed,qfixed);result['gen'][i,QG]=qfixed;result['gen'][i,GEN_STATUS]=0
      result['bus'][bidx,2]-=pfixed;result['bus'][bidx,3]-=qfixed
     for b in set(int(g[i,GEN_BUS]) for i in bad):
      active=[i for i in range(len(g)) if result['gen'][i,GEN_STATUS]>0 and int(g[i,GEN_BUS])==b]
      if not active:result['bus'][b,BUS_TYPE]=1
     if not np.any(result['bus'][:,BUS_TYPE]==3):
      candidates=[i for i in range(len(net.gen)) if result['gen'][i,GEN_STATUS]>0 and result['bus'][int(g[i,GEN_BUS]),BUS_TYPE]==2]
      if not candidates:success=0;break
      slack=max(candidates,key=lambda i:g[i,8]-g[i,PG]);result['bus'][int(g[slack,GEN_BUS]),BUS_TYPE]=3
     case=result
    row={'date':date,'scenario':s,'period':t,'converged':bool(success),'q_limited_generators':len(clamped)}
    if success:
     for i,(pfixed,qfixed) in clamped.items():
      bidx=int(result['gen'][i,GEN_BUS]);result['bus'][bidx,2]+=pfixed;result['bus'][bidx,3]+=qfixed
      result['gen'][i,PG]=pfixed;result['gen'][i,QG]=qfixed;result['gen'][i,GEN_STATUS]=1
     # A reference bus may contain several synchronous machines. Redistribute
     # its loss-balancing increment among those machines in proportion to their
     # original headroom. This preserves nodal injection and all AC flows.
     for bidx in set(result['gen'][:,GEN_BUS].astype(int)):
      ids=np.array([i for i in range(len(net.gen)) if result['gen'][i,GEN_STATUS]>0 and int(result['gen'][i,GEN_BUS])==bidx],int)
      if len(ids)<2:continue
      delta=float(np.sum(result['gen'][ids,PG]-initialpg[ids]))
      room=(result['gen'][ids,8]-initialpg[ids]) if delta>=0 else (initialpg[ids]-result['gen'][ids,9])
      if abs(delta)<=room.sum()+1e-6 and room.sum()>1e-9:
       result['gen'][ids,PG]=initialpg[ids]+delta*room/room.sum()
     br=result['branch'];g=result['gen'];v=result['bus'][:,VM];loading=np.maximum(np.hypot(br[:,13],br[:,14]),np.hypot(br[:,15],br[:,16]))/br[:,5]
     on=g[:,GEN_STATUS]>0;qexc=max(0,float((g[on,QG]-g[on,QMAX]).max()),float((g[on,QMIN]-g[on,QG]).max()))
     pexc=max(0,float((g[on,PG]-g[on,8]).max()),float((g[on,9]-g[on,PG]).max()))
     row.update(vmin=float(v.min()),vmax=float(v.max()),max_line_loading=float(loading.max()),loss_mw=float((br[:,13]+br[:,15]).sum()),
       generation_adjustment_mw=float((g[:,PG]-initialpg).sum()),max_q_excess_mvar=qexc,max_p_excess_mw=pexc,
       operating_limits_passed=bool(v.min()>=.95-1e-6 and v.max()<=1.05+1e-6 and loading.max()<=1+1e-6 and qexc<1e-5 and pexc<1e-5))
    day.append(row)
   print('AC',date,s,'converged',sum(x['converged'] for x in day[-48:]),'limits',sum(x.get('operating_limits_passed',False) for x in day[-48:]),flush=True)
  dump(dest,day);rows.extend(day)
 dump(RESULT/'ac_final_summary.json',rows)

if __name__=='__main__':
 with threadpool_limits(limits=1):
  {'protocol':protocol,'region':region,'ac':ac}[sys.argv[1]]()
