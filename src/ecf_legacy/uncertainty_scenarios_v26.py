"""Freeze joint DA/RT error scenarios before optimizing any method.

Finite two-stage trajectory uncertainty; no probabilistic coverage assertion.
Whole daily wind/PV/load errors are sampled jointly, never independently.
"""
from pathlib import Path
import sys,json,hashlib,datetime,copy
ROOT=__import__("repository_paths").ROOT
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from rts_network import DATA,make_rts_network
from green_direct_v18 import make_case
from dispatch_result_io import save,load
DATES=['2020-04-07','2020-10-17','2020-11-29']
OUT=ROOT/'results/uncertainty_v26'

def annual():
    result={}; hashes={}
    for kind,tail in [('Load','regional_Load'),('WIND','wind'),('PV','pv')]:
        arrays={}
        for mode in ['DAY_AHEAD','REAL_TIME']:
            path=DATA/f'RTS_Data/timeseries_data_files/{kind}/{mode}_{tail}.csv'
            df=pd.read_csv(path); hashes[str(path.relative_to(DATA))]=hashlib.sha256(path.read_bytes()).hexdigest()
            if mode=='DAY_AHEAD':
                assert len(df)==366*24
                a=np.repeat(df.iloc[:,4:].to_numpy().reshape(366,24,-1),2,axis=1)
            else:
                assert len(df)==366*288
                a=df.iloc[:,4:].to_numpy().reshape(366,48,6,-1).mean(axis=2)
            arrays[mode]=a
        result[kind]=dict(columns=list(df.columns[4:]),forecast=arrays['DAY_AHEAD'],error=arrays['REAL_TIME']-arrays['DAY_AHEAD'])
    return result,hashes

def medoids(x,k):
    dist=cdist(x,x,metric='cityblock')/x.shape[1]
    chosen=[int(np.argmin(dist.sum(axis=1)))]
    while len(chosen)<k:chosen.append(int(np.argmax(dist[:,chosen].min(axis=1))))
    for _ in range(50):
        labels=dist[:,chosen].argmin(axis=1);new=[]
        for j,c in enumerate(chosen):
            ix=np.flatnonzero(labels==j)
            new.append(int(ix[np.argmin(dist[np.ix_(ix,ix)].sum(axis=1))]) if len(ix) else c)
        if new==chosen:break
        chosen=new
    return sorted(chosen),float(dist[:,chosen].min(axis=1).mean())

def generate():
    OUT.mkdir(exist_ok=True)
    protocol=dict(version=26,dates=DATES,source_commit='3ece0d3725c844056132393ee252b3083dd4eab4',
      uncertainty='Finite joint whole-day forecast-error trajectories; uncertainty revealed before second-stage recourse.',
      seasonal_radius_days=30,training='Seasonal days excluding target and every fifth calendar day.',
      validation='Every fifth seasonal calendar day; none used to choose medoids.',
      reduction='Four deterministic L1 medoids plus zero-error nominal trajectory; equal feature normalization by training standard deviation with 1% rated-capacity floor.',
      sampling='Wind, PV and three regional load errors use the same historical day and unchanged half-hour ordering.',
      clipping='Renewables clipped to [0, source nameplate]; regional demand clipped at zero. Clipping quantities retained.',
      prescribed='Hydro, rooftop PV, nuclear and the common commitment follow the existing case.',
      carbon_allowance='97% of the existing common economic constant-production reference attribution.',
      maximum_contract_purchase_fraction=.20,contract_buy_usd_t=25.,contract_sell_usd_t=20.,
      comparison='Original reference method names and abbreviations retained. Shared case adaptations described in one settings paragraph. No outcome-driven removal of baseline constraints.',
      certificate='Network interpolation only within one complete industrial trajectory; original nonlinear carbon enclosure required.',
      optimizer='Nested CCG on a frozen finite certified recourse library; full extensive-form MILP cross-check.',
      claims='Finite model optimality only. Actual nonlinear execution and physical source emissions evaluated separately.')
    pp=OUT/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol,'Do not silently revise a frozen protocol.'
    else:pp.write_text(json.dumps(protocol,indent=2),encoding='utf8')
    datasets,hashes=annual();origin=make_rts_network()
    start=datetime.date(2020,1,1)
    for date in DATES:
        dest=OUT/date;dest.mkdir(exist_ok=True)
        if (dest/'manifest.json').exists():continue
        day=(datetime.date.fromisoformat(date)-start).days
        seasonal=np.array([d for d in range(366) if 0<min(abs(d-day),366-abs(d-day))<=30])
        train=seasonal[(seasonal+1)%5!=0];holdout=seasonal[(seasonal+1)%5==0]
        features=[];scales={}
        for kind,d in datasets.items():
            if kind=='Load':rated=np.max(d['forecast'][train],axis=(0,1))
            else:
                byid=origin.renewable.set_index('GEN UID')['PMax MW']
                rated=np.array([byid[c] for c in d['columns']])
            sigma=np.maximum(d['error'][train].std(axis=0),.01*rated[None,:])
            assert sigma.min()>0
            scales[kind]=sigma;features.append((d['error'][train]/sigma).reshape(len(train),-1))
        chosen,distortion=medoids(np.concatenate(features,axis=1),4)
        donors=[None]+[int(train[i]) for i in chosen]
        net,p=make_case(date);profiles=[];rows=[]
        for sid,donor in enumerate(donors+list(map(int,holdout))):
            q=copy.deepcopy(p); raw={}; clipped=0.
            for kind,d in datasets.items():
                a=d['forecast'][day].copy()
                if donor is not None:a+=d['error'][donor]
                raw[kind]=a
            loadraw=raw['Load'];loads=np.maximum(loadraw,0);clipped+=float(abs(loadraw-loads).sum())
            for area in [1,2,3]:
                mask=net.busdata['Area'].to_numpy()==area
                weights=net.busdata.loc[mask,'MW Load'].to_numpy();ix=np.flatnonzero(mask)
                q['background_mw'][ix]=weights[:,None]/weights.sum()*loads[:,datasets['Load']['columns'].index(str(area))][None,:]
            for j,g in net.renewable.iterrows():
                kind=g['Unit Type']
                if kind not in ['WIND','PV']:continue
                old,fraction=net.direct_profile_mapping[j];uid=origin.renewable.iloc[old]['GEN UID']
                a=fraction*raw[kind][:,datasets[kind]['columns'].index(uid)]
                b=np.clip(a,0,g['PMax MW']);clipped+=float(abs(a-b).sum())
                q['available_renewable_mw'][j]=b
            q['minimum_renewable_mw']=q['available_renewable_mw']*net.must_take[:,None]
            q['scenario_id']=sid;q['error_donor_date']=None if donor is None else str(start+datetime.timedelta(days=donor))
            q['scenario_role']='optimization' if sid<len(donors) else 'held_out_validation'
            q['source']='Paired RTS-GMLC day-ahead forecast and whole-day joint forecast error.'
            norm=0. if donor is None else float(sum(np.abs(d['error'][donor]/scales[kind]).sum() for kind,d in datasets.items()))
            q['normalized_error_budget']=norm;q['physical_clipping_mwh']=clipped*p['dt_h']
            save(dest,f'scenario_{sid:02}',q);profiles.append(q)
            rows.append({k:q[k] for k in ['scenario_id','error_donor_date','scenario_role','normalized_error_budget','physical_clipping_mwh']})
        manifest=dict(date=date,optimization_scenarios=len(donors),held_out_scenarios=len(holdout),training_days=[str(start+datetime.timedelta(days=int(i))) for i in train],
            gamma=max(r['normalized_error_budget'] for r in rows[:len(donors)]),medoid_mean_standardized_L1_distance=distortion,scenarios=rows,source_files_sha256=hashes,protocol_sha256=hashlib.sha256(pp.read_bytes()).hexdigest())
        (dest/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
        print(date,'frozen',len(donors),'optimization +',len(holdout),'held-out scenarios',flush=True)

if __name__=='__main__':generate()
