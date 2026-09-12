"""Explicit behind-PCC factory nodes with source-conserving PV reconnection.

All methods receive the same topology and source availability. PV is relocated,
never duplicated. Half of source 215_PV_1 serves each area-2 factory. This is a
declared RTS case extension, not an assertion about these real assets.
"""
import numpy as np
import pandas as pd
from rts_network import make_rts_network,rts_profiles

def make_case(date,mode='direct'):
    net=make_rts_network();prof=rts_profiles(net,date)
    if mode=='grid':return net,prof
    assert mode=='direct'
    n=len(net.bus);K=len(net.industrial_buses);parents=net.industrial_buses.copy()
    net.public_industrial_buses=parents;factories=n+np.arange(K)
    net.bus=np.pad(net.bus,((0,K),(0,0)));net.bus[n:,0]=factories;net.bus[n:,1]=1
    for col in [6,9]:net.bus[n:,col]=net.bus[parents,col]
    new=np.zeros((K,13));new[:,0]=parents;new[:,1]=factories;new[:,3]=.02
    new[:,5:8]=250;new[:,10]=1
    net.branch=np.vstack([net.branch,new]);nb=len(net.branch)
    inc=np.zeros((n+K,nb));ix=np.arange(nb)
    inc[net.branch[:,0].astype(int),ix]=1;inc[net.branch[:,1].astype(int),ix]=-1
    tap=np.where(net.branch[:,8]==0,1,net.branch[:,8]);sus=100/(net.branch[:,3]*tap)
    B=(inc*sus)@inc.T;inv=np.zeros_like(B);inv[1:,1:]=np.linalg.inv(B[1:,1:])
    net.ptdf=(sus[:,None]*inc.T)@inv;net.incidence=inc;net.flow_offset=np.zeros(nb)
    net.capacity=np.r_[net.capacity,np.full(K,250.)];net.genmap=np.pad(net.genmap,((0,K),(0,0)))
    # Each tuple is (original renewable source, fraction, factory).
    sources=[('113_PV_1',1.,0),('119_PV_1',1.,1),('215_PV_1',.5,2),('215_PV_1',.5,3),('313_PV_1',1.,4),('319_PV_1',1.,5)]
    lookup={v:i for i,v in enumerate(net.renewable['GEN UID'])};chosen={lookup[u] for u,_,_ in sources}
    oldavail=prof['available_renewable_mw'];oldmin=prof['minimum_renewable_mw'];rows=[];avail=[];minimum=[];buses=[];mapping=[];direct=[]
    for j,row in net.renewable.iterrows():
        if j in chosen:continue
        rows.append(row.to_dict());avail.append(oldavail[j]);minimum.append(oldmin[j]);buses.append(net.ren_buses[j]);mapping.append((j,1.))
    for uid,share,k in sources:
        j=lookup[uid];row=net.renewable.iloc[j].to_dict();row['GEN UID']=uid+f'_direct{k}';row['PMax MW']*=share
        direct.append(len(rows));rows.append(row);avail.append(share*oldavail[j]);minimum.append(share*oldmin[j]);buses.append(factories[k]);mapping.append((j,share))
    net.renewable=pd.DataFrame(rows);net.ren_buses=np.asarray(buses);net.renmap=np.eye(n+K)[:,buses]
    net.loadmap=np.eye(n+K)[:,factories];net.industrial_buses=factories
    net.must_take=~net.renewable['Unit Type'].isin(['WIND','PV']).to_numpy();net.curtailable=~net.must_take
    net.direct_source_indices=np.asarray(direct);net.direct_profile_mapping=mapping
    prof['available_renewable_mw']=np.asarray(avail);prof['minimum_renewable_mw']=np.asarray(minimum)
    prof['background_mw']=np.pad(prof['background_mw'],((0,K),(0,0)))
    assert np.max(abs(prof['available_renewable_mw'].sum(axis=0)-oldavail.sum(axis=0)))<1e-9
    net.name='rts_gmlc_direct_v18'
    prof['direct_connection']={'sources':sources,'grid_capacity_mw':250.,'lossless':True,'storage':False,'source_energy_preserved':True,'boundary':'operational generator emissions, physical attribution'}
    return net,prof

def recovery_metrics(net,prof,result,baseline,window_start_h=20.):
    p=result['pl'];p0=baseline['pl'];dt=prof['dt_h'];rec=prof['time_h']>=window_start_h
    plus=np.maximum(p-p0,0)*rec;minus=np.maximum(p0-p,0)*rec
    gross=((p0-p)*(~rec)).sum(axis=1)*dt;net_e=(p0-p).sum(axis=1)*dt
    ep=plus.sum(axis=1)*dt;em=minus.sum(axis=1)*dt
    assert np.max(abs(net_e-gross+ep-em))<1e-8
    rho=result['rho'][net.industrial_buses]
    r={'window_start_h':window_start_h,'gross_mwh':gross,'recovery_plus_mwh':ep,'recovery_minus_mwh':em,'net_factory_mwh':net_e,'recovery_allocated_t':(rho*plus).sum(axis=1)*dt}
    if hasattr(net,'direct_source_indices'):
        direct=result['pr'][net.direct_source_indices];direct0=baseline['pr'][net.direct_source_indices]
        purchase=np.maximum(p-direct,0);export=np.maximum(direct-p,0)
        parentrho=result['rho'][net.public_industrial_buses]
        expected=np.divide(parentrho*purchase,direct+purchase,out=np.zeros_like(p),where=direct+purchase>1e-9)
        r.update(direct_power=direct,grid_purchase=purchase,grid_export=export,
            rho_factory=rho,rho_public=parentrho,
            net_grid_mwh=net_e+(direct-direct0).sum(axis=1)*dt,
            factory_mixing_residual=float(np.max(abs(expected-rho))),
            pcc_balance_residual=float(np.max(abs(direct+purchase-p-export))))
        assert r['factory_mixing_residual']<1e-7 and r['pcc_balance_residual']<1e-7
    return r
