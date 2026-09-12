"""RTS-GMLC native topology, fuel curves and aligned public chronology.

Declared study extension: six industrial sites are added at buses 113,118,213,
218,313,318. The existing demand is not relabeled as measured industrial demand.
Common fixed commitment is selected before the four-method comparison; source
Pmin and ramp limits apply. Rooftop PV, hydro and nuclear profiles are prescribed; only wind and
utility PV are curtailed. CSP and the 50-MW battery are excluded consistently.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from network_model import Network

ROOT=__import__("repository_paths").ROOT
DATA=ROOT/'data/RTS-GMLC-v0.2.3'

class RTSNetwork(Network):
    def fuel_heat(self,pg,derivative=False):
        values=[]
        for p,x,y in zip(pg,self.fuel_power_points,self.fuel_heat_points):
            if p<=1e-8:values.append(0.);continue
            j=int(np.clip(np.searchsorted(x,p,side='right')-1,0,len(x)-2))
            slope=(y[j+1]-y[j])/(x[j+1]-x[j])
            values.append(slope if derivative else y[j]+slope*(p-x[j]))
        return np.array(values)
    def generation_emissions(self,pg):return self.fuel_heat(pg)*self.fuel_carbon_t_mmbtu
    def generation_emission_derivatives(self,pg):return self.fuel_heat(pg,True)*self.fuel_carbon_t_mmbtu
    def generation_cost(self,pg):return self.fuel_heat(pg)*self.fuel_price+self.vom*np.asarray(pg)

def make_rts_network(industrial_mw=150.):
    busdata=pd.read_csv(DATA/'RTS_Data/SourceData/bus.csv');gendata=pd.read_csv(DATA/'RTS_Data/SourceData/gen.csv')
    linedata=pd.read_csv(DATA/'RTS_Data/SourceData/branch.csv');n=len(busdata);lookup={b:i for i,b in enumerate(busdata['Bus ID'])}
    fossil=gendata[gendata['Fuel'].isin(['Coal','NG','Oil'])].reset_index(drop=True)
    renewable=gendata[gendata['Unit Type'].isin(['WIND','PV','RTPV','HYDRO','ROR','NUCLEAR'])].reset_index(drop=True)
    bus=np.zeros((n,13));bus[:,0]=np.arange(n);bus[:,1]=1;bus[0,1]=3
    bus[:,2]=busdata['MW Load'];bus[:,3]=busdata['MVAR Load'];bus[:,6]=busdata['Area'];bus[:,9]=busdata['BaseKV']
    branch=np.zeros((len(linedata),13));branch[:,0]=linedata['From Bus'].map(lookup);branch[:,1]=linedata['To Bus'].map(lookup)
    for col,key in [(2,'R'),(3,'X'),(4,'B'),(5,'Cont Rating'),(6,'LTE Rating'),(7,'STE Rating'),(8,'Tr Ratio')]:branch[:,col]=linedata[key]
    branch[:,10]=1
    inc=np.zeros((n,len(branch)));inc[branch[:,0].astype(int),np.arange(len(branch))]=1;inc[branch[:,1].astype(int),np.arange(len(branch))]=-1
    tap=np.where(branch[:,8]==0,1,branch[:,8]);susceptance=100/(branch[:,3]*tap)
    B=inc@np.diag(susceptance)@inc.T
    inv=np.zeros((n,n));inv[1:,1:]=np.linalg.inv(B[1:,1:])
    ptdf=np.diag(susceptance)@inc.T@inv
    gb=fossil['Bus ID'].map(lookup).to_numpy();rb=renewable['Bus ID'].map(lookup).to_numpy()
    ib=np.array([lookup[x] for x in [113,118,213,218,313,318]])
    gm=np.eye(n)[:,gb];rm=np.eye(n)[:,rb];lm=np.eye(n)[:,ib]
    gen=np.zeros((len(fossil),21));gen[:,0]=gb;gen[:,8]=fossil['PMax MW'];gen[:,9]=fossil['PMin MW']
    xp=[];hp=[]
    for _,row in fossil.iterrows():
        frac=np.array([row[f'Output_pct_{j}'] for j in range(5)],float);frac=frac[np.isfinite(frac)]
        power=frac*row['PMax MW'];heat=np.zeros(len(power));heat[0]=power[0]*row['HR_avg_0']/1000
        for j in range(1,len(power)):heat[j]=heat[j-1]+(power[j]-power[j-1])*row[f'HR_incr_{j}']/1000
        assert np.all(np.diff(power)>0)
        slopes=np.diff(heat)/np.diff(power)
        assert np.all(np.diff(slopes)>-1e-7),('nonconvex source fuel curve',row['GEN UID'])
        xp.append(power);hp.append(heat)
    factors=fossil['Emissions CO2 Lbs/MMBTU'].to_numpy(float)*.00045359237
    emission=np.array([y[-1]/x[-1] for x,y in zip(xp,hp)])*factors
    price=fossil['Fuel Price $/MMBTU'].to_numpy(float);vom=fossil['VOM'].to_numpy(float)
    cost=np.array([y[-1]/x[-1] for x,y in zip(xp,hp)])*price+vom
    net=RTSNetwork('rts_gmlc',bus,gen,branch,ptdf,np.zeros(len(branch)),inc,gm,rm,lm,rb,ib,
                   emission,cost,fossil['PMin MW'].to_numpy(float),fossil['PMax MW'].to_numpy(float),
                   linedata['Cont Rating'].to_numpy(float),np.full(len(ib),industrial_mw))
    net.fuel_power_points=xp;net.fuel_heat_points=hp;net.fuel_carbon_t_mmbtu=factors
    net.fuel_price=price;net.vom=vom;net.ramp_mw_per_hour=fossil['Ramp Rate MW/Min'].to_numpy(float)*60
    net.busdata=busdata;net.gendata=gendata;net.fossil=fossil;net.renewable=renewable
    net.must_take=~renewable['Unit Type'].isin(['WIND','PV']).to_numpy()
    net.curtailable=~net.must_take
    return net

def rts_profiles(net,date='2020-11-29',minutes=30):
    assert minutes%5==0 and 1440%minutes==0
    year,month,day=map(int,date.split('-'));factor=minutes//5
    files={'Load':'Load/REAL_TIME_regional_Load.csv','WIND':'WIND/REAL_TIME_wind.csv','PV':'PV/REAL_TIME_pv.csv',
           'RTPV':'RTPV/REAL_TIME_rtpv.csv','HYDRO':'Hydro/REAL_TIME_hydro.csv','ROR':'Hydro/REAL_TIME_hydro.csv'}
    datasets={}
    for category,relative in files.items():
        df=pd.read_csv(DATA/'RTS_Data/timeseries_data_files'/relative)
        df=df[(df.Year==year)&(df.Month==month)&(df.Day==day)]
        assert len(df)==288
        datasets[category]=df.iloc[:,4:].groupby(np.arange(288)//factor).mean()
    load=datasets['Load'];T=len(load);background=np.zeros((len(net.bus),T))
    for area in [1,2,3]:
        mask=net.busdata['Area'].to_numpy()==area;weights=net.busdata.loc[mask,'MW Load'].to_numpy()
        background[mask]=weights[:,None]/weights.sum()*load[str(area)].to_numpy()[None,:]
    available=[]
    for _,g in net.renewable.iterrows():
        if g['Unit Type']=='NUCLEAR':v=np.full(T,g['PMax MW'])
        else:v=datasets[g['Unit Type']][g['GEN UID']].to_numpy()
        available.append(v)
    available=np.asarray(available)
    return {'time_h':(np.arange(T)+.5)*minutes/60,'dt_h':minutes/60,'background_mw':background,
            'available_renewable_mw':available,'minimum_renewable_mw':available*net.must_take[:,None],
            'date':date,'source':'RTS-GMLC v0.2.3 original chronological profiles, six industrial loads added explicitly.',
            'source_commit':'3ece0d3725c844056132393ee252b3083dd4eab4'}

if __name__=='__main__':
    net=make_rts_network();p=rts_profiles(net)
    print('buses,branches,fossil,nonfossil',len(net.bus),len(net.branch),len(net.gen),len(net.renewable))
    print('native demand min/max',p['background_mw'].sum(axis=0).min(),p['background_mw'].sum(axis=0).max())
    print('all-on fossil minimum',net.pmin.sum(),'noncurtailable max',p['minimum_renewable_mw'].sum(axis=0).max())
    print('emission intensity full load min/max',net.emission.min(),net.emission.max())
