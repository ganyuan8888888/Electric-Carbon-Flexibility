"""Lossless network and exact proportional-sharing carbon attribution.

Topology/electrical parameters: PYPOWER distribution of MATPOWER cases 39/118.
All renewable placements, costs and emission factors below are synthetic study
assumptions, NOT measurements or the original benchmark's fuel classification.
"""
from dataclasses import dataclass
from pathlib import Path
import json,time
import numpy as np
from scipy.linalg import solve
from pypower.case39 import case39
from pypower.case118 import case118
from pypower.ext2int import ext2int
from pypower.makeBdc import makeBdc
import cvxpy as cp

ROOT=__import__("repository_paths").ROOT

@dataclass
class Network:
    name:str
    bus:np.ndarray
    gen:np.ndarray
    branch:np.ndarray
    ptdf:np.ndarray
    flow_offset:np.ndarray
    incidence:np.ndarray
    genmap:np.ndarray
    renmap:np.ndarray
    loadmap:np.ndarray
    ren_buses:np.ndarray
    industrial_buses:np.ndarray
    emission:np.ndarray
    cost:np.ndarray
    pmin:np.ndarray
    pmax:np.ndarray
    capacity:np.ndarray
    industrial_base:np.ndarray

    def generation_emissions(self,pg):
        return self.emission*np.asarray(pg)
    def generation_emission_derivatives(self,pg):
        return self.emission.copy()
    def generation_cost(self,pg):
        return self.cost*np.asarray(pg)+.003*np.asarray(pg)**2

    def injections(self,pg,pr,pl,background):
        return self.genmap@pg+self.renmap@pr-self.loadmap@pl-background
    def flows(self,pg,pr,pl,background):
        return self.ptdf@self.injections(pg,pr,pl,background)+self.flow_offset
    def carbon(self,pg,pr,pl,background,derivatives=False):
        """Solve D rho = e; positive Pg/Pr/pl are generator/renewable/load MW.

        Jacobians use the exact local flow direction. They are local derivatives,
        not a global affine carbon-factor approximation or a marginal-emission
        dispatch derivative. Total generation emissions are calculated separately.
        """
        f=self.flows(pg,pr,pl,background)
        u=self.branch[:,0].astype(int);v=self.branch[:,1].astype(int)
        sender=np.where(f>=0,u,v);receiver=np.where(f>=0,v,u)
        pin=self.genmap@pg+self.renmap@pr
        np.add.at(pin,receiver,np.abs(f))
        D=np.diag(pin)
        np.add.at(D,(receiver,sender),-np.abs(f))
        e=self.genmap@self.generation_emissions(pg)
        active=pin>1e-9
        rho=np.zeros(len(pin));Da=D[np.ix_(active,active)]
        rho[active]=solve(Da,e[active],assume_a='gen')
        rate=rho[self.industrial_buses]*pl
        demand=background+self.loadmap@pl
        result={'rho':rho,'D':D,'e':e,'flow':f,'pin':pin,'industrial_carbon_rate':rate,
            'total_generation_emission_rate':float(e.sum()),
            'carbon_balance_residual':float(np.max(abs(D@rho-e))),
            'attribution_conservation_residual':float(rho@demand-e.sum()),
            'active':active}
        if derivatives:
            maps=np.c_[self.genmap,self.renmap,-self.loadmap]
            df=self.ptdf@maps
            dabs=np.sign(f)[:,None]*df
            nin,nx=maps.shape
            direct=np.c_[self.genmap,self.renmap,np.zeros_like(self.loadmap)]
            dD=np.zeros((nx,nin,nin))
            de=np.c_[self.genmap*self.generation_emission_derivatives(pg)[None,:],np.zeros_like(self.renmap),np.zeros_like(self.loadmap)]
            if derivatives=='fast':
                # Contract dD with rho before solving; exactly the same local
                # derivative, without materializing nx dense n-by-n matrices.
                rhs=de-direct*rho[:,None]
                np.add.at(rhs,receiver,-dabs*(rho[receiver]-rho[sender])[:,None])
                drho=np.zeros((nin,nx));drho[active]=solve(Da,rhs[active])
                dcarbon=pl[:,None]*drho[self.industrial_buses]
                dcarbon[:,-len(pl):]+=np.diag(rho[self.industrial_buses])
                result.update(de=de,drho=drho,dcarbon=dcarbon,df=df)
                return result
            for j in range(nx):
                diag=direct[:,j].copy();np.add.at(diag,receiver,dabs[:,j])
                dD[j]=np.diag(diag)
                np.add.at(dD[j],(receiver,sender),-dabs[:,j])
            rhs=de-np.einsum('knm,m->nk',dD,rho)
            drho=np.zeros((nin,nx));drho[active]=solve(Da,rhs[active])
            dcarbon=pl[:,None]*drho[self.industrial_buses]
            dcarbon[:,-len(pl):]+=np.diag(rho[self.industrial_buses])
            result.update(dD=dD,de=de,drho=drho,dcarbon=dcarbon,df=df)
        return result

def make_network(name='case39'):
    raw={'case39':case39,'case118':case118}[name]()
    case=ext2int(raw)
    bus,gen,branch=case['bus'],case['gen'],case['branch']
    b,bf,pbus,pf=makeBdc(case['baseMVA'],bus,branch)
    b=b.toarray();bf=bf.toarray();n=len(bus)
    slack=int(np.flatnonzero(bus[:,1]==3)[0]);keep=np.array([i for i in range(n) if i!=slack])
    theta=np.zeros((n,n));theta[np.ix_(keep,keep)]=np.linalg.inv(b[np.ix_(keep,keep)])/case['baseMVA']
    h=case['baseMVA']*bf@theta
    offset=case['baseMVA']*pf-h@(case['baseMVA']*pbus)
    incidence=np.zeros((n,len(branch)))
    incidence[branch[:,0].astype(int),np.arange(len(branch))]=1
    incidence[branch[:,1].astype(int),np.arange(len(branch))]=-1
    gm=np.zeros((n,len(gen)));gm[gen[:,0].astype(int),np.arange(len(gen))]=1
    if name=='case39':
        rb=np.array([3,7,15,20,25,28]);ib=np.array([3,15,23]);ibase=np.array([300,260,220.])
    else:
        rb=np.array([9,25,39,48,65,79,88,102]);ib=np.array([14,26,49,58,89,110]);ibase=np.array([90,110,100,90,100,90.])
    rm=np.eye(n)[:,rb];lm=np.eye(n)[:,ib]
    # Original IEEE 118 branch ratings are mostly placeholders; use explicit
    # synthetic thermal limits and report them as a study modification.
    cap=branch[:,5].copy()
    cap=np.where((cap<=0)|(cap>3000),350. if name=='case118' else 1000.,cap)
    pm=gen[:,8].copy();ix=np.arange(len(gen));coal=ix%3!=1
    emission=np.where(coal,.88+.015*(ix%5),.36+.01*(ix%4))
    cost=np.where(coal,31.+1.8*(ix%5),48.+2*(ix%4))
    pmin=np.where(coal,.32,.1)*pm
    return Network(name,bus,gen,branch,h,offset,incidence,gm,rm,lm,rb,ib,emission,cost,pmin,pm,cap,ibase)

def profiles(net,periods=48,scenario=0,seed=20260908):
    rng=np.random.default_rng(seed+scenario)
    t=(np.arange(periods)+.5)*24/periods
    shape=.83+.09*np.cos(2*np.pi*(t-18)/24)+.07*np.exp(-((t-12)/3)**2)
    original=net.bus[:,2].copy()
    # Industrial loads replace part of native demand, preserving their baseline.
    background0=original-net.loadmap@net.industrial_base
    if np.min(background0)<0:
        # In the 118-bus synthetic extension the industrial sites can exceed
        # original local demand. The additional demand is explicitly retained.
        background0=np.maximum(background0,0)
    background=background0[:,None]*shape[None,:]
    ren=[]
    total_peak=net.bus[:,2].sum()*.72
    for j in range(len(net.ren_buses)):
        if j%2:
            x=np.maximum(0,np.sin(np.pi*(t-6)/12))**1.5
            x*=1-.12*np.exp(-((t-13-scenario*.5)/1.7)**2)
        else:
            x=.62+.22*np.cos(2*np.pi*(t+2*j+scenario)/24)+.1*np.sin(2*np.pi*t/5+j)
            noise=rng.normal(0,.035,periods);x=np.clip(x+noise,.1,1)
        ren.append(x*total_peak/len(net.ren_buses))
    ren=np.asarray(ren)*(1+.1*scenario)
    return {'time_h':t,'dt_h':24/periods,'background_mw':background,'available_renewable_mw':ren,
        'seed':seed+scenario,'scenario':scenario,'source':'Explicit synthetic load/wind/PV scenario, not recorded weather.'}

def baseline_dispatch(net,prof,industrial=None):
    """Convex network dispatch with fixed industrial trajectories, for validation."""
    start=time.perf_counter();T=len(prof['time_h']);ng=len(net.gen);nr=len(net.ren_buses)
    pg=cp.Variable((ng,T));pr=cp.Variable((nr,T))
    pl=np.tile(net.industrial_base[:,None],(1,T)) if industrial is None else industrial
    inj=net.genmap@pg+net.renmap@pr-net.loadmap@pl-prof['background_mw']
    flow=net.ptdf@inj+net.flow_offset[:,None]
    cons=[pg>=net.pmin[:,None],pg<=net.pmax[:,None],pr>=0,pr<=prof['available_renewable_mw'],
        cp.sum(inj,axis=0)==0,flow<=net.capacity[:,None],flow>=-net.capacity[:,None]]
    ramp=net.pmax[:,None]*.12*prof['dt_h']
    cons.extend([pg[:,1:]-pg[:,:-1]<=ramp,pg[:,:-1]-pg[:,1:]<=ramp])
    dt=prof['dt_h'];curt=prof['available_renewable_mw']-pr
    obj=dt*(cp.sum(cp.multiply(net.cost[:,None],pg))+.003*cp.sum_squares(pg)+150*cp.sum(curt))
    prob=cp.Problem(cp.Minimize(obj),cons)
    prob.solve(solver='CLARABEL',tol_gap_abs=1e-5,tol_feas=1e-9,max_iter=150)
    if prob.status not in ['optimal','optimal_inaccurate']:raise RuntimeError((net.name,prob.status))
    rho=[];rates=[];cfres=[];ares=[]
    for t in range(T):
        ca=net.carbon(pg.value[:,t],pr.value[:,t],pl[:,t],prof['background_mw'][:,t])
        rho.append(ca['rho']);rates.append(ca['industrial_carbon_rate']);cfres.append(ca['carbon_balance_residual']);ares.append(ca['attribution_conservation_residual'])
    result={'pg':pg.value,'pr':pr.value,'pl':pl,'rho':np.asarray(rho).T,'carbon_rate':np.asarray(rates).T,
        'flow':flow.value,'objective':float(prob.value),'elapsed_s':time.perf_counter()-start,
        'curtailment_mwh':float(np.sum(prof['available_renewable_mw']-pr.value)*dt),
        'generation_emission_t':float(np.sum(net.emission[:,None]*pg.value)*dt),
        'industrial_attributed_t':np.sum(rates,axis=0)*dt,
        'max_nodal_balance_mw':float(np.max(np.abs(net.incidence@flow.value-inj.value))),
        'max_carbon_balance_tph':max(cfres),'max_attribution_conservation_tph':float(max(abs(np.asarray(ares)))),
        'solver_status':prob.status}
    return result

if __name__=='__main__':
    out=ROOT/'results/network';out.mkdir(exist_ok=True,parents=True)
    for name in ['case39','case118']:
        net=make_network(name);prof=profiles(net);result=baseline_dispatch(net,prof)
        np.savez_compressed(out/f'{name}_baseline.npz',**{k:v for k,v in result.items() if isinstance(v,np.ndarray)},
            time_h=prof['time_h'],background=prof['background_mw'],available=prof['available_renewable_mw'])
        report={k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
        report['industrial_attributed_t']=result['industrial_attributed_t'].tolist()
        report['assumptions']={'scenario':prof['source'],'generator_fuel_cost_emission':'synthetic','flow_model':'lossless DC',
            'pmin':'0.32 Pmax for assigned coal; 0.10 Pmax for assigned gas','ramp':'0.12 Pmax per hour',
            'renewable_peak_mw':float(net.bus[:,2].sum()*.72),'industrial_baseline_mw':net.industrial_base.tolist(),
            'renewable_bus_external':(net.ren_buses+1).tolist(),'industrial_bus_external':(net.industrial_buses+1).tolist()}
        (out/f'{name}_baseline.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        print(name,json.dumps(report),flush=True)
