"""Spatial, conservative, discrete electrothermal benchmark for one cell.

This is a disclosed reduced benchmark, not a reproduction of an industrial CFD
cell. Conductivities and bath/ledge/shell heat-transfer coefficients are taken
from Wong et al. (2023), DOI 10.1007/s11663-022-02709-w, Table II. Geometry follows
the 11 x 2.7 m mechanical benchmark. Inventories, capacities, heat-loss closure,
temperature coefficient and linearized liquidus below are study assumptions.
All powers are W internally, current kA at the public interface, time seconds.
The state contains bath sensible enthalpy, shell enthalpy and ledge thickness.
Backward Euler enforces their joint energy identity without a hidden heat source.
"""
from dataclasses import dataclass
from pathlib import Path
import json, time
import numpy as np
from scipy.optimize import root

ROOT=__import__("repository_paths").ROOT

@dataclass
class ThermalCell:
    nx:int=4
    ny:int=3
    length:float=11.
    width:float=2.7
    bath_depth:float=.2
    nominal_current:float=180.
    nominal_acd:float=.043
    nominal_temperature:float=960.
    nominal_liquidus:float=950.
    ambient:float=25.
    bath_density:float=2100.
    bath_cp:float=1500.
    metal_mass:float=14000.
    metal_cp:float=1100.
    shell_capacity_total:float=12e6
    ledge_density:float=2100.
    fusion_heat:float=450e3
    ledge_conductivity:float=1.3
    wall_resistance:float=.01
    bath_ledge_h:float=600.
    shell_h:float=22.
    effective_conductivity:float=1000.
    v_reversible:float=1.75
    resistance_mohm:float=.0135
    external_resistance_mohm:float=.0005
    temperature_resistivity:float=.0025
    current_efficiency:float=.94
    spatial_loss_variation:float=.06

    def __post_init__(self):
        self.n=self.nx*self.ny
        self.dx=self.length/self.nx;self.dy=self.width/self.ny
        self.bath_mass0=self.length*self.width*self.bath_depth*self.bath_density/self.n
        # Each patch is assigned an equal share of sidewall exchange. This is
        # a reduced lateral partition, not the exact boundary mesh of a real pot.
        self.area=np.full(self.n,2*(self.length+self.width)*.4/self.n)
        flux=self.bath_ledge_h*(self.nominal_temperature-self.nominal_liquidus)
        self.shell0=self.ambient+flux/self.shell_h
        self.ledge0=self.ledge_conductivity*((self.nominal_liquidus-self.shell0)/flux-self.wall_resistance)
        self.laplacian=np.zeros((self.n,self.n))
        for x in range(self.nx):
            for y in range(self.ny):
                v=x*self.ny+y
                for xn,yn,d,a in [(x+1,y,self.dx,self.dy*self.bath_depth),(x,y+1,self.dy,self.dx*self.bath_depth)]:
                    if xn<self.nx and yn<self.ny:
                        w=xn*self.ny+yn;k=self.effective_conductivity*a/d
                        self.laplacian[v,v]+=k;self.laplacian[w,w]+=k
                        self.laplacian[v,w]-=k;self.laplacian[w,v]-=k
        q=self.electrical(np.full(self.n,self.nominal_temperature),self.nominal_current,0,self.nominal_acd)['heat_w']
        self.top_conductance=(q-self.area*flux)/(self.nominal_temperature-self.ambient)
        position=np.arange(self.n)
        self.top_conductance*=1+self.spatial_loss_variation*np.cos(2*np.pi*position/self.n)
        assert np.min(self.top_conductance)>0
        guess=np.r_[np.full(self.n,self.nominal_temperature),np.full(self.n,self.shell0),np.full(self.n,self.ledge0)]
        steady=root(lambda z:self.rates(z,self.nominal_current,0,self.nominal_acd)[0]/1e4,guess,
                    method='hybr',options={'xtol':1e-10})
        residual=float(np.max(abs(self.rates(steady.x,self.nominal_current,0,self.nominal_acd)[0])))
        if residual>=1e-5:
            steady=root(lambda z:self.rates(z,self.nominal_current,0,self.nominal_acd)[0]/1e4,steady.x,
                        method='hybr',options={'xtol':1e-12})
            residual=float(np.max(abs(self.rates(steady.x,self.nominal_current,0,self.nominal_acd)[0])))
        assert np.isfinite(steady.x).all() and residual<1e-5,(steady.message,residual)
        self.initial_equilibrium_residual_w=residual
        self._initial=steady.x

    def initial(self):
        return self._initial.copy()

    def mass(self,ledge):
        return self.bath_mass0-self.ledge_density*self.area*(ledge-self.ledge0)

    def liquidus(self,ledge):
        # First-order composition closure around 10.5 wt% excess AlF3.
        # Other dissolved species are maintained by an ideal mass-balanced feed.
        concentration=10.5*self.bath_mass0/self.mass(ledge)
        return self.nominal_liquidus-4.5*(concentration-10.5)

    def electrical(self,T,current,amplitude,acd):
        resistance=self.resistance_mohm*(.3+.7*acd/self.nominal_acd)
        local=self.n*resistance*np.exp(-self.temperature_resistivity*(T-self.nominal_temperature))
        g=1/local;fraction=g/g.sum();req=1/g.sum()
        second_moment=current**2+.5*amplitude**2
        electric_w=1000*(self.v_reversible*current+req*second_moment)
        external_w=1000*self.external_resistance_mohm*second_moment
        # Faraday: kg/s for current in kA. 6.60 kWh/kg includes feed heating;
        # it is deducted once and must not also be added as a separate sink.
        production_kg_s=current*1000*.0269815385*self.current_efficiency/(3*96485.33212)
        reaction_w=production_kg_s*6.60*3.6e6
        return {'power_w':electric_w,'heat_w':fraction*(electric_w-external_w-reaction_w),
                'external_w':external_w,'reaction_w':reaction_w,'production_kg_s':production_kg_s,
                'anode_fraction':fraction,'resistance_mohm':req}

    def energy(self,state):
        T,S,l=np.split(state,3)
        C=self.mass(l)*self.bath_cp+self.metal_mass/self.n*self.metal_cp
        return np.r_[C*(T-self.nominal_liquidus),self.shell_capacity_total/self.n*(S-self.ambient),
                     -self.ledge_density*self.area*self.fusion_heat*l]

    def rates(self,state,current,amplitude,acd,heat_exchange=1.,spatial_feed=None):
        T,S,l=np.split(state,3);liq=self.liquidus(l)
        electric=self.electrical(T,current,amplitude,acd)
        bath_side=self.area*self.bath_ledge_h*(T-liq)
        ledge_shell=self.area*(liq-S)/(l/self.ledge_conductivity+self.wall_resistance)
        ambient=self.area*self.shell_h*heat_exchange*(S-self.ambient)
        top=self.top_conductance*(T-self.ambient)
        disturbance=np.zeros(self.n) if spatial_feed is None else np.asarray(spatial_feed)
        rates=np.r_[electric['heat_w']-self.laplacian@T-bath_side-top-disturbance,
                    ledge_shell-ambient,bath_side-ledge_shell]
        net=float(np.sum(electric['heat_w']-top-ambient-disturbance))
        return rates,dict(electric,heat_loss_w=float(np.sum(top+ambient)),net_w=net)

    def step(self,previous,current,amplitude,acd,dt=180.,heat_exchange=1.,spatial_feed=None):
        e0=self.energy(previous)
        scale=np.r_[np.full(self.n,3e6),np.full(self.n,self.shell_capacity_total/self.n),
                    self.ledge_density*self.area*self.fusion_heat]
        def residual(state):
            rates,_=self.rates(state,current,amplitude,acd,heat_exchange,spatial_feed)
            return (self.energy(state)-e0-dt*rates)/scale
        sol=root(residual,previous,method='hybr',options={'xtol':1e-9})
        error=float(np.max(abs(residual(sol.x))))
        if not sol.success and error>1e-7:raise RuntimeError(('Thermal implicit solve',sol.message,error))
        _,aux=self.rates(sol.x,current,amplitude,acd,heat_exchange,spatial_feed)
        aux['energy_balance_residual_j']=float(np.sum(self.energy(sol.x)-e0)-dt*aux['net_w'])
        aux['scaled_equation_residual']=error
        return sol.x,aux

    def simulate(self,current,amplitude=None,acd=None,dt_schedule=1800.,dt=180.,heat_exchange=None,spatial_feed=None):
        current=np.asarray(current);T=len(current)
        amplitude=np.zeros(T) if amplitude is None else np.broadcast_to(amplitude,(T,))
        acd=np.full(T,self.nominal_acd) if acd is None else np.broadcast_to(acd,(T,))
        heat_exchange=np.ones(T) if heat_exchange is None else np.broadcast_to(heat_exchange,(T,))
        nstep=int(round(dt_schedule/dt));assert abs(nstep*dt-dt_schedule)<1e-8
        state=self.initial();states=[state];powers=[];auxs=[];idx=[]
        for t in range(T):
            for r in range(nstep):
                feed=None if spatial_feed is None else spatial_feed(t,r)
                state,aux=self.step(state,current[t],amplitude[t],acd[t],dt,heat_exchange[t],feed)
                states.append(state);powers.append(aux['power_w']);auxs.append(aux);idx.append(t)
        states=np.asarray(states);powers=np.asarray(powers)
        return {'states':states,'power_w':powers.reshape(T,nstep).mean(axis=1),
                'time_h':np.arange(len(states))*dt/3600,'schedule_index':np.asarray(idx),
                'minimum_temperature_c':float(states[:,:self.n].min()),'maximum_temperature_c':float(states[:,:self.n].max()),
                'minimum_ledge_m':float(states[:,2*self.n:].min()),'maximum_ledge_m':float(states[:,2*self.n:].max()),
                'terminal_temperature_error_c':float(np.max(abs(states[-1,:self.n]-states[0,:self.n]))),
                'terminal_ledge_error_m':float(np.max(abs(states[-1,2*self.n:]-states[0,2*self.n:]))),
                'maximum_spatial_temperature_range_c':float(np.max(np.ptp(states[:,:self.n],axis=1))),
                'maximum_energy_balance_residual_j':max(abs(a['energy_balance_residual_j']) for a in auxs),
                'maximum_scaled_equation_residual':max(a['scaled_equation_residual'] for a in auxs),
                'production_kg':float(np.sum(current)*dt_schedule*1000*.0269815385*self.current_efficiency/(3*96485.33212))}

def main():
    out=ROOT/'results/thermal';out.mkdir(parents=True,exist_ok=True)
    model=ThermalCell();hours=(np.arange(48)+.5)/2
    cases={'constant':np.full(48,180.),'daily_5pct':180*(1+.05*np.sin(2*np.pi*hours/24)),
           'daily_10pct':180*(1+.1*np.sin(2*np.pi*hours/24)),
           'two_hour_pulse':180+np.where((hours>=10)&(hours<12),18.,np.where((hours>=14)&(hours<16),-18.,0.))}
    reports={}
    for name,current in cases.items():
        start=time.perf_counter();res=model.simulate(current)
        np.savez_compressed(out/f'{name}.npz',**{k:v for k,v in res.items() if isinstance(v,np.ndarray)},current_ka=current)
        reports[name]={k:v for k,v in res.items() if not isinstance(v,np.ndarray)}
        reports[name]['runtime_s']=time.perf_counter()-start
        print(name,reports[name],flush=True)
    reports['model']={'nodes':model.n,'state_dimension':3*model.n,'nominal_ledge_m':model.ledge0,
                      'nominal_shell_c':model.shell0,'calibrated_top_conductance_w_k':float(model.top_conductance.sum()),
                      'calibration':'Synthetic steady energy closure, not a fit to company data.',
                      'source':'Wong et al. 2023 Table II for listed transport coefficients; other assumptions in script.'}
    (out/'validation.json').write_text(json.dumps(reports,indent=2),encoding='utf8')

if __name__=='__main__':main()
