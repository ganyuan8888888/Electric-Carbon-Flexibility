"""Independent actual-carrier integration of every admitted trajectory.

Numba changes evaluation speed only. Its instantaneous RHS is checked against
ThermalCell before integration, and candidate 0/12 is compared with the archived
uncompiled DOP853 result. No fastmath or changed integrator tolerance is used.
"""
from pathlib import Path
import sys,json,time
ROOT=__import__("repository_paths").ROOT
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from thermal_process import ThermalCell
from execution_validation import HZ,PHASES,waveform

@njit(cache=True)
def rhs_fast(sec,y,old,new,ao,an,p,area,top,lap):
    n=len(area);dim=3*n+2;out=np.empty_like(y)
    theta=2*np.pi*HZ*sec
    envelope=(1-np.cos(theta/2))/2 if sec<1/HZ else 1.
    mu=old+(new-old)*envelope;am=ao+(an-ao)*envelope
    for j in range(3):
        z=y[j*dim:(j+1)*dim];T=z[:n];S=z[n:2*n];ell=z[2*n:3*n]
        I=mu+am*np.cos(theta+2*np.pi*j/3)
        conductance=np.exp(p[0]*(T-p[1]))/(n*p[2])
        fraction=conductance/conductance.sum();req=1/conductance.sum()
        power=1000*(p[3]*I+req*I*I)
        production=I*1000*.0269815385*p[4]/(3*96485.33212)
        heat=fraction*(power-1000*p[5]*I*I-production*6.60*3.6e6)
        mass=p[6]-p[7]*area*(ell-p[8])
        liq=p[9]-4.5*(10.5*p[6]/mass-10.5)
        bath=area*p[10]*(T-liq)
        shell=area*(liq-S)/(ell/p[11]+p[12])
        dl=-(bath-shell)/(p[7]*area*p[13])
        dT=(heat-lap@T-bath-top*(T-p[14])+p[7]*area*p[15]*(T-p[9])*dl)/(mass*p[15]+p[16])
        dS=(shell-area*p[17]*(S-p[14]))/p[18]
        out[j*dim:j*dim+n]=dT;out[j*dim+n:j*dim+2*n]=dS
        out[j*dim+2*n:j*dim+3*n]=dl
        out[j*dim+3*n]=power;out[j*dim+3*n+1]=production
    return out

def params(m):
    return np.array([m.temperature_resistivity,m.nominal_temperature,
        m.resistance_mohm*(.3+.7*.043/m.nominal_acd),m.v_reversible,m.current_efficiency,
        m.external_resistance_mohm,m.bath_mass0,m.ledge_density,m.ledge0,
        m.nominal_liquidus,m.bath_ledge_h,m.ledge_conductivity,m.wall_resistance,
        m.fusion_heat,m.ambient,m.bath_cp,m.metal_mass/m.n*m.metal_cp,m.shell_h,
        m.shell_capacity_total/m.n])

def check_rhs(m,p):
    rng=np.random.default_rng(12009);n=m.n;maxerr=0.
    for seconds in [0.,3.5,22.1,27.8]:
        z=np.tile(np.r_[m.initial(),0.,0.],(3,1));z[:,:n]+=rng.normal(0,.1,(3,n))
        fast=rhs_fast(seconds,z.ravel(),178.,182.,6.,9.,p,m.area,m.top_conductance,m.laplacian).reshape(z.shape)
        for j,phase in enumerate(PHASES):
            theta=2*np.pi*HZ*seconds
            I=waveform(theta,178.,182.,6.,9.,phase) if seconds<1/HZ else 182.+9*np.cos(theta+phase)
            rates,aux=m.rates(z[j,:3*n],I,0.,.043)
            dl=-rates[2*n:]/(m.ledge_density*m.area*m.fusion_heat)
            cross=-m.ledge_density*m.area*m.bath_cp*(z[j,:n]-m.nominal_liquidus)
            dt=(rates[:n]-cross*dl)/(m.mass(z[j,2*n:3*n])*m.bath_cp+m.metal_mass/n*m.metal_cp)
            expected=np.r_[dt,rates[n:2*n]/(m.shell_capacity_total/n),dl,aux['power_w'],aux['production_kg_s']]
            err=np.max(abs(expected-fast[j])/np.maximum(1,abs(expected)))
            maxerr=max(maxerr,float(err))
    assert maxerr<2e-12,maxerr
    return maxerr

def integrate(site,candidate):
    loc=ROOT/'results/library/public_v3';out=ROOT/'results/execution/public_v12';out.mkdir(parents=True,exist_ok=True)
    dst=out/f'site{site}_candidate{candidate}'
    if dst.with_suffix('.json').exists():return json.loads(dst.with_suffix('.json').read_text())
    meta=json.loads((loc/f'site_{site}.json').read_text());raw=np.load(loc/f'site_{site}.npz')
    m=ThermalCell(**meta['parameters']);p=params(m);err=check_rhs(m,p);start=time.perf_counter()
    n=m.n;dim=3*n+2;y=np.tile(np.r_[m.initial(),0.,0.],3)
    states=[np.tile(m.initial(),(3,1))];powers=[];production=[];nfev=0
    minT=1e6;maxT=-1e6;minL=1e6;maxL=-1e6
    for t in range(48):
        current=raw['current_ka'][candidate];amp=raw['amplitude_ka'][candidate]
        old=180. if t==0 else current[t-1];ao=0. if t==0 else amp[t-1]
        args=(old,current[t],ao,amp[t],p,m.area,m.top_conductance,m.laplacian)
        before=y.reshape(3,dim).copy()
        sol=solve_ivp(rhs_fast,[0,1800],y,args=args,method='DOP853',rtol=2e-9,atol=1e-10,max_step=1/HZ/8,dense_output=True)
        assert sol.success,sol.message
        y=sol.y[:,-1];after=y.reshape(3,dim);nfev+=sol.nfev
        powers.append((after[:,-2]-before[:,-2])/1800);production.append(after[:,-1]-before[:,-1])
        dense=sol.y.reshape(3,dim,-1)
        minT=min(minT,float(dense[:,:n].min()));maxT=max(maxT,float(dense[:,:n].max()))
        minL=min(minL,float(dense[:,2*n:3*n].min()));maxL=max(maxL,float(dense[:,2*n:3*n].max()))
        for delta in range(300,1801,300):states.append(sol.sol(delta).reshape(3,dim)[:,:3*n])
    states=np.asarray(states).transpose(1,0,2);powers=np.asarray(powers).T;production=np.asarray(production).T
    coarse=raw['states'][candidate]
    report=dict(site=site,candidate=candidate,rhs_relative_error=err,runtime_s=time.perf_counter()-start,function_evaluations=nfev,
        minimum_temperature_c=minT,maximum_temperature_c=maxT,minimum_ledge_m=minL,maximum_ledge_m=maxL,
        terminal_temperature_error_c=float(abs(states[:,-1,:n]-states[:,0,:n]).max()),
        terminal_ledge_error_m=float(abs(states[:,-1,2*n:]-states[:,0,2*n:]).max()),
        maximum_temperature_difference_c=float(abs(states[:,:,:n]-coarse[:,:,:n]).max()),
        maximum_ledge_difference_m=float(abs(states[:,:,2*n:]-coarse[:,:,2*n:]).max()),
        maximum_cell_mean_power_difference_w=float(abs(powers.mean(axis=0)-raw['power_w'][candidate]).max()),production_kg=production.sum(axis=1).tolist())
    report['thermal_passed']=bool(minT>=940 and maxT<=980 and minL>=.04 and maxL<=.25 and report['terminal_temperature_error_c']<=2 and report['terminal_ledge_error_m']<=.015)
    prior=ROOT/f'results/execution/public_v3/site{site}_candidate{candidate}.npz'
    if prior.exists():
        old=np.load(prior)
        report['archived_uncompiled_state_difference']=float(abs(old['states']-states).max())
        report['archived_uncompiled_power_difference_w']=float(abs(old['power_w']-powers).max())
        assert report['archived_uncompiled_state_difference']<1e-5
    np.savez_compressed(dst.with_suffix('.npz'),states=states,power_w=powers,time_h=np.arange(289)/12,production_kg=production)
    dst.with_suffix('.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True);return report

if __name__=='__main__':
    admitted=np.load(ROOT/'results/library/public_v3/candidate_library.npz')['certified']
    pairs=[(0,12)]+[(int(k),int(b)) for k,b in zip(*np.where(admitted)) if (k,b)!=(0,12)]
    if len(sys.argv)>1:pairs=[tuple(map(int,sys.argv[1:3]))]
    records=[integrate(k,b) for k,b in pairs]
    (ROOT/'results/execution/public_v12/summary.json').write_text(json.dumps(records,indent=2))
