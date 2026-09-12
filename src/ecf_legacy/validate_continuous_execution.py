"""Independent continuous thermal integration of a selected library candidate.

The actual carrier and its one-period transition are applied to the nonlinear
thermal equations. DOP853 is independent of the library's moment-averaged
backward-Euler scheme. This is a discretization/model-reduction check; the
dispatch feasibility statement remains explicitly scoped to its discrete model.
"""
from pathlib import Path
import json,sys,time
import numpy as np
from scipy.integrate import solve_ivp
from thermal_process import ThermalCell
from execution_validation import PHASES,HZ,waveform

ROOT=__import__("repository_paths").ROOT
SITE=int(sys.argv[1]) if len(sys.argv)>1 else 0
CANDIDATE=int(sys.argv[2]) if len(sys.argv)>2 else 12
location=ROOT/'results/library/public_v3'
meta=json.loads((location/f'site_{SITE}.json').read_text(encoding='utf8'))
raw=np.load(location/f'site_{SITE}.npz');model=ThermalCell(**meta['parameters'])
current=raw['current_ka'][CANDIDATE];amplitude=raw['amplitude_ka'][CANDIDATE]
out=ROOT/'results/execution/public_v3';out.mkdir(parents=True,exist_ok=True)
start=time.perf_counter();n=model.n;dimension=3*n+2
initial=np.r_[model.initial(),0.,0.];y=np.tile(initial,3)
states=[np.tile(model.initial(),(3,1))];powers=[];production=[];times=[0.];nfev=0
period=1/HZ
for t in range(48):
    old=180. if t==0 else current[t-1];ao=0. if t==0 else amplitude[t-1]
    def rhs(seconds,flat):
        rows=flat.reshape(3,dimension);dy=[]
        for j,phase in enumerate(PHASES):
            theta=2*np.pi*HZ*seconds
            I=waveform(theta,old,current[t],ao,amplitude[t],phase) if seconds<period else current[t]+amplitude[t]*np.cos(theta+phase)
            T,sh,ledge=np.split(rows[j,:3*n],3)
            rates,aux=model.rates(rows[j,:3*n],I,0.,.043)
            cap=model.mass(ledge)*model.bath_cp+model.metal_mass/n*model.metal_cp
            dledge=-rates[2*n:]/(model.ledge_density*model.area*model.fusion_heat)
            cross=-model.ledge_density*model.area*model.bath_cp*(T-model.nominal_liquidus)
            dT=(rates[:n]-cross*dledge)/cap
            dsh=rates[n:2*n]/(model.shell_capacity_total/n)
            dy.append(np.r_[dT,dsh,dledge,aux['power_w'],aux['production_kg_s']])
        return np.asarray(dy).ravel()
    before=y.reshape(3,dimension).copy()
    sol=solve_ivp(rhs,[0,1800],y,method='DOP853',rtol=2e-9,atol=1e-10,
                  max_step=period/8,dense_output=True)
    if not sol.success:raise RuntimeError(sol.message)
    y=sol.y[:,-1];nfev+=sol.nfev;after=y.reshape(3,dimension)
    powers.append((after[:,-2]-before[:,-2])/1800)
    production.append(after[:,-1]-before[:,-1])
    for delta in range(300,1801,300):
        states.append(sol.sol(delta).reshape(3,dimension)[:,:3*n]);times.append((t*1800+delta)/3600)
    if t%8==7:print('actual-carrier validation',SITE,CANDIDATE,(t+1)/2,'hours',round(time.perf_counter()-start,1),'s',flush=True)
states=np.asarray(states).transpose(1,0,2);powers=np.asarray(powers).T;production=np.asarray(production).T
coarse=raw['states'][CANDIDATE]
report=dict(site=SITE,candidate=CANDIDATE,method='DOP853 with actual 0.045-Hz carrier and nonlinear instantaneous electrothermal feedback',
            maximum_temperature_difference_c=float(abs(states[:,:,:n]-coarse[:,:,:n]).max()),
            maximum_ledge_difference_m=float(abs(states[:,:,2*n:]-coarse[:,:,2*n:]).max()),
            maximum_cell_mean_power_difference_w=float(abs(powers.mean(axis=0)-raw['power_w'][CANDIDATE]).max()),
            production_kg=production.sum(axis=1).tolist(),function_evaluations=nfev,runtime_s=time.perf_counter()-start,
            maximum_temperature_c=float(states[:,:,:n].max()),minimum_temperature_c=float(states[:,:,:n].min()),
            terminal_temperature_error_c=float(abs(states[:,-1,:n]-states[:,0,:n]).max()),
            terminal_ledge_error_m=float(abs(states[:,-1,2*n:]-states[:,0,2*n:]).max()),
            scope='Continuous reduced thermal model validation; does not replace the disclosed discrete network optimization model or certify full industrial MHD dynamics.')
np.savez_compressed(out/f'site{SITE}_candidate{CANDIDATE}.npz',states=states,power_w=powers,time_h=times,production_kg=production)
(out/f'site{SITE}_candidate{CANDIDATE}.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report),flush=True)
