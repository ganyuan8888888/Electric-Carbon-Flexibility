"""Aluminium-cell coupled Mathieu benchmark and exact power retraction.

Model: Horstmann, Kuhn and Dohnal (2025), Eqs. (6), (35) and (38).
The mechanical analogue is a reduced-order benchmark, not a plant-identified
high-fidelity MHD solver. Nondimensional time is explicitly retained.
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp

ROOT=__import__("repository_paths").ROOT
WX=1.5302; WY=0.3756

def critical_coupling(damping=0.005):
    mean=(WX**2+WY**2)/2
    diff=(WX**2-WY**2)/2
    return np.sqrt(diff**2+4*damping**2*mean)

def matrices(phase,coupling,amplitude,frequency,damping=0.005):
    coupling,amplitude,frequency=np.broadcast_arrays(coupling,amplitude,frequency)
    a=coupling*(1+amplitude*np.cos(phase))
    out=np.zeros(a.shape+(4,4))
    out[...,0,2]=1;out[...,1,3]=1
    out[...,2,0]=-WX**2;out[...,3,1]=-WY**2
    out[...,2,1]=-a;out[...,3,0]=a
    out[...,2,2]=-2*damping;out[...,3,3]=-2*damping
    return out/frequency[...,None,None]

def monodromy(coupling,amplitude,frequency,damping=0.005,steps=256):
    coupling,amplitude,frequency=np.broadcast_arrays(coupling,amplitude,frequency)
    phi=np.broadcast_to(np.eye(4),coupling.shape+(4,4)).copy()
    ds=2*np.pi/steps
    for k in range(steps):
        s=k*ds
        a=matrices(s,coupling,amplitude,frequency,damping)
        b=matrices(s+ds/2,coupling,amplitude,frequency,damping)
        c=matrices(s+ds,coupling,amplitude,frequency,damping)
        k1=a@phi;k2=b@(phi+ds*k1/2);k3=b@(phi+ds*k2/2);k4=c@(phi+ds*k3)
        phi+=ds*(k1+2*k2+2*k3+k4)/6
    return phi

def exponent(coupling,amplitude,frequency,damping=0.005,steps=256):
    phi=monodromy(coupling,amplitude,frequency,damping,steps)
    return np.log(np.max(np.abs(np.linalg.eigvals(phi)),axis=-1))*np.asarray(frequency)/(2*np.pi)

def independent_monodromy(coupling,amplitude,frequency,damping=0.005):
    def rhs(s,v):return (matrices(s,coupling,amplitude,frequency,damping)@v.reshape(4,4)).ravel()
    result=solve_ivp(rhs,(0,2*np.pi),np.eye(4).ravel(),method='DOP853',rtol=2e-12,atol=1e-13)
    assert result.success
    return result.y[:,-1].reshape(4,4)

def power(current,vrev,resistance,ncells=1):
    """Current in kA, resistance in mOhm, power in MW."""
    return ncells*(vrev*np.asarray(current)+resistance*np.asarray(current)**2)/1000

def current_for_power(p,vrev,resistance,ncells=1):
    """Positive root, evaluated without catastrophic cancellation."""
    v=np.asarray(p)*1000/ncells
    return 2*v/(vrev+np.sqrt(vrev*vrev+4*resistance*v))

def exact_pair_retraction(command,i1,vrev,resistance,ncells):
    p1=power(i1,vrev[0],resistance[0],ncells[0])
    i2=current_for_power(np.asarray(command)-p1,vrev[1],resistance[1],ncells[1])
    return np.stack([i1,i2],axis=-1)

def annulus_bounds(lower,upper):
    lower=np.asarray(lower);upper=np.asarray(upper)
    return max(0.0,float(np.max(lower-(upper.sum()-upper)))),float(upper.sum())

if __name__=='__main__':
    out=ROOT/'results/process';out.mkdir(exist_ok=True,parents=True)
    freq=np.linspace(1.55,2.65,121);amp=np.linspace(0,.45,91)
    ff,aa=np.meshgrid(freq,amp)
    summaries=[]
    for over in [1.001,1.005,1.01,1.02,1.04]:
        coupling=critical_coupling()*over
        ex=exponent(coupling,aa,ff,steps=256)
        np.savez_compressed(out/f'stability_{over:.3f}.npz',frequency=freq,amplitude=amp,exponent=ex,coupling=coupling,damping=.005)
        idx=np.argwhere(ex<=-.001)
        minamp=float(amp[idx[:,0].min()]) if len(idx) else None
        loc=np.unravel_index(np.argmin(ex),ex.shape)
        summaries.append(dict(overcriticality=over,coupling=coupling,minimum_sampled_amplitude=minamp,best_frequency=float(ff[loc]),best_amplitude=float(aa[loc]),best_exponent=float(ex[loc]),feasible_grid_points=len(idx)))
        print(json.dumps(summaries[-1]),flush=True)
    (out/'scan_summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf8')
