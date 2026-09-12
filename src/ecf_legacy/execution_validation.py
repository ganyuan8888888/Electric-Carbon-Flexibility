"""Smooth current execution and finite-horizon modal reachability validation.

The 0.045-Hz mapping of normalized frequency 2.05 is a declared study scaling.
Independent phase-shifted series share a common smooth envelope. Their aggregate
fundamental and second harmonic cancel for the frozen quadratic power mapping.
Disturbances are specified by an L2 energy bound at carrier-cycle boundaries;
this is neither a probabilistic noise model nor arbitrary pointwise disturbances.
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp
from harmonic_process import matrix
from process_model import WX,WY

ROOT=__import__("repository_paths").ROOT
OUT=ROOT/'results/execution';OUT.mkdir(exist_ok=True,parents=True)
NU=2.05;HZ=.045;CYCLES=81;PHASES=2*np.pi*np.arange(3)/3
S=np.diag([1.,1.,1/WX,1/WY]);SI=np.linalg.inv(S)

def moment_average(current,amplitude):
    """Exact first/second moments of a raised-cosine transition over one cycle.

    The three-series phase average eliminates carrier cross-terms pointwise.
    The returned equivalent amplitude is a thermal RMS parameter, not a new
    commanded carrier waveform. First moments telescope to the original output.
    """
    current=np.asarray(current);amplitude=np.asarray(amplitude)
    previous=np.r_[180.,current[:-1]];aprevious=np.r_[0.,amplitude[:-1]]
    d=current-previous;da=amplitude-aprevious;q=1/CYCLES
    mean=current-.5*q*d
    second=current**2+q*(-previous*d-.625*d*d)
    asecond=amplitude**2+q*(-aprevious*da-.625*da*da)
    effective=np.sqrt(np.maximum(0,2*(second-mean**2)+asecond))
    return mean,effective

def waveform(theta,old_mu,new_mu,old_amp,new_amp,phase):
    envelope=.5*(1-np.cos(np.asarray(theta)/2))
    mu=old_mu+(new_mu-old_mu)*envelope
    amp=old_amp+(new_amp-old_amp)*envelope
    return mu+amp*np.cos(np.asarray(theta)+phase)


def phase_moments(current,amplitude,phase):
    """Per-series first and second moments, including envelope/carrier products."""
    current=np.asarray(current);amplitude=np.asarray(amplitude)
    old=np.r_[180.,current[:-1]];ao=np.r_[0.,amplitude[:-1]]
    dm=current-old;da=amplitude-ao;q=1/CYCLES
    k1=2*np.sin(phase)/(3*np.pi);k2=4*np.sin(2*phase)/(15*np.pi)
    mu=current-.5*q*dm
    mu2=current**2-q*(old*dm+.625*dm**2)
    a2=amplitude**2-q*(ao*da+.625*da**2)
    first=mu+q*da*k1
    second=mu2+.5*a2+q*(2*(old*da+ao*dm)*k1+2*dm*da*(k1+np.cos(phase)/16)+.5*(2*ao*da+da**2)*k2)
    rms=np.sqrt(np.maximum(0,2*(second-first**2)))
    return first,rms

def transfer(old_mu,new_mu,old_amp,new_amp,phase,steps=256,independent=False):
    def a(theta):return matrix(waveform(theta,old_mu,new_mu,old_amp,new_amp,phase),.043)/NU
    if independent:
        r=solve_ivp(lambda s,v:(a(s)@v.reshape(4,4)).ravel(),[0,2*np.pi],np.eye(4).ravel(),
                    method='DOP853',rtol=2e-12,atol=1e-13)
        assert r.success;return r.y[:,-1].reshape(4,4)
    h=2*np.pi/steps;M=np.eye(4);maxnorm=1.
    for j in range(steps):
        s=j*h;a0=a(s);a1=a(s+h/2);a2=a(s+h)
        k1=a0;k2=a1@(np.eye(4)+h*k1/2);k3=a1@(np.eye(4)+h*k2/2);k4=a2@(np.eye(4)+h*k3)
        F=np.eye(4)+h*(k1+2*k2+2*k3+k4)/6
        M=F@M;maxnorm=max(maxnorm,float(np.linalg.norm(S@M@SI,2)))
    return S@M@SI,maxnorm

def main():
    raw=np.load(ROOT/'results/library/candidate_library_step_v1.npz')
    current=raw['current_ka'];amp=raw['amplitude_ka'];choices={}
    transitions=set()
    for row,arow in zip(current,amp):
        for old,new,ao,an in zip(np.r_[180.,row[:-1]],row,np.r_[0.,arow[:-1]],arow):
            transitions.add((old,new,ao,an));transitions.add((new,new,an,an))
    numerical_errors=[];maxrate=0.
    for j,key in enumerate(sorted(transitions)):
        for phase in PHASES:
            M,gain=transfer(*key,phase);choices[(*key,phase)]=(M,gain)
            check=S@transfer(*key,phase,independent=True)@SI
            numerical_errors.append(float(np.linalg.norm(M-check,2)))
            theta=np.linspace(0,2*np.pi,4097);I=waveform(theta,*key,phase)
            maxrate=max(maxrate,float(np.max(abs(np.diff(I)))/(1/HZ/4096)))
    sigma=1e-3;epsilon=1e-5;limit=.05
    records=[]
    for j,(row,arow) in enumerate(zip(current,amp)):
        bounds=[];terminal=[];radii=[]
        for phase in PHASES:
            Phi=np.eye(4);W=np.zeros((4,4));worst=0.
            for old,new,ao,an in zip(np.r_[180.,row[:-1]],row,np.r_[0.,arow[:-1]],arow):
                for cycle in range(CYCLES):
                    key=(old,new,ao,an,phase) if cycle==0 else (new,new,an,an,phase)
                    M,gain=choices[key]
                    Wstart=W+epsilon**2*np.eye(4)
                    radius=sigma*np.linalg.norm(Phi,2)+np.sqrt(max(0,np.linalg.eigvalsh(Wstart)[-1]))
                    worst=max(worst,gain*radius)
                    Phi=M@Phi;W=M@Wstart@M.T;W=(W+W.T)/2
            bounds.append(float(worst));terminal.append(float(sigma*np.linalg.norm(Phi,2)+np.sqrt(np.linalg.eigvalsh(W)[-1])))
            radii.append(float(np.max(abs(np.linalg.eigvals(Phi)))))
        records.append({'candidate_index':j,'reachability_upper_bound':max(bounds),'series_bounds':bounds,
                        'terminal_reachability_bound':max(terminal),'daily_monodromy_radius':max(radii),
                        'bounded_state_feasible':bool(max(bounds)<=limit)})
        if j%20==0:print('execution',j,records[-1],flush=True)
    report={'frequency_hz':HZ,'normalized_frequency':NU,'cycles_per_30min':CYCLES,'smooth_transition_cycles':1,
            'maximum_current_rate_ka_s':maxrate,'initial_normalized_radius':sigma,'disturbance_l2_energy_radius':epsilon,
            'allowed_normalized_state_radius':limit,'maximum_discrete_vs_DOP853_matrix_error':max(numerical_errors),
            'unique_transition_phase_checks':len(numerical_errors),'records':records,
            'scope':'Analytic norm propagation for a sampled reduced modal model with an L2 disturbance budget. Floating-point integration discrepancy is reported; a validated interval bound for the original continuous plant is not claimed.'}
    (OUT/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('feasible',sum(x['bounded_state_feasible'] for x in records),'of',len(records),'max rate',maxrate,flush=True)

if __name__=='__main__':main()
