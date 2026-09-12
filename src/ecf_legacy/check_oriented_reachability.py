"""Retain transition orientation when bounding modal reachability.

The previous product of two operator norms can fail a sufficient check without
proving physical infeasibility. This audit preserves both covariance directions.
It does not change the disturbance budget or the state limit.
"""
from pathlib import Path
import json
import numpy as np
from execution_validation import NU,HZ,CYCLES,PHASES,S,SI,waveform
from harmonic_process import matrix

ROOT=__import__("repository_paths").ROOT

def transitions(key,phase,steps=512,stride=8):
    def A(s):return S@matrix(waveform(s,*key,phase),.043)@SI/NU
    h=2*np.pi/steps;M=np.eye(4);samples=[M.copy()]
    for j in range(steps):
        s=j*h;a0=A(s);a1=A(s+h/2);a2=A(s+h)
        k1=a0@M;k2=a1@(M+h*k1/2);k3=a1@(M+h*k2/2);k4=a2@(M+h*k3)
        M=M+h*(k1+2*k2+2*k3+k4)/6
        if (j+1)%stride==0:samples.append(M.copy())
    lo=min(key[:2])-max(key[2:]);hi=max(key[:2])+max(key[2:])
    L=max(np.linalg.norm(S@matrix(v,.043)@SI/NU,2) for v in [lo,hi])
    return M,np.array(samples),float(np.exp(L*h*stride))

def evaluate(row,amp,cache,sigma=.001,eps=1e-5):
    bounds=[];sample_bounds=[]
    for phase in PHASES:
        Phi=np.eye(4);W=np.zeros((4,4));worst=0.;ws=0.
        for old,new,ao,an in zip(np.r_[180.,row[:-1]],row,np.r_[0.,amp[:-1]],amp):
            for cycle in range(CYCLES):
                key=(old,new,ao,an) if cycle==0 else (new,new,an,an)
                ck=(*key,phase)
                if ck not in cache:cache[ck]=transitions(key,phase)
                M,B,guard=cache[ck];W0=W+eps**2*np.eye(4)
                BP=B@Phi
                X=BP@np.swapaxes(BP,-1,-2)
                Y=B@W0@np.swapaxes(B,-1,-2)
                values=sigma*np.sqrt(np.maximum(0,np.linalg.eigvalsh(X)[...,-1]))+np.sqrt(np.maximum(0,np.linalg.eigvalsh(Y)[...,-1]))
                ws=max(ws,float(values.max()));worst=max(worst,float(values.max())*guard)
                Phi=M@Phi;W=M@W0@M.T;W=(W+W.T)/2
        bounds.append(worst);sample_bounds.append(ws)
    return dict(series_bounds=bounds,reachability_upper_bound=max(bounds),sampled_bound=max(sample_bounds),
                sufficient_check_passed=bool(max(bounds)<=.05))

if __name__=='__main__':
    raw=np.load(ROOT/'results/library/candidate_library_step_v1.npz');cache={};records=[]
    import sys
    indices=list(map(int,sys.argv[1:])) or [0,66,87]
    for j in indices:
        r=dict(candidate_index=j,**evaluate(raw['current_ka'][j],raw['amplitude_ka'][j],cache));records.append(r);print(r,flush=True)
    out=ROOT/'results/execution/oriented_validation.json'
    out.write_text(json.dumps(dict(records=records,state_limit=.05,initial_radius=.001,disturbance_radius=1e-5,
        note='Orientation-preserving sufficient bound, with intersample exponential norm guard. Applies to the reduced modal system; numerical integration error is not a validated interval enclosure.'),indent=2),encoding='utf8')
