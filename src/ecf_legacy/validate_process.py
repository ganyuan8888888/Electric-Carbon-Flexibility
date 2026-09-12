"""Independent numerical checks; failures stop, no result filtering."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import brentq
from process_model import (critical_coupling,monodromy,independent_monodromy,
    exponent,power,exact_pair_retraction)

ROOT=__import__("repository_paths").ROOT
def main():
    rng=np.random.default_rng(20260908)
    records=[]
    for over in [1.001,1.005,1.01,1.02,1.04]:
        data=np.load(ROOT/f'results/process/stability_{over:.3f}.npz')
        aa,ff=np.meshgrid(data['amplitude'],data['frequency'],indexing='ij')
        flat=np.argsort(np.abs(data['exponent'].ravel()+.001))[:6]
        flat=np.r_[flat,rng.choice(aa.size,10,replace=False)]
        for loc in flat:
            amp=float(aa.ravel()[loc]);freq=float(ff.ravel()[loc]);a=critical_coupling()*over
            phi=independent_monodromy(a,amp,freq)
            ref=np.log(max(abs(np.linalg.eigvals(phi))))*freq/(2*np.pi)
            approx=float(exponent(a,amp,freq,steps=256))
            records.append(dict(over=over,amplitude=amp,frequency=freq,reference_exponent=ref,
                rk4_exponent=approx,error=abs(approx-ref),matrix_error=float(np.linalg.norm(monodromy(a,amp,freq)-phi,np.inf))))
    maxerr=max(r['error'] for r in records)
    assert maxerr<2e-6,(maxerr,'Floquet discretization error exceeds threshold')
    t=np.linspace(0,240,12001)
    v=np.array([1.75,1.78]);res=np.array([.0135,.0133]);nc=np.array([180,180])
    command=power(180,v[0],res[0],nc[0])+power(180,v[1],res[1],nc[1])+3*np.sin(2*np.pi*t/90)
    i1=180+18*np.cos(2*np.pi*t/22)
    currents=exact_pair_retraction(command,i1,v,res,nc)
    residual=power(currents[:,0],v[0],res[0],nc[0])+power(currents[:,1],v[1],res[1],nc[1])-command
    # A separate static counterpart quantifies current cancellation error.
    naive=power(i1,v[0],res[0],nc[0])+power(360-i1,v[1],res[1],nc[1])
    steady=power(180,v[0],res[0],nc[0])+power(180,v[1],res[1],nc[1])
    assert max(abs(residual))<1e-10
    summary={'seed':20260908,'n_independent_floquet_checks':len(records),'max_exponent_error':maxerr,
        'power_tracking_max_abs_mw':float(max(abs(residual))),
        'naive_equal_current_peak_power_error_mw':float(max(abs(naive-steady))),
        'compensating_current_min_ka':float(currents[:,1].min()),'compensating_current_max_ka':float(currents[:,1].max()),
        'scope':'Synthetic potline parameters; power equality verified; this test does not certify both potlines stability.'}
    out=ROOT/'results/process'
    (out/'validation.json').write_text(json.dumps({'summary':summary,'checks':records},indent=2),encoding='utf8')
    np.savez_compressed(out/'power_retraction.npz',time_s=t,command_mw=command,current_ka=currents,residual_mw=residual,naive_mw=naive)
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
