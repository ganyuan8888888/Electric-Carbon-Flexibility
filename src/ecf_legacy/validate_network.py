from pathlib import Path
import json
import numpy as np
from network_model import make_network,profiles
from carbon_certificate import certificate

ROOT=__import__("repository_paths").ROOT
def main():
    rng=np.random.default_rng(20260909);records=[]
    for name in ['case39','case118']:
        net=make_network(name);prof=profiles(net);r=np.load(ROOT/f'results/network/{name}_baseline.npz')
        ng=len(net.gen);nr=len(net.ren_buses);nk=len(net.industrial_buses)
        for t in [3,12,24,35,45]:
            pg,pr,pl=r['pg'][:,t],r['pr'][:,t],r['pl'][:,t]
            background=prof['background_mw'][:,t]
            ref=net.carbon(pg,pr,pl,background,derivatives=True)
            for magnitude in [.001,.01,.1,1.,5.]:
                delta=rng.normal(0,1,ng+nr+nk)
                delta[ng:ng+nr]=0  # Preserve zero/nighttime renewable injections.
                # Conservation: sum dPg + sum dPr = sum dPl.
                delta[0]+=delta[-nk:].sum()-delta[:ng+nr].sum()
                delta*=magnitude/max(abs(delta))
                actual=net.carbon(pg+delta[:ng],pr+delta[ng:ng+nr],pl+delta[-nk:],background)
                cert=certificate(ref,delta,actual)
                record={k:v for k,v in cert.items() if not isinstance(v,np.ndarray)}
                record.update(network=name,t=t,perturbation_mw=magnitude,
                    carbon_conservation_error=actual['attribution_conservation_residual'])
                if cert['valid']:assert cert['actual_remainder']<=cert['remainder_bound']+1e-10,record
                records.append(record)
    valid=[r for r in records if r['valid']]
    summary={'samples':len(records),'certified_samples':len(valid),
        'max_valid_remainder_ratio':max(r['actual_remainder']/max(r['remainder_bound'],1e-16) for r in valid),
        'max_carbon_conservation_error':max(abs(r['carbon_conservation_error']) for r in records),
        'scope':'Numerical validation of an analytic remainder bound; not a global dispatch optimality certificate.'}
    (ROOT/'results/network/certificate_validation.json').write_text(json.dumps({'summary':summary,'records':records},indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
