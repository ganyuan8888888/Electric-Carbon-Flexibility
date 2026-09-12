"""Common four-method library, extended only by independently certified controls."""
from pathlib import Path
import json,numpy as np
from run_revision_v12 import continuous_library
ROOT=__import__("repository_paths").ROOT

def expanded_library(net):
    lib=continuous_library(net);K,B,T=lib['P'].shape
    records=[];powers=[];admission=[]
    for date in ['2020-04-07','2020-10-17','2020-11-29']:
        pp=lib['P'][:,0].copy();ok=np.zeros(K,dtype=bool);entries=[]
        for k in range(K):
            stem=ROOT/f'results/full_cycle_v13/{date}_cycle_spectral_site{k}_execution'
            if stem.with_suffix('.json').exists():
                report=json.loads(stem.with_suffix('.json').read_text())
                if report['certified']:
                    arrays=np.load(stem.with_suffix('.npz'));pp[k]=arrays['power_w'].mean(axis=0)*int(lib['cells_per_series'][k]*3)/1e6
                    ok[k]=True;entries.append(dict(site=k,file=str(stem.relative_to(ROOT))))
        if ok.any():powers.append(pp);admission.append(ok);records.append(dict(date=date,entries=entries))
    if powers:
        lib['P']=np.concatenate([lib['P'],np.stack(powers,axis=1)],axis=1)
        lib['admissible']=np.concatenate([lib['admissible'],np.stack(admission,axis=1)],axis=1)
    lib['ids']=np.arange(lib['P'].shape[1]);lib['metadata']={**lib['metadata'],'version':'continuous_checked_extended_v13','extensions':records}
    return lib

def extend_weights(r,lib):
    r=r.copy();w=np.zeros(lib['P'].shape[:2]);w[:,:r['weights'].shape[1]]=r['weights'];r['weights']=w
    return r
