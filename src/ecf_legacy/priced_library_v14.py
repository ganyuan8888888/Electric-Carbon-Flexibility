"""Admit priced columns only after independent continuous execution checks."""
from pathlib import Path
import json,numpy as np
from expanded_library_v13 import expanded_library
ROOT=__import__("repository_paths").ROOT

def priced_library(net,include_v15=False,include_v16=False):
    lib=expanded_library(net);K,B,T=lib['P'].shape;extensions=[]
    for date in ['2020-04-07','2020-10-17','2020-11-29']:
        pp=lib['P'][:,0].copy();ok=np.zeros(K,bool)
        for k in range(K):
            stem=ROOT/f'results/full_cycle_v13/{date}_priced_cycle_spectral_site{k}_execution'
            if not stem.with_suffix('.json').exists():continue
            row=json.loads(stem.with_suffix('.json').read_text())
            if row['certified']:
                pp[k]=np.load(stem.with_suffix('.npz'))['power_w'].mean(axis=0)*int(lib['cells_per_series'][k]*3)/1e6;ok[k]=True
        if ok.any():
            lib['P']=np.concatenate([lib['P'],pp[:,None,:]],axis=1)
            lib['admissible']=np.concatenate([lib['admissible'],ok[:,None]],axis=1)
            extensions.append(dict(date=date,sites=np.flatnonzero(ok).tolist()))
    lib['metadata']={**lib['metadata'],'version':'priced_certified_columns_v14','priced_extensions':extensions}
    if include_v15:
        for stem in sorted((ROOT/'results/full_cycle_v13').glob('*_retracted_midpoint_step*_execution.json')):
            row=json.loads(stem.read_text())
            if not row.get('certified',False):continue
            k=int(row['site']);pp=lib['P'][:,0].copy();ok=np.zeros(K,bool)
            pp[k]=np.load(stem.with_suffix('.npz'))['power_w'].mean(axis=0)*int(lib['cells_per_series'][k]*3)/1e6;ok[k]=True
            lib['P']=np.concatenate([lib['P'],pp[:,None,:]],axis=1)
            lib['admissible']=np.concatenate([lib['admissible'],ok[:,None]],axis=1)
            extensions.append(dict(date=row['date'],sites=[k],label=row['label']))
        lib['metadata']={**lib['metadata'],'version':'midpoint_retracted_certified_v15'}
    if include_v16:
        for stem in sorted((ROOT/'results/full_cycle_v13').glob('*_balanced_v16*_execution.json')):
            row=json.loads(stem.read_text())
            if not row.get('certified',False):continue
            k=int(row['site']);pp=lib['P'][:,0].copy();ok=np.zeros(K,bool)
            pp[k]=np.load(stem.with_suffix('.npz'))['power_w'].mean(axis=0)*int(lib['cells_per_series'][k]*3)/1e6;ok[k]=True
            lib['P']=np.concatenate([lib['P'],pp[:,None,:]],axis=1)
            lib['admissible']=np.concatenate([lib['admissible'],ok[:,None]],axis=1)
            extensions.append(dict(date=row['date'],sites=[k],label=row['label']))
        lib['metadata']={**lib['metadata'],'version':'balanced_priced_certified_v16'}
    return lib
