"""Only independently certified direct-case columns may extend M4."""
from pathlib import Path
import json,numpy as np
from priced_library_v14 import priced_library
ROOT=__import__("repository_paths").ROOT

def library(net):
    lib=priced_library(net,include_v15=True,include_v16=True);K=lib['P'].shape[0];records=[]
    for source in sorted((ROOT/'results/full_cycle_v13').glob('*_direct_v20_step*_execution.json')):
        report=json.loads(source.read_text())
        if not report.get('certified',False):continue
        k=report['site'];power=lib['P'][:,0].copy();mask=np.zeros(K,bool);mask[k]=True
        power[k]=np.load(source.with_suffix('.npz'))['power_w'].mean(axis=0)*int(lib['cells_per_series'][k]*3)/1e6
        lib['P']=np.concatenate([lib['P'],power[:,None,:]],axis=1)
        lib['admissible']=np.concatenate([lib['admissible'],mask[:,None]],axis=1)
        records.append(str(source.relative_to(ROOT)))
    lib['metadata']={**lib['metadata'],'version':'direct_case_certified_v20','new_execution_records':records}
    return lib
