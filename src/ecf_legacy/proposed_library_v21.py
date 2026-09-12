"""Append only actual-carrier and modal-certified priority trajectories."""
from pathlib import Path
import json,numpy as np
from proposed_library_v20 import library as original
ROOT=__import__("repository_paths").ROOT

def library(net):
    lib=original(net);K=lib['P'].shape[0];records=[]
    for source in sorted((ROOT/'results/full_cycle_v13').glob('*_priority_v21_*_execution.json')):
        record=json.loads(source.read_text(encoding='utf8'))
        if not record.get('certified'):continue
        k=record['site'];power=lib['P'][:,0].copy();admissible=np.zeros(K,bool);admissible[k]=True
        power[k]=np.load(source.with_suffix('.npz'))['power_w'].mean(axis=0)*int(lib['cells_per_series'][k]*3)/1e6
        lib['P']=np.concatenate([lib['P'],power[:,None,:]],axis=1)
        lib['admissible']=np.concatenate([lib['admissible'],admissible[:,None]],axis=1)
        records.append(str(source.relative_to(ROOT)))
    assert len(records)==18,(len(records),'The complete priority batch is required before freezing the library.')
    lib['metadata']={**lib['metadata'],'version':'certified_priority_v21','priority_execution_records':records}
    return lib
