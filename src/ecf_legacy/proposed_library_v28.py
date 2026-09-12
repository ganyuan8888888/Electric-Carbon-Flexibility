"""Extend the frozen v21 library with independently validated v28 trajectories."""
from pathlib import Path
import json,hashlib
import numpy as np
from proposed_library_v21 import library as original
ROOT=__import__("repository_paths").ROOT

def library(net):
    lib=original(net);K=lib['P'].shape[0];records=[]
    manifest=ROOT/'results/pareto_v28/frozen_columns.json'
    assert manifest.exists(),'Finish all column executions and freeze their manifest first.'
    for record in json.loads(manifest.read_text())['columns']:
        source=ROOT/record['file'];report=json.loads(source.read_text())
        assert report['certified'] and hashlib.sha256(source.with_suffix('.npz').read_bytes()).hexdigest()==record['array_sha256']
        k=report['site'];power=lib['P'][:,0].copy();mask=np.zeros(K,bool);mask[k]=True
        power[k]=np.load(source.with_suffix('.npz'))['power_w'].mean(axis=0)*int(lib['cells_per_series'][k]*3)/1e6
        lib['P']=np.concatenate([lib['P'],power[:,None,:]],axis=1)
        lib['admissible']=np.concatenate([lib['admissible'],mask[:,None]],axis=1)
        records.append(record)
    lib['metadata']={**lib['metadata'],'version':'pareto_certified_v28','v28_execution_records':records}
    return lib

def freeze():
    records=[]
    for date in ['2020-04-07','2020-10-17']:
        summary=ROOT/f'results/pareto_v28/{date}/physical_columns_summary.json'
        assert summary.exists(),('incomplete',date)
        rows=json.loads(summary.read_text());assert {r['site'] for r in rows}==set(range(6)),rows
        for row in rows:
            file=ROOT/'results/full_cycle_v13'/row['execution_file']
            assert json.loads(file.read_text())['certified']
            records.append(dict(file=str(file.relative_to(ROOT)),array_sha256=hashlib.sha256(file.with_suffix('.npz').read_bytes()).hexdigest()))
    (ROOT/'results/pareto_v28/frozen_columns.json').write_text(json.dumps(dict(columns=records),indent=2),encoding='utf8')
    print('Frozen original-physics columns:',len(records))

if __name__=='__main__':freeze()
