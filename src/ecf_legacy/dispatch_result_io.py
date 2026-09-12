"""Result serialization only; independent of every scheduling method."""
from pathlib import Path
import json,hashlib
import numpy as np

def save(out,name,result):
    out=Path(out)
    def encode(value):
        if isinstance(value,np.generic):return value.item()
        if isinstance(value,np.ndarray):return value.tolist()
        raise TypeError(f'Unsupported result type: {type(value).__name__}')
    metadata={k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
    # Serialize first, so an unsupported scalar cannot replace just the arrays.
    json.dumps(metadata,default=encode)
    array_tmp=out/f'{name}.pending.npz';json_tmp=out/f'{name}.pending.json'
    np.savez_compressed(array_tmp,**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
    metadata['_arrays_sha256']=hashlib.sha256(array_tmp.read_bytes()).hexdigest()
    json_tmp.write_text(json.dumps(metadata,indent=2,default=encode),encoding='utf-8')
    array_tmp.replace(out/f'{name}.npz');json_tmp.replace(out/f'{name}.json')

def load(out,name):
    out=Path(out);path=out/f'{name}.json'
    if not path.exists():return None
    metadata=json.loads(path.read_text(encoding='utf-8'));arrays=out/f'{name}.npz'
    if '_arrays_sha256' in metadata:
        assert hashlib.sha256(arrays.read_bytes()).hexdigest()==metadata['_arrays_sha256'],f'Mismatched result pair: {name}'
    return {**metadata,**dict(np.load(arrays))}
