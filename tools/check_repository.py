"""Check public-package structure, portable source imports and citation metadata."""
from pathlib import Path
import ast, csv, json, re, sys
ROOT=Path(__file__).resolve().parents[1]
assert (ROOT/'README.md').is_file() and (ROOT/'CITATION.cff').is_file()
modules={p.stem for p in (ROOT/'src/ecf_legacy').glob('*.py')}
count=0
for p in (ROOT/'src/ecf_legacy').glob('*.py'):
    text=p.read_text(encoding='utf8');ast.parse(text)
    assert not re.search(r'[A-Za-z]:[\\/](?:Users|Program Files)',text),p
    assert 'tmp/revision_runtime' not in text and 'tmp/evidence_runtime' not in text,p
    count+=1
with (ROOT/'data/REFERENCE_MANIFEST.csv').open(encoding='utf8') as f:
    rows=list(csv.DictReader(f))
assert all(int(r['bytes'])<25*1024*1024 for r in rows)
assert all(all(ord(c)<128 for c in r['path']) for r in rows)
citation=(ROOT/'CITATION.cff').read_text(encoding='utf8')
assert 'cff-version: 1.2.0' in citation and citation.count('family-names:')==4
assert '10.0000/' not in citation and 'YOUR_USERNAME' not in citation
print(json.dumps({'passed':True,'compiled_scientific_modules':count,'archived_files':len(rows),'all_filenames_ascii':True,'largest_file_bytes':max(int(r['bytes']) for r in rows)}))
