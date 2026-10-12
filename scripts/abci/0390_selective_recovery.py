#!/usr/bin/env python3
"""Verified Python launcher for additive selective evaluation recovery."""
import hashlib
import json
from pathlib import Path
import sys

def main():
    manifest = Path(sys.argv[1]).resolve()
    if sys.argv[2] != '--':
        raise ValueError('Expected -- before original Python arguments')
    meta = json.loads(manifest.read_text())
    plan = json.loads((Path(meta['campaign'])/'plan.json').read_text())
    expected = dict(manifest=str(manifest), sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    policy=meta['policy'], ref=meta['ref'])
    if plan.get('evaluation_repair') != expected or plan['source_hash'] != meta['scientific_source_hash']:
        raise ValueError('Evaluation repair/scientific provenance mismatch')
    for name, digest in meta['files'].items():
        path = (manifest.parent/name).resolve()
        if not path.is_relative_to(manifest.parent) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen evaluation repair changed: '+name)
    sys.path.insert(0,str(Path(meta['original_entry']).resolve().parents[2]/'src'))
    import experiments
    experiments.__path__.append(str(manifest.parent/'code/src/experiments'))
    from experiments.selective_recovery import dispatch
    dispatch(meta,sys.argv[3:])

if __name__ == '__main__':
    main()
