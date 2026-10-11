#!/usr/bin/env python3
"""Freeze the running server's verified parent plus the selective implementation."""
import argparse
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

OVERLAY=['src/experiments/selective_'+n+'.py' for n in
         ('common','design','data','eval','train','calibrate','prepare','campaign','scheduler','grid','export')]+[
 'scripts/abci/0390_selective.py','scripts/abci/0390_selective_worker.sh',
 'scripts/abci/0390_plan_selective_extensive.py','src/conrep/night/io.py','src/conrep/night/trainer.py']

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare(root,parent,campaign,ref):
    root,parent,campaign=(Path(p).resolve() for p in (root,parent,campaign))
    if campaign==parent or campaign.is_relative_to(parent) or parent.is_relative_to(campaign):raise ValueError('Separate campaign required')
    plan=json.loads((parent/'plan.json').read_text());source=json.loads((parent/'source.json').read_text())
    if Path(plan['project_root']).resolve()!=root or plan['source_hash']!=source['source_hash']:raise ValueError('Parent provenance mismatch')
    for name,value in source['files'].items():
        p=(parent/'code'/name).resolve()
        if not p.is_relative_to(parent/'code') or digest(p)!=value:raise ValueError('Changed parent source: '+name)
    for name in ('src/experiments/validation_v5.py','src/experiments/pmc_scoring_v5.py','src/experiments/pmc_schema.py'):
        if name not in source['files']:raise ValueError('Need actual frozen v5 evaluator: '+name)
    commit=subprocess.check_output(['git','rev-parse','--verify',ref+'^{commit}'],cwd=root,text=True).strip()
    blobs={name:subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=root) for name in OVERLAY}
    trainer='src/conrep/night/trainer.py'
    def objective(data):
        tree=ast.parse(data)
        return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='losses'),include_attributes=False)
    if objective((parent/'code'/trainer).read_bytes())!=objective(blobs[trainer]):
        raise ValueError('Parent ConRep objective differs from the verified implementation; refusing a silent algorithm change')
    for p in (root/'third_party/0390_baselines').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':blobs[str(p.relative_to(root))]=p.read_bytes()
    metadata=dict(ref=commit,project_root=str(root),parent=str(parent),parent_source_hash=source['source_hash'],
                  overlay={n:hashlib.sha256(b).hexdigest() for n,b in blobs.items()})
    campaign.mkdir(parents=True,exist_ok=True)
    with (campaign/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        marker=campaign/'PREPARING.json'
        if marker.exists():
            if json.loads(marker.read_text())!=metadata:raise ValueError('Campaign already frozen from another revision')
        else:
            if set(p.name for p in campaign.iterdir())!={'prepare.lock'}:raise ValueError('Destination is not empty')
            marker.write_text(json.dumps(metadata,indent=2)+'\n')
        expected=dict(source['files'],**metadata['overlay']);code=campaign/'code'
        if not code.exists():
            stage=campaign/'.code-building'
            if stage.exists():shutil.rmtree(stage)
            for name in source['files']:
                dest=stage/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(parent/'code'/name,dest)
            for name,b in blobs.items():
                dest=stage/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
            stage.rename(code)
        for name,value in expected.items():
            if digest(code/name)!=value:raise ValueError('Changed frozen selective source: '+name)
        (campaign/'code-files.json').write_text(json.dumps(expected,indent=2)+'\n')
    print(f'FROZEN_CODE={code}\nNo jobs submitted.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','parent','campaign','ref'):p.add_argument('--'+name,required=True)
    a=p.parse_args();prepare(a.root,a.parent,a.campaign,a.ref)
