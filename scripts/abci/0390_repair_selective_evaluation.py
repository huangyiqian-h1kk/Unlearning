#!/usr/bin/env python3
"""Install an additive evaluation launcher in both 3+1 campaigns; keep all weights."""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

POLICY = 'selective-binary-mcq-recovery-v1'
FILES = ('src/experiments/selective_mcq.py', 'src/experiments/selective_recovery.py',
         'scripts/abci/0390_selective_recovery.py')


def read(path):
    return json.loads(Path(path).read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_capacity(plan):
    policy = plan['temporary_extra_node']
    path = Path(policy['entry'])
    if digest(path.read_bytes()) != policy['entry_sha256']:
        raise ValueError('Frozen capacity supervisor changed')
    spec = importlib.util.spec_from_file_location('selective_capacity_repair',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.policy_plans(plan)
    return module


def freeze(root, campaign, ref, capacity):
    plan = read(campaign/'plan.json')
    folder = campaign/'controllers'/('evaluation-repair-v1-'+ref[:12])
    manifest = folder/'manifest.json'
    original = plan.get('evaluation_repair')
    if original:
        if original['ref'] != ref:
            raise ValueError('A different evaluation repair is installed; use its pinned revision')
        meta = read(original['manifest'])
        if digest(Path(original['manifest']).read_bytes()) != original['sha256']:
            raise ValueError('Installed repair manifest changed')
        for name, expected in meta['files'].items():
            if digest((Path(original['manifest']).parent/name).read_bytes()) != expected:
                raise ValueError('Installed repair file changed: '+name)
        return original, meta
    blobs = {'code/'+name:subprocess.check_output(['git','show',f'{ref}:{name}'],cwd=root) for name in FILES}
    command = [plan['python'],'-I',str(folder/'code/scripts/abci/0390_selective_recovery.py'),str(manifest),'--']
    blobs['python'] = ('#!/bin/bash\nset -euo pipefail\nexec '+shlex.join(command)+' "$@"\n').encode()
    meta = dict(policy=POLICY, ref=ref, campaign=str(campaign), original_entry=plan['entry'],
        real_python=plan['python'], scientific_source_hash=plan['source_hash'],
        files={name:digest(data) for name,data in blobs.items()},
        compatibility='Original identity remains the base cache key; execution_identity records this additive repair',
        augmentation='Only incomplete ReLearn calibration; at most six additional missing-fact rounds; unchanged teacher/filters')
    if manifest.exists():
        if read(manifest) != meta:
            raise ValueError('Existing repair snapshot differs')
    for name,data in blobs.items():
        target = folder/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise ValueError('Frozen repair snapshot changed: '+name)
        target.write_bytes(data)
    (folder/'python').chmod(0o750)
    capacity.write(manifest,meta)
    return dict(manifest=str(manifest),sha256=digest(manifest.read_bytes()),policy=POLICY,ref=ref),meta


def preflight(campaign, meta):
    program = '''
import sys,json
from pathlib import Path
campaign=Path(sys.argv[1]);entry=Path(sys.argv[2]);addon=Path(sys.argv[3])
sys.path.insert(0,str(entry.parents[2]/'src'))
import experiments
experiments.__path__.append(str(addon/'code/src/experiments'))
from conrep.night.io import read
from conrep.night import campaign as night
from experiments.selective_campaign import verify,trained
from experiments.selective_mcq import extract_binary
from experiments.selective_data import rows
plan=read(campaign/'plan.json');verify(campaign,plan)
active=night.active_jobs(night.qstat(),__import__('getpass').getuser())
assert extract_binary('A',{'A':'male','B':'female'})['predicted']=='A'
assert extract_binary('C',{'A':'male','B':'female'})['predicted'] is None
assert extract_binary('A or B',{'A':'male','B':'female'})['predicted'] is None
checked=set();binary=four=0
for task in plan['tasks']:
 if task['kind']=='reference':continue
 cfg=read(task['config']);root=Path(cfg['selective']['task_path'])
 for partition in ('validation','test'):
  path=root/'probes'/partition/'mcq.jsonl'
  if path in checked:continue
  checked.add(path)
  for row in rows(path):
   keys=set(row['mapping'])
   if keys==set('AB') and row['attribute']=='gender':binary+=1
   elif keys==set('ABCD'):four+=1
   else:raise ValueError('Unsupported MCQ schema: '+str(path))
print(json.dumps(dict(campaign=str(campaign),binary_mcq_rows=binary,four_choice_rows=four,
                     active_account_jobs=list(active),frozen_science_verified=True)),flush=True)
'''
    subprocess.run([meta['real_python'],'-I','-c',program,str(campaign),meta['original_entry'],
                    str(Path(read(campaign/'plan.json').get('evaluation_repair',{}).get('manifest',
                        campaign/'controllers'/('evaluation-repair-v1-'+meta['ref'][:12])/'manifest.json')).parent)],
                   check=True,timeout=120)


def owned_repair_watch(campaign, capacity, ref):
    try:
        with capacity.locked(campaign/'supervisor.lock',blocking=False):
            return True
    except BlockingIOError:
        marker = campaign/'evaluation-repair-supervisor.json'
        return (marker.exists() and (campaign/'supervisor.json').exists()
            and read(marker).get('pid') == read(campaign/'supervisor.json').get('pid')
            and read(marker).get('ref') == ref)


def install_pair(root, campaigns, records, capacity):
    originals, plans = {}, {}
    committed = False
    try:
        with ExitStack() as locks:
            try:
                # Companion first: a legacy watch can take parent/start.lock.
                for campaign in reversed(campaigns):
                    locks.enter_context(capacity.locked(campaign/'controller-upgrade.lock',blocking=False))
                    locks.enter_context(capacity.locked(campaign/'start.lock'))
                    if (campaign/'STOP').exists():
                        raise ValueError('STOP is preserved; explicitly resume first: '+str(campaign))
                    with capacity.locked(campaign/'state.lock'):
                        state = read(campaign/'state.json')
                        receipt = Path(records[campaign][0]['manifest']).parent/'handoff.json'
                        original = state['status']
                        if original == 'controller_handoff' and receipt.exists():
                            original = read(receipt)['previous_status']
                        originals[campaign] = original
                        capacity.write(receipt,dict(previous_status=original,time=time.time()))
                        state['status'] = 'controller_handoff'
                        capacity.write(campaign/'state.json',state)
                    print('Handing off supervisor; native training children continue: '+str(campaign),flush=True)
                    locks.enter_context(capacity.wait_supervisor(campaign))
                with capacity.locked(root/'logs/0390/jobs/submit.lock'):
                    for campaign in campaigns:
                        record,meta = records[campaign]
                        current = read(campaign/'plan.json')
                        plans[campaign] = dict(current)
                        folder = Path(record['manifest']).parent
                        if not (folder/'plan-before.json').exists():
                            capacity.write(folder/'plan-before.json',current)
                        current.update(python=str(folder/'python'),evaluation_repair=record)
                        capacity.write(campaign/'plan.json',current)
                        capacity.write(folder/'INSTALLED.json',dict(time=time.time(),source_hash=current['source_hash'],
                            unchanged=['entry','shell','controller','temporary_extra_node','tasks','configs','checkpoints'],
                            changed=['Python subprocess dispatch for evaluation, incomplete calibration and supervisor recovery']))
                    capacity.policy_plans(read(campaigns[0]/'plan.json'))
                    committed = True
            finally:
                if not committed:
                    for campaign,plan in plans.items():
                        capacity.write(campaign/'plan.json',plan)
                for campaign,original in originals.items():
                    with capacity.locked(campaign/'state.lock'):
                        state = read(campaign/'state.json')
                        if state['status'] == 'controller_handoff':
                            state['status'] = original
                            capacity.write(campaign/'state.json',state)
    except Exception:
        for campaign,original in originals.items():
            if original == 'running' and not (campaign/'STOP').exists():
                p=read(campaign/'plan.json')
                subprocess.run([p['python'],'-I',p['temporary_extra_node']['entry'],'arm','--campaign',str(campaign)],
                               cwd=root,check=False,timeout=60)
        raise


def repair(root, parent, campaign, ref):
    root,parent,campaign = map(lambda p:Path(p).resolve(),(root,parent,campaign))
    ref=subprocess.check_output(['git','rev-parse','--verify',ref+'^{commit}'],cwd=root,text=True).strip()
    plan=read(campaign/'plan.json')
    capacity=load_capacity(plan)
    if (Path(plan['companion_parent']).resolve()!=parent or plan['project_root']!=str(root)
            or Path(plan['temporary_extra_node']['companion'])!=campaign):
        raise ValueError('Expected the authorized selective parent and T0 companion pair')
    campaigns=(parent,campaign)
    with capacity.locked(campaign/'evaluation-repair-install.lock',blocking=False):
        records={p:freeze(root,p,ref,capacity) for p in campaigns}
        for p,(_,meta) in records.items():
            preflight(p,meta)
        if not all(read(p/'plan.json').get('evaluation_repair')==records[p][0]
                   and owned_repair_watch(p,capacity,ref) for p in campaigns):
            install_pair(root,campaigns,records,capacity)
        else:
            print('Evaluation repair already installed; existing supervisors preserved.',flush=True)
    for p in campaigns:
        current=read(p/'plan.json')
        if read(p/'state.json')['status']=='completed':
            continue
        subprocess.run([current['python'],'-I',current['temporary_extra_node']['entry'],'arm','--campaign',str(p)],
                       cwd=root,check=True,timeout=60)
    print(json.dumps(dict(status='repair installed; eligible failed stages requeued once',ref=ref,
        parent=str(parent),companion=str(campaign),account_limit=4,training_source_unchanged=True),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','parent','campaign','ref'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    repair(args.root,args.parent,args.campaign,args.ref)
