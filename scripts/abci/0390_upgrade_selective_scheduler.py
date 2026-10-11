#!/usr/bin/env python3
"""Install a separate frozen controller and hand off the supervisor, not PBS jobs."""
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

CONTROLLER = 'src/experiments/selective_campaign.py'
POLICY_FILE = 'src/experiments/selective_scheduler.py'
POLICY = 'selective-dependencies-v2'


def checked_overlay(before, after):
    """Only dispatch/lifecycle functions may change; training/eval code is frozen."""
    old, new = ast.parse(before), ast.parse(after)
    allowed = {'verify', 'ready', 'worker', 'watch', 'main'}
    def fixed(tree):
        return [ast.dump(n, include_attributes=False) for n in tree.body
                if not (isinstance(n, ast.FunctionDef) and n.name in allowed)
                and not (isinstance(n, ast.ImportFrom) and n.module == 'selective_scheduler')]
    if fixed(old) != fixed(new):
        raise ValueError('Controller overlay changes non-dispatch code')
    def training_block(tree):
        worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
        return next(n for n in ast.walk(worker) if isinstance(n, ast.If)
                    and ast.unparse(n.test) == "task['kind'] == 'train' and (not trained(task, cfg))")
    if ast.dump(training_block(old), include_attributes=False) != ast.dump(training_block(new), include_attributes=False):
        raise ValueError('Controller overlay changes calibration/smoke/training execution')


def freeze(campaign, root, ref):
    from conrep.night.io import read, write, sha, file_sha
    source = read(campaign/'source.json')
    commit = subprocess.check_output(['git','rev-parse','--verify',ref+'^{commit}'], cwd=root, text=True).strip()
    blobs = {name:subprocess.check_output(['git','show',f'{commit}:{name}'], cwd=root)
             for name in (CONTROLLER, POLICY_FILE)}
    checked_overlay((campaign/'code'/CONTROLLER).read_bytes(), blobs[CONTROLLER])
    old_reader = ast.parse((campaign/'code/src/experiments/selective_data.py').read_text())
    row_reader = next(n for n in old_reader.body if isinstance(n, ast.FunctionDef) and n.name=='rows')
    if any(isinstance(n, ast.Attribute) and n.attr=='splitlines' for n in ast.walk(row_reader)):
        raise ValueError('Apply the MMLU reader repair before upgrading dispatch')
    folder = campaign/'controllers'/('dependencies-v2-'+commit[:12])
    expected = dict(source['files'])
    expected.update({name:hashlib.sha256(data).hexdigest() for name,data in blobs.items()})
    metadata = dict(ref=commit, policy=POLICY, base_source_hash=source['source_hash'],
                    files=expected, source_hash=sha(expected))
    if folder.exists():
        if read(folder/'source.json') != metadata:
            raise ValueError('Existing controller snapshot differs')
    else:
        folder.parent.mkdir(parents=True, exist_ok=True)
        stage = folder.with_name(folder.name+'.building')
        if stage.exists():
            shutil.rmtree(stage)
        for name in source['files']:
            destination = stage/'code'/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(campaign/'code'/name, destination)
        for name,data in blobs.items():
            destination = stage/'code'/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        write(stage/'source.json', metadata)
        stage.rename(folder)
    for name,expected_hash in expected.items():
        if file_sha(folder/'code'/name) != expected_hash:
            raise ValueError('Frozen controller changed: '+name)
    return dict(policy=POLICY, root=str(folder), ref=commit,
                base_source_hash=source['source_hash'], source_hash=metadata['source_hash'],
                manifest_sha256=file_sha(folder/'source.json'))


def controller_plan(plan, controller):
    result = dict(plan)
    root = Path(controller['root'])/'code/scripts/abci'
    result.update(controller=controller, entry=str(root/'0390_selective.py'),
                  shell=str(root/'0390_selective_worker.sh'))
    # The historical renderer may prefer this optional field.
    if 'controller_shell' in result:
        result['controller_shell'] = result['shell']
    return result


def load_policy(controller):
    spec = importlib.util.spec_from_file_location('selective_upgrade_policy',Path(controller['root'])/'code'/POLICY_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def wait_supervisor(campaign, timeout=95):
    """Old supervisors notice the campaign handoff state on their next poll."""
    from conrep.night.io import locked
    limit = time.monotonic()+timeout
    announced = False
    while True:
        manager = locked(campaign/'supervisor.lock', blocking=False)
        try:
            manager.__enter__()
            break
        except BlockingIOError:
            if not announced:
                print('Waiting for supervisor handoff; running PBS children continue.', flush=True)
                announced = True
            if time.monotonic() >= limit:
                raise TimeoutError('Supervisor did not release its lock; existing jobs were not stopped')
            time.sleep(.5)
    try:
        yield
    finally:
        manager.__exit__(None,None,None)


def handoff(campaign, controller):
    from conrep.night.io import read, write, locked
    from conrep.night.campaign import state_transaction
    folder = Path(controller['root'])
    policy = load_policy(controller)
    committed = False
    original_status = None
    with locked(campaign/'start.lock'):
        if (campaign/'STOP').exists():
            raise ValueError('STOP is set; explicitly resume before requesting a live upgrade')
        current = read(campaign/'plan.json')
        if current.get('controller') == controller:
            policy.verify_controller(campaign,current)
            with state_transaction(campaign) as state:
                if state['status']=='controller_handoff':
                    state['status']='running'
            print('Controller already installed; no task/config/result changes.', flush=True)
            return
        if current.get('controller'):
            raise ValueError('A different controller is already installed; no replacement performed')
        if not (folder/'plan-before.json').exists():
            write(folder/'plan-before.json',current)
        with state_transaction(campaign) as state:
            original_status = state['status']
            if original_status == 'completed':
                raise ValueError('Campaign already complete; no scheduler upgrade needed')
            if original_status=='controller_handoff' and (folder/'handoff.json').exists():
                original_status=read(folder/'handoff.json')['previous_status']
            write(folder/'handoff.json',dict(time=time.time(),previous_status=original_status,
                  workers={k:dict(v) for k,v in state['workers'].items()}))
            state['status']='controller_handoff'
        try:
            with wait_supervisor(campaign):
                # Any previous in-flight submit is now finished. Keep the same
                # account lock while installing the new dispatch pointer.
                project=Path(current['project_root'])
                with locked(project/'logs/0390/jobs/submit.lock'):
                    plan=controller_plan(read(campaign/'plan.json'),controller)
                    policy.verify_controller(campaign,plan)
                    with state_transaction(campaign) as state:
                        if (campaign/'STOP').exists():
                            raise ValueError('STOP requested during handoff')
                        write(campaign/'plan.json',plan)
                        state['status']='running'
                    committed=True
                    write(folder/'INSTALLED.json',dict(time=time.time(),controller=controller,
                        scientific_source_hash=plan['source_hash'],
                        note='No PBS cancellation, config rewrite, identity migration or result edit'))
        finally:
            if not committed:
                with state_transaction(campaign) as state:
                    if state['status']=='controller_handoff':
                        state['status']=original_status


def start_controller(campaign):
    from conrep.night.io import read
    plan=read(campaign/'plan.json')
    subprocess.run([plan['python'],'-I',plan['entry'],'arm','--campaign',str(campaign)],
                   cwd=plan['project_root'],check=True,timeout=60)
    # Wait for the first dispatch tick, not for GPUs or experiment completion.
    limit=time.monotonic()+45
    while time.monotonic()<limit:
        if (campaign/'heartbeat.json').exists():
            heartbeat=read(campaign/'heartbeat.json')
            supervisor=read(campaign/'supervisor.json')
            if heartbeat.get('pid')==supervisor.get('pid') and heartbeat.get('dispatch',{}).get('policy')==POLICY:
                print(json.dumps(dict(status='dependency scheduler active',dispatch=heartbeat['dispatch'],
                    account_active_at_tick=heartbeat.get('account_active'),peer_reserved=heartbeat.get('peer_reserved')),indent=2),flush=True)
                return
        if read(campaign/'state.json')['status'] not in ('running','controller_handoff'):
            break
        time.sleep(.5)
    print('Supervisor started; inspect current status/heartbeat for its next tick.',flush=True)


def upgrade(root,campaign,ref):
    root,campaign=Path(root).resolve(),Path(campaign).resolve()
    sys.path.insert(0,str(campaign/'code/src'))
    from conrep.night.io import read,locked
    from conrep.night import campaign as night
    from experiments.selective_campaign import verify
    with locked(campaign/'controller-upgrade.lock',blocking=False):
        plan=read(campaign/'plan.json')
        if Path(plan['project_root']).resolve()!=root:
            raise ValueError('Project root mismatch')
        verify(campaign,plan)
        # Fail before handoff if the scheduler cannot be queried.
        active=night.active_jobs(night.qstat(),__import__('getpass').getuser())
        print(json.dumps(dict(active_account_jobs=list(active),source_hash=plan['source_hash'])),flush=True)
        controller=freeze(campaign,root,ref)
        if plan.get('controller'):
            if plan['controller']['source_hash']!=controller['source_hash']:
                raise ValueError('A different controller revision is installed; no live replacement attempted')
            controller=plan['controller']
        error=None
        try:
            handoff(campaign,controller)
        except Exception as exc:
            error=exc
        # A failed handoff restores the old campaign state. Re-arm whichever
        # controller owns the plan, unless the user requested STOP meanwhile.
        if not (campaign/'STOP').exists():
            if error:
                plan=read(campaign/'plan.json')
                subprocess.run([plan['python'],'-I',plan['entry'],'arm','--campaign',str(campaign)],
                               cwd=root,check=True,timeout=60)
            else:
                start_controller(campaign)
        if error:
            raise error
        print('PLAN_ENTRY='+read(campaign/'plan.json')['entry'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','campaign','ref'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    upgrade(args.root,args.campaign,args.ref)
