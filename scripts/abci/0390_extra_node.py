#!/usr/bin/env python3
"""Scoped 3+1 supervisor; workers always use their original frozen entry points."""
import argparse
import contextlib
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

POLICY = '0390-t0-extra-node-v1'
KEY = 'temporary_extra_node'
SOURCE = 'scripts/abci/0390_extra_node.py'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextlib.contextmanager
def locked(path, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def context(campaign):
    plan = read(campaign/'plan.json')
    # Respect an already installed dependency controller. No science is overlaid.
    sys.path.insert(0, str(Path(plan['entry']).resolve().parents[2]/'src'))
    from experiments import selective_campaign
    from conrep.night import campaign as night
    return plan, selective_campaign, night


def policy_plans(plan):
    policy = plan[KEY]
    if policy['policy'] != POLICY or policy['limits'] != dict(parent=3, companion=1, account=4):
        raise ValueError('Unexpected temporary capacity policy')
    entry = Path(policy['entry']).resolve()
    child = Path(policy['companion']).resolve()
    if not entry.is_relative_to(child/'controllers') or file_sha(entry) != policy['entry_sha256']:
        raise ValueError('Frozen capacity supervisor changed')
    if Path(plan['campaign']).resolve() not in (Path(policy['parent']), child):
        raise ValueError('Capacity exception belongs to a different campaign')
    plans = {role: read(Path(policy[role])/'plan.json') for role in ('parent', 'companion')}
    for role, other in plans.items():
        if (other.get(KEY) != policy or other['source_hash'] != policy['science'][role]
                or other['project_root'] != plan['project_root']
                or Path(other['campaign']).resolve() != Path(policy[role])):
            raise ValueError('The paired capacity policy/source identities do not agree')
    if Path(plans['companion']['companion_parent']).resolve() != Path(policy['parent']):
        raise ValueError('General-only parent mismatch')
    return policy, plans


def occupied(state, active):
    return {w['job_id'] for w in state['workers'].values() if w.get('job_id')
            and (w['job_id'] in active or w.get('reconciled_job') != w['job_id'])}


def free_slots(plan, state, active, allocated_nodes):
    """Called under the existing shared submit.lock, including both qsub journals."""
    policy, plans = policy_plans(plan)
    role = 'companion' if Path(plan['campaign']).resolve() == Path(policy['companion']) else 'parent'
    states = {r: state if r == role else read(Path(p['campaign'])/'state.json') for r, p in plans.items()}
    if any(w['status'] == 'submitting' and not w.get('job_id')
           for s in states.values() for w in s['workers'].values()):
        return 0  # An ambiguous qsub may already own the fourth node.
    owned = {r: occupied(s, active) for r, s in states.items()}
    if owned['parent'] & owned['companion']:
        raise ValueError('A PBS allocation cannot belong to both campaigns')
    child_jobs = owned['companion']
    if len(child_jobs) > 1 or any(allocated_nodes(active[j]) != 1 for j in child_jobs if j in active):
        return 0
    unseen = (owned['parent'] | child_jobs) - set(active)
    paired = {str(Path(policy[r]).resolve()) for r in ('parent', 'companion')}
    peers = {str(Path(p).resolve()) for other in plans.values() for p in other.get('peers', [])} - paired
    reserve = 0
    for peer in peers:
        root = Path(peer)
        if not (root/'state.json').is_file() or (root/'COMPLETE.json').exists() or (root/'STOP').exists():
            continue
        record = read(root/'state.json')
        if (record.get('job_id') not in active
                and record.get('status') not in ('blocked', 'completed', 'stopped', 'failed')):
            reserve += 1
    jobs = len(active) + len(unseen) + reserve
    nodes = sum(allocated_nodes(j) for j in active.values()) + len(unseen) + reserve
    if role == 'companion':
        return max(0, min(1-len(child_jobs), 4-jobs, 4-nodes))
    # Only this companion is exempt from the ordinary cap of three. The parent
    # never borrows its fourth slot, including after the companion completes.
    return max(0, min(3-len(owned['parent']), 3-(jobs-len(child_jobs)),
                      3-(nodes-len(child_jobs)), 4-jobs, 4-nodes))


def refresh_companion(campaign, plan):
    policy_plans(plan)
    with locked(campaign/'state.lock'):
        state = read(campaign/'state.json')
        for key, worker in state['workers'].items():
            if key != '0':
                if worker.get('job_id') and worker.get('reconciled_job') != worker['job_id']:
                    raise ValueError('Extra-node mode permits only companion worker 0')
                worker['status'] = 'deferred_to_parent'
        state['job_id'] = state['workers']['0'].get('job_id')
        state['allocation_limit'] = 1
        write(campaign/'state.json', state)
    return False


def install_hooks(campaign, plan, api, night):
    policy, _ = policy_plans(plan)
    api.free_slots = lambda p, s, a: free_slots(p, s, a, night.allocated_nodes)
    original_submit = api.submit_available
    def submit(path, current):
        original_submit(path, current)
        write(path/'capacity-heartbeat.json', dict(time=time.time(), pid=os.getpid(), policy=POLICY,
              parent_limit=3, companion_limit=1, account_limit=4))
    api.submit_available = submit
    if str(campaign) == policy['companion']:
        from experiments import t0_general_only as companion
        companion.attach_reservation = lambda path, p: policy_plans(p)
        companion.reservation_acknowledged = lambda path, p: bool(policy_plans(p))
        companion.refresh_workers = refresh_companion
        return companion.watch
    return api.watch


def arm(campaign, plan, api, night, *, resume=False, retry_failed=False, requeue_jobs=()):
    policy_plans(plan)
    api.verify(campaign, plan)
    with locked(campaign/'start.lock'):
        try:
            with locked(campaign/'supervisor.lock', blocking=False):
                pass
        except BlockingIOError:
            marker = campaign/'capacity-supervisor.json'
            current = read(campaign/'supervisor.json') if (campaign/'supervisor.json').exists() else {}
            if not marker.exists() or read(marker).get('pid') != current.get('pid'):
                raise ValueError('A legacy supervisor is running; repeat activate to hand it off')
            if retry_failed or requeue_jobs:
                raise ValueError('Use stop and wait for supervisor exit before explicit recovery')
            print('3+1 supervisor already running; no duplicate submission', flush=True)
            return
        if (campaign/'STOP').exists() and not resume:
            raise ValueError('STOP preserved; use resume explicitly')
        active = night.active_jobs(night.qstat(), getpass.getuser())
        night.reconcile(campaign, plan, active)
        if str(campaign) == plan[KEY]['companion']:
            refresh_companion(campaign, plan)
        if requeue_jobs:
            night.requeue_explicit_jobs(campaign, requeue_jobs, active)
        with night.state_transaction(campaign) as state:
            if state['status'] == 'completed':
                print('Campaign already complete', flush=True)
                return
            if retry_failed:
                for task in state['tasks'].values():
                    if task['status'] == 'failed':
                        task.update(status='pending', error=None)
            for worker in state['workers'].values():
                if worker['status'] == 'drained':
                    worker['status'] = 'available'
            state.update(status='running', started_at=state.get('started_at') or time.time(), deadline=253402300799)
        if resume:
            (campaign/'STOP').unlink(missing_ok=True)
        with (campaign/'supervisor.log').open('a') as log:
            proc = subprocess.Popen([plan['python'], '-I', plan[KEY]['entry'], 'watch', '--campaign', str(campaign)],
                    cwd=plan['project_root'], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
        time.sleep(.5)
        if proc.poll() is not None:
            raise RuntimeError('Supervisor did not start; inspect supervisor.log')
        limit = 1 if str(campaign) == plan[KEY]['companion'] else 3
        print(f'Supervisor PID={proc.pid}; this campaign limit={limit}; paired account cap=4.', flush=True)


@contextlib.contextmanager
def wait_supervisor(campaign, timeout=95):
    end = time.monotonic()+timeout
    while True:
        manager = locked(campaign/'supervisor.lock', blocking=False)
        try:
            manager.__enter__()
            break
        except BlockingIOError:
            if time.monotonic() >= end:
                raise TimeoutError('Old supervisor has not released its lock; no PBS jobs cancelled')
            time.sleep(.25)
    try:
        yield
    finally:
        manager.__exit__(None, None, None)


def freeze(root, parent, child, ref):
    commit = subprocess.check_output(['git', 'rev-parse', '--verify', ref+'^{commit}'], cwd=root, text=True).strip()
    content = subprocess.check_output(['git', 'show', f'{commit}:{SOURCE}'], cwd=root)
    folder = child/'controllers'/('extra-node-'+commit[:12])
    entry = folder/'0390_extra_node.py'
    digest = hashlib.sha256(content).hexdigest()
    policy = dict(policy=POLICY, parent=str(parent), companion=str(child), entry=str(entry), entry_sha256=digest,
                  ref=commit, limits=dict(parent=3, companion=1, account=4),
                  science={r: read(p/'plan.json')['source_hash'] for r, p in (('parent', parent), ('companion', child))},
                  expiry='Only this finite 14-training companion; no transferable fourth parent slot')
    if entry.exists():
        if file_sha(entry) != digest or read(folder/'policy.json') != policy:
            raise ValueError('Existing capacity supervisor snapshot differs')
    else:
        folder.mkdir(parents=True, exist_ok=True)
        entry.write_bytes(content)
        write(folder/'policy.json', policy)
    return policy


def command(policy, campaign, action, *extra):
    plan = read(campaign/'plan.json')
    return [plan['python'], '-I', policy['entry'], action, '--campaign', str(campaign), *extra]


def matching_supervisor(campaign):
    try:
        with locked(campaign/'supervisor.lock', blocking=False):
            return True  # No live owner; arm can restart the installed policy.
    except BlockingIOError:
        marker, owner = campaign/'capacity-supervisor.json', campaign/'supervisor.json'
        return (marker.exists() and owner.exists()
                and read(marker).get('policy') == POLICY and read(marker).get('pid') == read(owner).get('pid'))


def handoff_pair(root, parent, child, policy):
    originals, original_plans = {}, {}
    installed = False
    try:
        with contextlib.ExitStack() as locks:
            try:
                # Child first: its legacy watch may briefly take parent/start.lock.
                for path in (child, parent):
                    locks.enter_context(locked(path/'controller-upgrade.lock', blocking=False))
                    locks.enter_context(locked(path/'start.lock'))
                    current = read(path/'plan.json')
                    if current.get(KEY) not in (None, policy):
                        raise ValueError('Another temporary capacity policy is already installed')
                    if (path/'STOP').exists():
                        raise ValueError(f'STOP is preserved at {path}; resume it explicitly first')
                    with locked(path/'state.lock'):
                        state = read(path/'state.json')
                        receipt = Path(policy['entry']).parent/('handoff-'+path.name+'.json')
                        original = state['status']
                        if original == 'controller_handoff' and receipt.exists():
                            original = read(receipt)['previous_status']
                        originals[path] = original
                        write(receipt, dict(previous_status=original, time=time.time()))
                        state['status'] = 'controller_handoff'
                        write(path/'state.json', state)
                    print('Handing off supervisor: '+str(path)+'; existing training children continue.', flush=True)
                    locks.enter_context(wait_supervisor(path))
                with locked(root/'logs/0390/jobs/submit.lock'):
                    state = read(child/'state.json')
                    if any(w.get('job_id') and w.get('reconciled_job') != w['job_id']
                           for key, w in state['workers'].items() if key != '0'):
                        raise ValueError('Companion already has extra allocations; no jobs cancelled or policy changed')
                    # Fresh reads preserve updates made by workers during handoff.
                    for path in (parent, child):
                        current = read(path/'plan.json')
                        original_plans[path] = dict(current)
                        backup = Path(policy['entry']).parent/('before-'+path.name+'.json')
                        if not backup.exists():
                            write(backup, current)
                        current[KEY] = policy
                        if path == parent:
                            current['peers'] = [p for p in current.get('peers', []) if Path(p).resolve() != child]
                        write(path/'plan.json', current)
                    policy_plans(read(parent/'plan.json'))
                    refresh_companion(child, read(child/'plan.json'))
                    write(Path(policy['entry']).parent/'INSTALLED.json', dict(time=time.time(), policy=policy,
                        preserved='Original entries/shells, sources, configs, task identities, checkpoints and PBS job IDs'))
                    installed = True
            finally:
                if not installed:
                    for path, before in original_plans.items():
                        write(path/'plan.json', before)
                for path, original in originals.items():
                    with locked(path/'state.lock'):
                        state = read(path/'state.json')
                        if state['status'] == 'controller_handoff':
                            state['status'] = original
                            write(path/'state.json', state)
    except Exception:
        # Locks have released. Re-arm an interrupted old supervisor on failure;
        # tasks and PBS allocations are untouched, including concurrent progress.
        for path, original in originals.items():
            if original == 'running' and not (path/'STOP').exists():
                current = read(path/'plan.json')
                entry = current.get(KEY, {}).get('entry', current['entry'])
                subprocess.run([current['python'], '-I', entry, 'arm', '--campaign', str(path)],
                               cwd=root, timeout=60, check=False)
        raise


def activate(root, parent, child, ref, allow_four_nodes):
    if not allow_four_nodes:
        raise ValueError('This mode requires the explicit --allow-four-nodes authorization')
    root, parent, child = map(lambda p: Path(p).resolve(), (root, parent, child))
    parent_plan, child_plan = read(parent/'plan.json'), read(child/'plan.json')
    if (parent == child or parent_plan['project_root'] != str(root) or child_plan['project_root'] != str(root)
            or Path(child_plan['companion_parent']).resolve() != parent):
        raise ValueError('Expected one selective parent and its isolated general-only companion')
    if (len(child_plan['tasks']) != 14 or child_plan['grid_phase'] != 3
            or any(t['kind'] != 'train' or t['target'] != 'T0-General-only' for t in child_plan['tasks'])):
        raise ValueError('The fourth slot is restricted to the fixed 14-training T0 supplement')
    policy = freeze(root, parent, child, ref)
    # Verify each campaign with its own frozen imports in a fresh process.
    for path in (parent, child):
        subprocess.run(command(policy, path, 'verify'), cwd=root, check=True, timeout=60)
    with locked(child/'extra-node-install.lock', blocking=False):
        if not all(read(p/'plan.json').get(KEY) == policy and matching_supervisor(p) for p in (parent, child)):
            handoff_pair(root, parent, child, policy)
        else:
            policy_plans(read(parent/'plan.json'))
            print('3+1 policy already installed; preserving running supervisors.', flush=True)
    # Starting happens only after both plans agree and all handoff locks release.
    for path in (parent, child):
        status = read(path/'state.json')['status']
        if status in ('running', 'prepared'):
            subprocess.run(command(policy, path, 'arm'), cwd=root, check=True, timeout=60)
        else:
            print(f'{path.name}: preserved status={status}; no failed tasks reset.', flush=True)
    print(json.dumps(dict(policy=POLICY, parent_limit=3, companion_limit=1, account_limit=4,
                          entry=policy['entry'], no_scientific_changes=True), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('activate', 'verify', 'arm', 'resume', 'watch', 'status', 'stop', 'export'))
    for name in ('root', 'parent', 'campaign', 'ref'):
        parser.add_argument('--'+name)
    parser.add_argument('--allow-four-nodes', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--requeue-job', action='append', default=[])
    parser.add_argument('--include-predictions', action='store_true')
    args = parser.parse_args()
    if args.command == 'activate':
        if not all((args.root, args.parent, args.campaign, args.ref)):
            parser.error('activate requires --root, --parent, --campaign and --ref')
        activate(args.root, args.parent, args.campaign, args.ref, args.allow_four_nodes)
        return 0
    if not args.campaign:
        parser.error('--campaign is required')
    campaign = Path(args.campaign).resolve()
    plan, api, night = context(campaign)
    if args.command == 'verify':
        api.verify(campaign, plan)
        night.active_jobs(night.qstat(), getpass.getuser())
        print('Verified scientific snapshot and PBS access: '+str(campaign), flush=True)
        return 0
    watch = install_hooks(campaign, plan, api, night)
    if args.command in ('arm', 'resume'):
        arm(campaign, plan, api, night, resume=args.command == 'resume',
            retry_failed=args.retry_failed, requeue_jobs=args.requeue_job)
    elif args.command == 'watch':
        write(campaign/'capacity-supervisor.json', dict(pid=os.getpid(), time=time.time(), policy=POLICY))
        watch(campaign)
    else:
        if args.command == 'status':
            print(json.dumps(dict(policy=POLICY, parent_limit=3, companion_limit=1, account_limit=4)))
        sys.argv = [plan['entry'], args.command, '--campaign', str(campaign)]
        if args.include_predictions:
            sys.argv.append('--include-predictions')
        return api.main()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
