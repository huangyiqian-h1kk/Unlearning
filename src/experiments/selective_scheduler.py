"""Dependency-aware dispatch; scientific configs and task identities stay fixed."""
from collections import Counter
from pathlib import Path
from conrep.night import campaign as night
from conrep.night.io import read, file_sha, sha

POLICY = 'selective-dependencies-v2'
WAITING = 'waiting_for_reference'


def reference_status(task, state):
    if task['kind'] == 'reference':
        return 'completed'
    return state['tasks'][task['model']+'-sft-reference']['status']


def can_evaluate(task, state):
    return reference_status(task, state) == 'completed'


def ready(plan, state):
    """Stages are priorities inside fixed work, and strict gates for later grids."""
    eligible = []
    for task in plan['tasks']:
        record = state['tasks'][task['id']]
        if not night.ready_to_claim(record):
            continue
        if task['stage'] >= 100 and any(
            t['stage'] < task['stage'] and state['tasks'][t['id']]['status'] != 'completed'
            for t in plan['tasks']
        ):
            continue
        if task['kind'] != 'reference':
            ref = reference_status(task, state)
            if ref != 'completed':
                # Existing weights and already-trained tasks only need evaluation.
                # A failed reference must be explicitly repaired, not worked around.
                if (task['kind'] != 'train' or record.get(WAITING)
                        or ref not in ('pending', 'running', 'paused', 'interrupted')):
                    continue
        eligible.append(task)
    return eligible


def task_order(task, worker_id):
    return (task['stage'], task.get('preferred_worker') != worker_id, task['id'])


def record_result(record, rc, *, deferred=False, stopped=False, error=None, elapsed=0):
    if deferred:
        if rc != 0 or stopped:
            raise ValueError('Only successfully trained work can await a reference')
        status = 'pending'
    else:
        status = ('interrupted' if stopped else 'paused') if rc == 75 else 'completed' if rc == 0 else 'failed'
    record.update(status=status, exit_code=rc, error=error,
                  elapsed_seconds=record.get('elapsed_seconds', 0)+elapsed)
    record[WAITING] = deferred


def describe(plan, state):
    eligible = ready(plan, state)
    return dict(policy=POLICY, ready_tasks=len(eligible),
        ready_by_stage=dict(Counter(str(t['stage']) for t in eligible)),
        ready_by_kind=dict(Counter(t['kind'] for t in eligible)),
        reference_statuses={t['id']:state['tasks'][t['id']]['status']
                            for t in plan['tasks'] if t['kind']=='reference'},
        trained_waiting_for_reference=sum(r['status']=='pending' and bool(r.get(WAITING)) for r in state['tasks'].values()),
        fixed_incomplete=sum(t['stage']<100 and state['tasks'][t['id']]['status']!='completed'
                             for t in plan['tasks']),
        fixed_stages='priority only; no global barrier',
        grid_gate='all earlier fixed/grid tasks must be completed')


def verify_controller(campaign, plan):
    """Verify a separately frozen controller without changing training provenance."""
    controller = plan.get('controller')
    if not controller:
        return
    root = Path(controller['root']).resolve()
    if not root.is_relative_to(Path(campaign).resolve()/'controllers'):
        raise ValueError('Controller must be frozen inside this campaign')
    if controller['policy'] != POLICY or controller['base_source_hash'] != plan['source_hash']:
        raise ValueError('Controller/scientific source mismatch')
    marker = root/'source.json'
    if file_sha(marker) != controller['manifest_sha256']:
        raise ValueError('Controller source manifest changed')
    manifest = read(marker)
    if sha(manifest['files']) != controller['source_hash'] or manifest['base_source_hash'] != plan['source_hash']:
        raise ValueError('Controller source identity changed')
    if plan['entry'] != str(root/'code/scripts/abci/0390_selective.py') or plan['shell'] != str(root/'code/scripts/abci/0390_selective_worker.sh'):
        raise ValueError('Controller entry/shell mismatch')
    base = read(Path(campaign)/'source.json')['files']
    allowed = {'src/experiments/selective_campaign.py', 'src/experiments/selective_scheduler.py'}
    if set(manifest['files']) != set(base) | {'src/experiments/selective_scheduler.py'}:
        raise ValueError('Unexpected controller file set')
    for name, expected in manifest['files'].items():
        path = (root/'code'/name).resolve()
        if not path.is_relative_to(root/'code') or file_sha(path) != expected:
            raise ValueError('Frozen controller changed: '+name)
        if name not in allowed and expected != base[name]:
            raise ValueError('Scientific source changed in controller overlay: '+name)
