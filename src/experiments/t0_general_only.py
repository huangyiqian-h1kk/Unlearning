"""Frozen T0 supplement; one next-free slot beside the existing campaign."""
import copy
import getpass
import json
import os
from pathlib import Path
import shutil
import sys
import time

from conrep.night import campaign as night
from conrep.night.io import file_sha, locked, read, sha, write
from . import selective_campaign as campaign_api
from .selective_common import BASELINES
from .selective_design import CANDIDATES, digest
from .selective_prepare import environment, publish_task, task_config
from .retain_policy import POLICY, general_only

TARGET = 'T0-General-only'
ENTRY = 'scripts/abci/0390_t0_general_only.py'
SHELL = 'scripts/abci/0390_t0_general_only_worker.sh'
SUPERVISOR_POLICY = 't0-general-only-companion-v1'
_BASE_FREE_SLOTS = campaign_api.free_slots


def shared_free_slots(plan, state, active):
    available = _BASE_FREE_SLOTS(plan, state, active)
    parent_state = read(Path(plan['companion_parent'])/'state.json')
    # qsub visibility can lag qstat. Account for every parent reservation while
    # holding the same account-wide submission lock used by both controllers.
    unseen = sum(bool(w.get('job_id') and w['job_id'] not in active
                      and w.get('reconciled_job') != w['job_id'])
                 or bool(w['status'] == 'submitting' and not w.get('job_id'))
                 for w in parent_state['workers'].values())
    return max(0, available - unseen)


def selected_controls(parent_plan):
    result = []
    for model, variants in CANDIDATES.items():
        for variant in (variants[0], *BASELINES):
            matches = [task for task in parent_plan['tasks'] if task['model'] == model
                       and task['target'] == 'T0-C100' and task['variant'] == variant
                       and task['seed'] == 42 and task['kind'] == 'train']
            if len(matches) != 1:
                raise ValueError(f'Need one frozen T0-C100 control for {model}/{variant}')
            result.append(matches[0])
    return result


def make_data(parent, campaign):
    source = parent/'data/T0-C0'
    root = campaign/'data'/TARGET
    root.mkdir(parents=True, exist_ok=True)
    task = copy.deepcopy(read(source/'task.json'))
    if (task['counts']['forget'], task['counts']['retain_same_eval'],
            task['counts']['retain_other_eval']) != (100, 200, 900):
        raise ValueError('Expected the historical T0 F100/R-same200/R-other900')
    for name in ('forget.jsonl', 'general.jsonl'):
        shutil.copyfile(source/name, root/name)
    from .selective_data import rows, write_rows
    forget = rows(root/'forget.jsonl')
    id_map = {r['relation_id']: r['id'] for r in forget}
    if len(forget) != 100 or len(set(id_map.values())) != 100 or set(id_map) != set(task['membership']['forget']):
        raise ValueError('Original T0 training IDs do not map uniquely to the frozen forget facts')
    requests = rows(source/'forget_requests.jsonl')
    if {r['fact_id'] for r in requests} != set(id_map):
        raise ValueError('T0 canonical requests do not cover the same forget facts')
    # Legacy F rows use forget:0 etc., while selective requests use patient:attribute.
    # Preserve F bytes/order; adapt only request metadata so ReLearn joins correctly.
    write_rows(root/'forget_requests.jsonl', [dict(r, fact_id=id_map[r['fact_id']]) for r in requests])
    write(root/'forget-id-map.json', id_map)
    # Probe bytes and scoring targets are unchanged; only training membership differs.
    shutil.copytree(source/'probes', root/'probes', dirs_exist_ok=True)
    for name in ('retain.jsonl', 'retain_requests.jsonl'):
        if (root/name).exists():
            raise ValueError('Specified training data must not exist in the general-only directory')
    task['id'] = TARGET
    task['membership']['retain_train'] = []
    task['membership_sha256'] = digest(task['membership'])
    task['counts'].update(retain_train=0, retain_train_same=0, retain_train_other=0,
                          retain_same_unseen=200)
    task['train_hashes'] = {name: file_sha(root/(name+'.jsonl')) for name in ('forget', 'general')}
    task['training_source'] = 'Original T0 F and G bytes; specified retain unavailable to training'
    write(root/'task.json', task)
    write(campaign/'data/manifest.json', dict(schema='selective-v1', tasks={TARGET: dict(
        path=str(root), counts=task['counts'])}, parent=str(parent),
        retain_policy=POLICY, probes_equal_to='T0-C0', main_prompt_provided_facts=0))
    return task


def ablate_config(base, model, method, variant, campaign, source_hash, assets_hash):
    cfg = task_config(base, model, method, variant, TARGET, 42, campaign, source_hash, assets_hash)
    cfg['unlearn']['batch_sizes']['retain'] = 0
    cfg['conrep'].update(specified_cl_weight=0., specified_lm_weight=0.)
    cfg['baseline']['specified_weight'] = 0.
    cfg['data'].pop('retain_requests', None)
    cfg['data'].pop('retain_csv', None)
    cfg['selective']['training']['retain_policy'] = POLICY
    cfg['selective']['ablation'] = dict(control='T0', changed='specified retain removed from every training path',
        matched='SFT/F bytes/F draws/G bytes/G draws/optimizer/schedule', compute_matched=False,
        historical_variant=variant)
    calibration = sha(dict(training=cfg['selective']['training'], model=model, assets=assets_hash,
                           source=source_hash, method=method,
                           options={k: v for k, v in cfg['baseline'].items()
                                    if k not in ('layer_selection', 'augmented_file')}))
    cfg['selective']['calibration_identity'] = calibration
    if method == 'falcon':
        cfg['baseline']['layer_selection'] = str(campaign/'calibration'/calibration/'layers.json')
    if method == 'relearn':
        cfg['baseline']['augmented_file'] = str(campaign/'calibration'/calibration/'augmentation.jsonl')
    general_only(cfg)
    if cfg['unlearn']['max_steps'] != 125 or cfg['unlearn']['batch_sizes']['forget'] != 8:
        raise ValueError('General-only must retain the T0 exposure budget')
    return cfg


def prepare(campaign):
    campaign = Path(campaign).resolve()
    meta = read(campaign/'PREPARING.json')
    parent = Path(meta['parent'])
    parent_plan = read(parent/'plan.json')
    # Parent science/configs are verified before deriving anything, never rewritten.
    campaign_api.verify(parent, parent_plan)
    if (campaign/'plan.json').exists():
        campaign_api.verify(campaign, read(campaign/'plan.json'))
        print('General-only campaign already prepared; existing work preserved')
        return
    with locked(campaign/'build.lock'):
        if (campaign/'plan.json').exists():
            campaign_api.verify(campaign, read(campaign/'plan.json'))
            return
        files = read(campaign/'code-files.json')
        for name, expected in files.items():
            if file_sha(campaign/'code'/name) != expected:
                raise ValueError('Changed frozen code: '+name)
        make_data(parent, campaign)
        source_hash = sha(files)
        assets = read(parent/'model-assets.json')
        write(campaign/'source.json', dict(files=files, source_hash=source_hash))
        write(campaign/'model-assets.json', assets)
        plan = {key: copy.deepcopy(parent_plan[key]) for key in (
            'project_root', 'workers', 'world_size', 'queue', 'account', 'rtype', 'walltime',
            'poll_seconds', 'reserve_bytes', 'initial_task_seconds', 'recovery_policy')}
        plan.update(schema='0390-selective-v1', campaign=str(campaign), source_hash=source_hash,
                    python=sys.executable, entry=str(campaign/'code'/ENTRY), shell=str(campaign/'code'/SHELL),
                    environment=environment(), tasks=[], peers=list(parent_plan.get('peers', [])),
                    grid_phase=3, fixed_first=True, companion_parent=str(parent),
                    allocation_policy='one next-free slot alongside parent; up to three once parent completes',
                    budget=dict(fixed_trainings=14, later_grid=0, reused_references=2))
        plan['peers'] = [p for p in plan['peers'] if Path(p).resolve() != campaign]
        inputs = {str(p): file_sha(p) for p in (campaign/'data').rglob('*') if p.is_file()}
        # Inherit the immutable input audit (including MMLU/refusal data), not parent results.
        inputs.update(read(parent/'inputs.json'))
        ledger = []
        for control in selected_controls(parent_plan):
            base = read(control['config'])
            cfg = ablate_config(base, control['model'], control['method'], control['variant'],
                                campaign, source_hash, base['night']['model_assets_hash'])
            task_id = f"{control['model']}-{TARGET}-{control['variant']}-s42"
            task = publish_task(plan, cfg, task_id, 10)
            plan['tasks'].append(task)
            inputs[control['config']] = control['config_hash']
            ledger.append(dict(task=task_id, control=control['id'], config_sha256=control['config_hash'],
                               action='train_new', changed='remove all specified retain access',
                               compute_matched=False, training_identity=task['identity']))
        references = [task for task in parent_plan['tasks'] if task['kind'] == 'reference']
        if {t['model'] for t in references} != set(CANDIDATES) or len(references) != 2:
            raise ValueError('Need the two original shared SFT references')
        for reference in references:
            inputs[reference['config']] = reference['config_hash']
        write(campaign/'reference-sources.json', references)
        inputs[str(campaign/'reference-sources.json')] = file_sha(campaign/'reference-sources.json')
        write(campaign/'inputs.json', inputs)
        write(campaign/'reuse-ledger.json', dict(tasks=ledger, references='import completed parent outputs, no SFT retraining'))
        states = {task['id']: dict(status='pending', attempts=0, failures=0) for task in plan['tasks']}
        states.update({t['id']: dict(status='pending', external_reference=True) for t in references})
        write(campaign/'state.json', dict(status='prepared', started_at=None, deadline=None, tasks=states,
              workers={str(i): dict(job_id=None, status='new' if i == 0 else 'deferred_to_parent',
                                    allocations=0, recovery_failures=0) for i in range(3)}))
        write(campaign/'plan.json', plan)
    campaign_api.verify(campaign, plan)
    print(json.dumps(dict(status='prepared; no jobs submitted', trainings=14, target=TARGET,
                          steps=125, seed=42, specified_training_rows=0, compute_matched=False)))


def import_references(campaign, plan):
    """Reuse complete immutable parent references; never race their evaluator."""
    receipt = campaign/'references-import.json'
    if receipt.exists():
        for name, expected in read(receipt)['files'].items():
            if file_sha(name) != expected:
                raise ValueError('Imported reference changed: '+name)
        return True
    from .selective_eval import valid
    sources = read(campaign/'reference-sources.json')
    parent = Path(plan['companion_parent'])
    parent_state = read(parent/'state.json')
    if any(parent_state['tasks'][task['id']]['status'] != 'completed' for task in sources):
        return False
    files = {}
    for task in sources:
        cfg = read(task['config'])
        for partition in ('validation', 'test'):
            source = Path(cfg['selective']['reference_dir'])/partition
            if not valid(source, cfg, cfg['unlearn']['checkpoint'], partition, True):
                raise ValueError('Completed parent reference has invalid outputs: '+str(source))
            dest = campaign/'references'/task['model']/partition
            dest.mkdir(parents=True, exist_ok=True)
            for name in ('metrics.json', 'predictions.jsonl', 'mmlu_predictions.jsonl', 'EVAL_ID.json', 'VALIDATED.json'):
                if (source/name).exists():
                    shutil.copyfile(source/name, dest/name)
                    files[str(dest/name)] = file_sha(dest/name)
            if not valid(dest, cfg, cfg['unlearn']['checkpoint'], partition, True):
                raise ValueError('Copied reference did not preserve its identity')
    # State first is harmless if interrupted: no worker is submitted until receipt exists.
    with night.state_transaction(campaign) as state:
        for task in sources:
            state['tasks'][task['id']].update(status='completed', reused_from=task['output'])
    write(receipt, dict(parent=str(parent), files=files, time=time.time()))
    return True


def parent_complete(plan):
    parent = Path(plan['companion_parent'])
    return (parent/'COMPLETE.json').exists() and read(parent/'state.json')['status'] == 'completed'


def attach_reservation(campaign, plan):
    """Only the parent's operational peer list changes; no task/state/source edits."""
    parent = Path(plan['companion_parent'])
    with locked(parent/'start.lock'), locked(parent/'grid.lock'):
        current = read(parent/'plan.json')
        if Path(current['project_root']).resolve() != Path(plan['project_root']).resolve():
            raise ValueError('Parent project changed')
        if str(campaign) not in current.get('peers', []):
            current.setdefault('peers', []).append(str(campaign))
            write(parent/'plan.json', current)
            night.event(parent, event='general-only-companion-attached', companion=str(campaign), reserved_slots=1)
        receipt = campaign/'peer-attachment.json'
        if not receipt.exists():
            heartbeat = read(parent/'heartbeat.json') if (parent/'heartbeat.json').exists() else {}
            write(receipt, dict(parent=str(parent), initial=[heartbeat.get('pid'), heartbeat.get('time')],
                                observations=[], acknowledged=False))


def reservation_acknowledged(campaign, plan):
    """Do not submit while an old parent dispatch cycle may still hold its old plan."""
    path = campaign/'peer-attachment.json'
    receipt = read(path)
    if receipt['acknowledged']:
        return True
    parent = Path(plan['companion_parent'])
    if parent_complete(plan):
        receipt['acknowledged'] = True
    else:
        try:
            with locked(parent/'supervisor.lock', blocking=False):
                # Any future supervisor reads the amended plan when it starts.
                receipt['acknowledged'] = True
        except BlockingIOError:
            heartbeat = read(parent/'heartbeat.json') if (parent/'heartbeat.json').exists() else {}
            stamp = [heartbeat.get('pid'), heartbeat.get('time')]
            if stamp[1] is not None and stamp != receipt['initial'] and stamp not in receipt['observations']:
                receipt['observations'].append(stamp)
            # One in-flight cycle may have read the old plan; two completed cycles
            # prove it has reloaded the new peer list before the child can qsub.
            receipt['acknowledged'] = len(receipt['observations']) >= 2
    write(path, receipt)
    return receipt['acknowledged']


def refresh_workers(campaign, plan):
    complete = parent_complete(plan)
    with night.state_transaction(campaign) as state:
        if complete:
            for worker in state['workers'].values():
                if worker['status'] == 'deferred_to_parent':
                    worker['status'] = 'new'
        # The old selective controller's single-slot peer contract reads this field.
        state['job_id'] = state['workers']['0'].get('job_id')
        state['allocation_limit'] = 3 if complete else 1
    return complete


def watch(campaign):
    with locked(campaign/'supervisor.lock', blocking=False):
        write(campaign/'supervisor.json', dict(pid=os.getpid(), started=time.time(), policy=SUPERVISOR_POLICY))
        while not (campaign/'STOP').exists():
            try:
                plan = read(campaign/'plan.json')
                state = read(campaign/'state.json')
                if state['status'] != 'running':
                    return
                active = night.active_jobs(night.qstat(), getpass.getuser())
                night.reconcile(campaign, plan, active)
                if not import_references(campaign, plan):
                    write(campaign/'heartbeat.json', dict(time=time.time(), pid=os.getpid(),
                          policy=SUPERVISOR_POLICY, waiting='parent SFT references; original jobs continue'))
                    time.sleep(30)
                    continue
                refresh_workers(campaign, plan)
                attach_reservation(campaign, plan)
                if not reservation_acknowledged(campaign, plan):
                    write(campaign/'heartbeat.json', dict(time=time.time(), pid=os.getpid(),
                          policy=SUPERVISOR_POLICY, waiting='parent supervisor to reload the reserved slot'))
                    time.sleep(30)
                    continue
                state = read(campaign/'state.json')
                if all(state['tasks'][t['id']]['status'] == 'completed' for t in plan['tasks']):
                    with night.state_transaction(campaign) as current:
                        current['status'] = 'completed'
                    write(campaign/'COMPLETE.json', dict(time=time.time(), trainings=14))
                    return
                owned = any(w.get('job_id') in active or w['status'] == 'submitting'
                            or w.get('job_id') and w.get('reconciled_job') != w['job_id']
                            for w in state['workers'].values())
                waiting = any(t['status'] == 'paused' and t.get('retry_after', 0) > time.time()
                              for t in state['tasks'].values())
                usable = [w for w in state['workers'].values() if w['status'] != 'deferred_to_parent']
                if not owned and not waiting and (not campaign_api.ready(plan, state)
                        or all(w['status'] in night.BLOCKED_WORKER_STATES for w in usable)):
                    with night.state_transaction(campaign) as current:
                        current['status'] = 'blocked'
                    write(campaign/'BLOCKED.json', dict(reason='General-only failed or requires explicit recovery'))
                    return
                campaign_api.submit_available(campaign, plan)
                refresh_workers(campaign, plan)
                write(campaign/'heartbeat.json', dict(time=time.time(), pid=os.getpid(), policy=SUPERVISOR_POLICY,
                      account_active=len(active), allocation_limit=read(campaign/'state.json')['allocation_limit'],
                      dispatch=campaign_api.describe(plan, read(campaign/'state.json'))))
            except Exception as exc:
                night.event(campaign, event='general-only-supervisor-error', error=repr(exc))
            time.sleep(30)


def main():
    campaign_api.free_slots = shared_free_slots
    if len(sys.argv) > 1 and sys.argv[1] in ('prepare', 'watch'):
        import argparse
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('command', choices=('prepare', 'watch'))
        parser.add_argument('--campaign', required=True, type=Path)
        args = parser.parse_args()
        (prepare if args.command == 'prepare' else watch)(args.campaign.resolve())
        return 0
    return campaign_api.main()
