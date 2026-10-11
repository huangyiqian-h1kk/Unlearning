#!/usr/bin/env python3
"""Audited, restartable reader-only repair of a blocked stage-zero campaign.

No PBS submission, state reset, input edits, or evaluation-cache edits occur here.
The frozen reader is changed to the existing MMLU reader; all source/config hashes
are updated before the separate explicit `resume --retry-failed` command.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

REL = 'src/experiments/selective_data.py'
REPAIR = 'mmlu-reader-v1'
OLD = '''def rows(path):
    import json
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
'''
NEW = '''def rows(path):
    # Use the evaluator's reader: str.splitlines also splits U+0085/U+2028/U+2029
    # inside otherwise valid JSON strings (e.g. MMLU question text).
    from .config import read_rows
    return read_rows(path)
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def corrected_source(before, after):
    text = before.decode()
    if text.count(OLD) != 1 or text.replace(OLD, NEW, 1).encode() != after:
        raise ValueError('Ref must contain exactly the reviewed rows() reader fix')


def check_mmlu(configs, reader):
    """Use the same reader as actual MMLU scoring; never rewrite input data."""
    cache = {}
    for cfg in configs:
        options = cfg['evaluation']
        paths = [options.get(k) for k in ('mmlu_file', 'mmlu_test_file')]
        if not all(paths):
            raise ValueError('Both validation and test MMLU paths are required')
        for name in paths:
            if name in cache:
                continue
            items = reader(name)
            if not items or any(not isinstance(r.get('prompt'), str)
                                or not r.get('subject') or r.get('answer') not in tuple('ABCD') for r in items):
                raise ValueError('Invalid MMLU records: ' + name)
            keys = {(r['subject'], r['prompt']) for r in items}
            cache[name] = keys
            text = Path(name).read_text(encoding='utf-8-sig')
            print(json.dumps(dict(mmlu=name, rows=len(items),
                unicode_separators={f'U+{ord(c):04X}': text.count(c) for c in '\x85\u2028\u2029'})), flush=True)
        if cache[paths[0]] & cache[paths[1]]:
            raise ValueError('MMLU test overlaps validation')


def guard_idle(plan, state, active):
    if state['status'] != 'blocked' or plan.get('grid_phase') != 0:
        raise ValueError('This repair is only for a blocked stage-zero campaign')
    for task in plan['tasks']:
        rec = state['tasks'][task['id']]
        if task['kind'] == 'reference':
            if rec['status'] not in ('failed', 'completed'):
                raise ValueError('Reference is not terminal: ' + task['id'])
        elif rec['status'] != 'pending' or rec.get('attempts', 0):
            raise ValueError('Non-reference work has already started: ' + task['id'])
    for worker in state['workers'].values():
        if worker['status'] not in ('new', 'available') or worker.get('job_id') in active:
            raise ValueError('Selective allocation still active or unreconciled')


def stage(campaign, folder, changes, details):
    """Persist both sides before replacing any campaign file."""
    entries = []
    for relative, after in sorted(changes.items()):
        before = (campaign / relative).read_bytes()
        for side, data in [('before', before), ('after', after)]:
            path = folder / side / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        entries.append(dict(path=relative, before=digest(before), after=digest(after)))
    from conrep.night.io import write
    write(folder / 'transaction.json', dict(**details, files=entries))


def apply_transaction(campaign, folder):
    """An interrupted apply can be completed, but foreign edits are rejected."""
    from conrep.night.io import read
    manifest = read(folder / 'transaction.json')
    for row in manifest['files']:
        for side in ('before', 'after'):
            if digest((folder / side / row['path']).read_bytes()) != row[side]:
                raise ValueError('Repair backup changed: ' + row['path'])
        if digest((campaign / row['path']).read_bytes()) not in (row['before'], row['after']):
            raise ValueError('Foreign edit during repair: ' + row['path'])
    for row in manifest['files']:
        target = campaign / row['path']
        temporary = target.with_name(target.name + '.reader-repair.tmp')
        temporary.write_bytes((folder / 'after' / row['path']).read_bytes())
        temporary.replace(target)


def repair(root, campaign, ref):
    root, campaign = Path(root).resolve(), Path(campaign).resolve()
    sys.path.insert(0, str(campaign / 'code/src'))
    from conrep.night.io import read, write, sha, file_sha, locked
    from conrep.night import campaign as night
    from experiments.selective_campaign import verify
    from experiments.selective_common import training_identity
    from experiments.config import read_rows
    folder = campaign / 'repairs' / REPAIR
    with ExitStack() as locks:
        for name in ('start.lock', 'supervisor.lock', 'state.lock'):
            locks.enter_context(locked(campaign / name, blocking=False))
        plan = read(campaign / 'plan.json')
        if Path(plan['project_root']).resolve() != root:
            raise ValueError('Project root mismatch')
        if (folder / 'COMMITTED.json').exists():
            verify(campaign, plan)
            receipt = read(folder / 'COMMITTED.json')
            if file_sha(campaign / 'code' / REL) != receipt['reader_sha256']:
                raise ValueError('Committed repair reader changed')
            print('Reader repair already committed; existing state preserved.', flush=True)
            return
        active = night.active_jobs(night.qstat(), __import__('getpass').getuser())
        guard_idle(plan, read(campaign / 'state.json'), active)
        if not (folder / 'transaction.json').exists():
            verify(campaign, plan)
            commit = subprocess.check_output(['git', 'rev-parse', '--verify', ref+'^{commit}'], cwd=root, text=True).strip()
            after = subprocess.check_output(['git', 'show', f'{commit}:{REL}'], cwd=root)
            corrected_source((campaign / 'code' / REL).read_bytes(), after)
            configs = {t['id']: read(t['config']) for t in plan['tasks']}
            check_mmlu(configs.values(), read_rows)
            source = read(campaign / 'source.json')
            files = read(campaign / 'code-files.json')
            if files != source['files'] or plan['source_hash'] != source['source_hash'] or sha(files) != source['source_hash']:
                raise ValueError('Source manifest mismatch')
            # Check the original evaluation signatures before the sole reader edit.
            from experiments.selective_eval import signature, valid
            caches = []
            for task in plan['tasks']:
                if task['kind'] != 'reference':
                    continue
                cfg = configs[task['id']]
                for partition in ('validation', 'test'):
                    out = Path(cfg['selective']['reference_dir']) / partition
                    identity = signature(cfg, cfg['unlearn']['checkpoint'], partition, True)
                    if (out / 'EVAL_ID.json').exists() and read(out / 'EVAL_ID.json')['identity'] != identity:
                        raise ValueError('Foreign reference cache: ' + str(out))
                    if (out / 'VALIDATED.json').exists() and not valid(out, cfg, cfg['unlearn']['checkpoint'], partition, True):
                        raise ValueError('Invalid completed reference: ' + str(out))
                    if (out / 'predictions.jsonl').exists():
                        if not (out / 'EVAL_ID.json').exists():
                            raise ValueError('Reference predictions have no identity: ' + str(out))
                        predictions = read_rows(out / 'predictions.jsonl')
                        ids = [r['id'] for r in predictions]
                        probes = read_rows(Path(cfg['selective']['task_path']) / 'reference' / partition / 'generation.jsonl')
                        expected = {r['fact_id']+':generation:'+family for r in probes for family in r['probe_prompts']}
                        if len(ids) != len(set(ids)) or set(ids) - expected:
                            raise ValueError('Foreign/duplicate reference predictions: ' + str(out))
                        caches.append(dict(output=str(out), count=len(ids), expected=len(expected),
                            sha256=file_sha(out / 'predictions.jsonl'), identity=identity))
            old_hash = source['source_hash']
            files[REL] = digest(after)
            new_hash = sha(files)
            note = dict(repair=REPAIR, ref=commit, previous_source_hash=old_hash, source_hash=new_hash)
            source.update(files=files, source_hash=new_hash)
            source.setdefault('repairs', []).append(note)
            plan.update(source_hash=new_hash)
            plan.setdefault('repairs', []).append(note)
            changes = {'code/'+REL: after, 'code-files.json': encode(files), 'source.json': encode(source)}
            for task in plan['tasks']:
                cfg = configs[task['id']]
                cfg['night']['source_hash'] = new_hash
                relative = str(Path(task['config']).resolve().relative_to(campaign))
                changes[relative] = encode(cfg)
                task.update(config_hash=digest(changes[relative]), identity=training_identity(cfg))
                if task['kind'] == 'reference':
                    for cache in caches:
                        out = Path(cache['output'])
                        if out.parent == Path(cfg['selective']['reference_dir']):
                            if signature(cfg, cfg['unlearn']['checkpoint'], out.name, True) != cache['identity']:
                                raise ValueError('Reader-only repair unexpectedly changed evaluation identity')
            ledger = read(campaign / 'reuse-ledger.json')
            identities = {t['id']: t['identity'] for t in plan['tasks']}
            for item in ledger['tasks']:
                if 'training_identity' in item:
                    item['training_identity'] = identities[item['task']]
            changes.update({'plan.json': encode(plan), 'reuse-ledger.json': encode(ledger)})
            stage(campaign, folder, changes, dict(**note, time=time.time(), preserved_caches=caches))
        apply_transaction(campaign, folder)
        plan = read(campaign / 'plan.json')
        verify(campaign, plan)
        receipt = read(folder / 'transaction.json')
        for cache in receipt['preserved_caches']:
            if file_sha(Path(cache['output']) / 'predictions.jsonl') != cache['sha256']:
                raise ValueError('Reference predictions changed during repair')
        write(folder / 'COMMITTED.json', dict(time=time.time(), source_hash=plan['source_hash'],
            reader_sha256=file_sha(campaign / 'code' / REL), ref=receipt['ref'],
            preserved_caches=receipt['preserved_caches']))
        print(json.dumps(dict(status='reader repaired; no jobs submitted',
            preserved_caches=receipt['preserved_caches'], backup=str(folder / 'before')), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'campaign', 'ref'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    repair(args.root, args.campaign, args.ref)
