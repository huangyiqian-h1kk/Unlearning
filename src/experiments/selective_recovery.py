"""Additive MCQ evaluation and bounded augmentation recovery on frozen science.

The original protocol identity remains the cache/legacy-worker compatibility key.
An additional execution identity binds the actual repair code to validated output.
Four-option MCQ, generation, SFT references and every training command stay native.
"""
import copy
import os
from pathlib import Path
import runpy
import sys
import time

from conrep.night.io import read, write, sha, file_sha, locked
from .selective_data import rows, write_rows

POLICY = 'selective-binary-mcq-recovery-v1'
PROMPTS = (
    'For patient {patient_id}, what {attribute} is documented in the clinical record?',
    'Please identify the recorded {attribute} associated with patient identifier {patient_id}.',
    'A record is indexed by patient ID {patient_id}. What is its {attribute} entry?',
    'Question about patient {patient_id}: what is the value of the {attribute} field?',
    'Consult the clinical case for patient {patient_id} and state the documented {attribute}.',
    'Which {attribute} value belongs to the medical record of patient {patient_id}?',
)


def probe_items(cfg, partition, reference=False):
    root = Path(cfg['selective']['task_path'])
    probe = root/('reference' if reference else 'probes')/partition
    items = []
    for row in rows(probe/'generation.jsonl'):
        for family, prompt in row['probe_prompts'].items():
            items.append(dict(row, id=row['fact_id']+':generation:'+family,
                              kind='generation', task=family, prompt=prompt))
    if not reference:
        items.extend(dict(r, kind='mcq') for r in rows(probe/'mcq.jsonl'))
        items.extend(dict(r, kind='safe_background', task='safe_background')
                     for r in rows(probe/'safe_background.jsonl'))
    result = {r['id']: r for r in items}
    if len(result) != len(items):
        raise ValueError('Duplicate probe identity')
    return result


def evaluation_wrapper(original_run, original_signature, scoring, meta):
    def run(cfg, checkpoint, output, partition='validation', reference=False):
        if reference:
            return original_run(cfg, checkpoint, output, partition, reference)
        from .selective_eval import load_partial
        output = Path(output)
        with locked(output/'evaluation-repair.lock'):
            identity = original_signature(cfg, checkpoint, partition, reference)
            marker = output/'EVAL_ID.json'
            predictions = output/'predictions.jsonl'
            if marker.exists() and read(marker)['identity'] != identity:
                raise ValueError('Foreign evaluation identity; prediction cache was not touched')
            if predictions.exists() and not marker.exists():
                raise ValueError('Prediction cache has no original evaluation identity')
            expected = probe_items(cfg, partition)
            for row in expected.values():
                if row['kind'] == 'mcq':
                    keys = set(row['mapping'])
                    if keys not in (set('AB'), set('ABCD')) or (keys == set('AB') and row['attribute'] != 'gender'):
                        raise ValueError('Unsupported MCQ schema')
            cached = load_partial(predictions)
            for key, record in cached.items():
                item = expected.get(key)
                if item is None or any(record.get(k) != v for k, v in item.items()):
                    raise ValueError('Prediction/probe metadata mismatch: '+key)
                if item['kind'] == 'mcq':
                    scored = scoring.score_mcq(record['output'], item['mapping'], item['attribute'])
                    scored['correct'] = scored['predicted'] == item['correct_letter']
                else:
                    scored = scoring.score_answer(record['output'], item['target'], item['patient_id'])
                if any(record.get(k) != v for k, v in scored.items()):
                    raise ValueError('Cached score differs under the audited parser: '+key)
            receipt = output/'EVALUATION_EXTENSION.json'
            extension = dict(policy=POLICY, ref=meta['ref'], files=meta['files'],
                base_protocol_hash=identity, binary_parser='direct-choice-binary-v1',
                four_choice_and_generation='Original frozen scorers, unchanged')
            execution = sha(extension)
            if receipt.exists():
                if read(receipt)['execution_identity'] != execution:
                    raise ValueError('Evaluation extension changed')
            else:
                write(receipt, dict(**extension, execution_identity=execution,
                    preserved_predictions=len(cached), prefix_sha256=file_sha(predictions) if predictions.exists() else None,
                    time=time.time()))
            # Insert provenance during the native metrics->VALIDATED write order,
            # avoiding a second post-validation rewrite window.
            from . import selective_eval
            native_write = selective_eval.write
            def audited_write(path,value):
                if Path(path) == output/'metrics.json':
                    value['evaluation_extension'] = dict(policy=POLICY, execution_identity=execution, ref=meta['ref'])
                elif Path(path) == output/'VALIDATED.json':
                    value.update(execution_identity=execution,extension_sha256=file_sha(receipt))
                return native_write(path,value)
            selective_eval.write = audited_write
            try:
                return original_run(cfg, checkpoint, output, partition, reference)
            finally:
                selective_eval.write = native_write
    return run


def extend_relearn(cfg, meta):
    """Continue only an untrained, incomplete augmentation with the original filters."""
    from .baselines.prepare import relearn_augment
    from .pmc_scoring_v5 import score_answer
    from .selective_calibrate import valid
    dest = Path(cfg['baseline']['augmented_file'])
    root = dest.parent
    with locked(root/'calibration.lock'):
        if valid(cfg):
            return
        if (root/'CALIBRATED.json').exists():
            raise ValueError('Completed calibration changed; no repair overwrite allowed')
        training = Path(cfg['run']['output_dir'])
        if (training/'TRAINING_COMPLETE.json').exists() or any(training.glob('checkpoint-*')):
            raise ValueError('Cannot change augmentation after training has started')
        facts = {r['id']: r for r in rows(Path(cfg['data']['prepared_dir'])/'forget.jsonl')}
        candidates = rows(dest) if dest.exists() else []
        def accepted_rows(data):
            result = {}
            for row in data:
                if row['fact_id'] not in facts:
                    raise ValueError('Foreign augmentation fact')
                if score_answer(row['answer'], facts[row['fact_id']]['target'])['correct']:
                    raise ValueError('Saved augmentation reveals the target answer')
                result[(row['fact_id'], row['prompt'], row['answer'])] = row
            return list(result.values())
        candidates = accepted_rows(candidates)
        covered = {r['fact_id'] for r in candidates}
        backup = root/'recovery-v1-original-augmentation.jsonl'
        if not backup.exists():
            write_rows(backup, candidates)
        audit = []
        for index, template in enumerate(PROMPTS, 3):
            if covered == set(facts):
                break
            prefix = root/f'recovery-v1-round-{index}'
            request_file = Path(str(prefix)+'-requests.jsonl')
            output = Path(str(prefix)+'.jsonl')
            provenance = Path(str(prefix)+'.provenance.json')
            if not request_file.exists():
                write_rows(request_file, [dict(fact_id=key, kind='qa', prompt=template.format(**facts[key]),
                           answer=facts[key]['target']['answer']) for key in sorted(set(facts)-covered)])
            requests = rows(request_file)
            if any(r['fact_id'] not in facts or r['answer'] != facts[r['fact_id']]['target']['answer']
                   or r['prompt'] != template.format(**facts[r['fact_id']]) for r in requests):
                raise ValueError('Recovery request changed')
            expected = dict(policy=POLICY, ref=meta['ref'], files=meta['files'],
                calibration_identity=cfg['selective']['calibration_identity'], requests_sha256=file_sha(request_file))
            ready = (output.exists() and Path(str(output)+'.audit.json').exists()
                     and provenance.exists() and read(provenance).get('status') == 'complete')
            if ready:
                if not provenance.exists():
                    raise ValueError('Recovery augmentation has no provenance')
                previous = read(provenance)
                if (any(previous.get(k) != v for k,v in expected.items())
                        or previous['output_sha256'] != file_sha(output)
                        or previous['audit_sha256'] != file_sha(Path(str(output)+'.audit.json'))):
                    raise ValueError('Recovery augmentation changed')
            else:
                if provenance.exists() and any(read(provenance).get(k) != v for k,v in expected.items()):
                    raise ValueError('Recovery augmentation intent changed')
                write(provenance,dict(**expected,status='preparing'))
                temporary = copy.deepcopy(cfg)
                temporary['data']['forget_requests'] = str(request_file)
                relearn_augment(temporary, output)
                write(provenance, dict(**expected, status='complete', output_sha256=file_sha(output),
                                      audit_sha256=file_sha(Path(str(output)+'.audit.json'))))
            new = [r for r in rows(output) if not score_answer(r['answer'],facts[r['fact_id']]['target'])['correct']]
            candidates = accepted_rows(candidates+new)
            covered = {r['fact_id'] for r in candidates}
            audit.append(dict(round=index, covered=len(covered), required=len(facts), requested=len(requests)))
            write(root/'recovery-v1-coverage.json', dict(policy=POLICY, ref=meta['ref'], rounds=audit,
                covered=len(covered), required=len(facts), missing=sorted(set(facts)-covered), accepted=len(candidates)))
        # The original three rounds remain intact. No fabricated refusal, weaker
        # disclosure filter, foreign teacher, or reduced forget set is permitted.
        if covered != set(facts):
            raise ValueError(f'blocked_augmentation_recovery_exhausted: {len(facts)-len(covered)} uncovered facts')
        write_rows(dest, [dict(r,id=f'relearn:{i}') for i,r in enumerate(candidates)])
        write(root/'CALIBRATED.json', dict(identity=cfg['selective']['calibration_identity'], sha256=file_sha(dest),
            preparation_extension=dict(policy=POLICY, ref=meta['ref'], files=meta['files'],
                additional_rounds=len(audit), original_sha256=file_sha(backup),
                coverage_sha256=file_sha(root/'recovery-v1-coverage.json'))))


def requeue_supported(campaign, plan, meta):
    """One evidence-backed retry per task; no unrelated failure is reset."""
    from conrep.night import campaign as night
    from .selective_campaign import trained
    campaign = Path(campaign)
    state = read(campaign/'state.json')
    selected = []
    for task in plan['tasks']:
        record = state['tasks'][task['id']]
        if record['status'] != 'failed' or record.get('evaluation_repair_retry') == POLICY:
            continue
        cfg = read(task['config'])
        output = Path(task['output'])
        reason = None
        if 'Validation incomplete' in record.get('error', '') or 'Evaluation failed' in record.get('error', ''):
            if task['kind'] == 'train' and not trained(task, cfg):
                continue
            for log in output.glob('*/*/console.log'):
                with log.open('rb') as stream:
                    stream.seek(max(0,log.stat().st_size-65536))
                    text = stream.read().decode('utf-8',errors='replace')
                if 'ValueError: Four nonempty options A/B/C/D required' in text:
                    reason = 'binary_mcq_parser'; break
        elif task['method'] == 'relearn' and 'Calibration failed' in record.get('error', ''):
            root = Path(cfg['baseline']['augmented_file']).parent
            if (root/'coverage.json').is_file() and not (root/'CALIBRATED.json').exists():
                if read(root/'coverage.json').get('missing'):
                    reason = 'bounded_missing_fact_augmentation'
        if reason:
            selected.append((task['id'], reason))
    with night.state_transaction(campaign) as current:
        for task_id, reason in selected:
            record = current['tasks'][task_id]
            if record['status'] != 'failed' or record.get('evaluation_repair_retry') == POLICY:
                continue
            record.setdefault('repair_history', []).append(dict(time=time.time(), policy=POLICY,
                ref=meta['ref'], reason=reason, previous_error=record.get('error')))
            record.update(status='pending', error=None, evaluation_repair_retry=POLICY)
        if selected and current['status'] == 'blocked':
            current['status'] = 'running'
    if selected:
        night.event(campaign, event='evaluation-repair-requeue', tasks=selected, policy=POLICY)
    return selected


def export_wrapper(original_export):
    """Include the execution receipt and frozen repair in the native summary ZIP."""
    def export(campaign, predictions=False):
        import zipfile
        original_zip = zipfile.ZipFile
        campaign = Path(campaign)
        class AuditedZip(original_zip):
            def close(self):
                if (self.fp and self.mode == 'w' and str(self.filename).endswith('-summary.zip')
                        and not getattr(self, '_repair_added', False)):
                    self._repair_added = True
                    plan = read(campaign/'plan.json')
                    manifest = Path(plan['evaluation_repair']['manifest'])
                    for name in ('manifest.json', *read(manifest)['files']):
                        path = manifest.parent/name
                        self.write(path, str(path.relative_to(campaign)))
                    for task in plan['tasks']:
                        for receipt in Path(task['output']).glob('*/*/EVALUATION_EXTENSION.json'):
                            self.write(receipt, str(receipt.relative_to(campaign)))
                            marker = receipt.parent/'VALIDATED.json'
                            if marker.exists():
                                self.write(marker, str(marker.relative_to(campaign)))
                super().close()
        zipfile.ZipFile = AuditedZip
        try:
            return original_export(campaign, predictions)
        finally:
            zipfile.ZipFile = original_zip
    return export


def install(meta, command):
    from . import pmc_scoring_v5, selective_eval, selective_calibrate
    from .selective_mcq import extract_binary
    original_extract = pmc_scoring_v5.extract
    def extract(output, options):
        if set(options) == set('AB'):
            return extract_binary(output, options)
        return original_extract(output, options)
    pmc_scoring_v5.extract = extract
    selective_eval.run = evaluation_wrapper(selective_eval.run, selective_eval.signature, pmc_scoring_v5, meta)
    original_calibrate = selective_calibrate.run
    def calibrate(cfg):
        try:
            return original_calibrate(cfg)
        except ValueError as exc:
            if cfg['baseline']['method'] != 'relearn' or not str(exc).startswith('blocked_augmentation:'):
                raise
            return extend_relearn(cfg, meta)
    selective_calibrate.run = calibrate
    if command == 'export':
        from . import selective_export
        selective_export.export = export_wrapper(selective_export.export)
    if command in ('watch', 'arm', 'resume'):
        from conrep.night import campaign as night
        original_reconcile = night.reconcile
        def reconcile(campaign, plan, active, **kwargs):
            result = original_reconcile(campaign,plan,active,**kwargs)
            requeue_supported(campaign,plan,meta)
            return result
        night.reconcile = reconcile
    if command == 'watch':
        write(Path(meta['campaign'])/'evaluation-repair-supervisor.json',dict(pid=os.getpid(),policy=POLICY,ref=meta['ref']))


def dispatch(meta, args):
    stripped = list(args)
    while stripped and stripped[0] in ('-I','-u'):
        stripped.pop(0)
    allowed = {str(Path(meta['original_entry']).resolve())}
    plan = read(Path(meta['campaign'])/'plan.json')
    if plan.get('temporary_extra_node'):
        allowed.add(str(Path(plan['temporary_extra_node']['entry']).resolve()))
    if len(stripped) >= 2 and str(Path(stripped[0]).resolve()) in allowed:
        command = stripped[1]
        if command in ('evaluate','calibrate','watch','arm','start','resume','status','export','worker'):
            install(meta,command)
            sys.argv = stripped
            runpy.run_path(stripped[0],run_name='__main__')
            return
    # In particular, torchrun and its native scientific trainer are untouched.
    os.execv(meta['real_python'],[meta['real_python'],*args])
