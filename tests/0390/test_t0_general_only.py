"""Real tiny-model updates plus data-access and companion scheduling regressions."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from conrep.night.io import read, write, file_sha
from experiments import selective_campaign as c
from experiments import t0_general_only as supplement
from experiments.retain_policy import POLICY, active_groups, general_only, sampling_seed
from experiments.selective_common import BASELINES, complete_checkpoint, training_identity
from experiments.selective_data import rows, write_rows
from experiments.selective_train import run

spec = importlib.util.spec_from_file_location('general_test_helpers', Path(__file__).with_name('test_core.py'))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
torch.set_num_threads(1)


def fixture(root, method, family='llama'):
    cfg, tok = helpers.tiny_setup(root)
    if family == 'gemma':
        from transformers import Gemma2Config, Gemma2ForCausalLM
        torch.manual_seed(19)
        model = Gemma2ForCausalLM(Gemma2Config(vocab_size=len(tok), hidden_size=16, intermediate_size=32,
            num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=256, sliding_window=64, pad_token_id=tok.pad_token_id))
        model.save_pretrained(cfg['unlearn']['checkpoint'])
    cfg['baseline'].update(method=method, specified_weight=0., layer=1, train_layers=[0, 1],
        fit_learning_rate=.001, scheduler_gamma=.9, scheduler_every_steps=10,
        direction_coefficient=1., calibration_samples=2)
    cfg['unlearn']['batch_sizes'] = dict(forget=8, retain=0, general=8)
    cfg['conrep'].update(specified_cl_weight=0., specified_lm_weight=0., forget_cl_weight=2., negative_views=2)
    cfg['selective'] = dict(training={'retain_policy': POLICY})
    cfg['night'] = dict(source_hash='test', model_assets_hash='test')
    cfg['diagnostics'] = dict(sampling_coverage=True)
    cfg['data'].pop('retain_requests', None)
    root_data = Path(cfg['data']['prepared_dir'])
    (root_data/'retain.jsonl').unlink()
    for group in ('forget', 'general'):
        original = rows(root_data/(group+'.jsonl'))
        write_rows(root_data/(group+'.jsonl'), [dict(original[i % 4], id=f'{group}:{i}') for i in range(8)])
    if method == 'falcon':
        path = root/'layers.json'
        write(path, dict(checkpoint=cfg['unlearn']['checkpoint'], selected_layer=1))
        cfg['baseline'].update(layer_selection=str(path), optimizer='sophia', temperature=.7,
                               conflict_weights=[.8, 1.2], align_weights=[.1, 1.9])
    if method == 'relearn':
        path = root/'aug.jsonl'
        write_rows(path, [dict(id=f'a{i}', fact_id=f'forget:{i}', kind='qa',
                               prompt=f'What is condition of P{i} ?', answer='general medical knowledge') for i in range(8)])
        cfg['baseline']['augmented_file'] = str(path)
    if method == 'lunar':
        path = root/'forget-requests.jsonl'
        write_rows(path, [dict(fact_id=f'forget:{i}', prompt=f'What is condition of P{i} ?') for i in range(8)])
        cfg['data']['forget_requests'] = str(path)
        path = root/'refusal.jsonl'
        write_rows(path, [dict(prompt='general medical knowledge')])
        cfg['baseline']['refusal_file'] = str(path)
    return cfg, tok


@pytest.mark.parametrize('family', ['llama', 'gemma'])
@pytest.mark.parametrize('method', ['conrep', *BASELINES])
def test_no_specified_file_real_updates_resume_exact(tmp_path, family, method):
    cfg, _ = fixture(tmp_path, method, family)
    cfg['run']['output_dir'] = str(tmp_path/'full')
    assert run(cfg) == 0
    cfg['run']['output_dir'] = str(tmp_path/'resumed')
    assert run(cfg, stop_after_step=1) == 75
    assert run(cfg, tmp_path/'resumed/checkpoint-1') == 0
    name = 'adapter_model.safetensors' if method in ('conrep', 'npo', 'sago', 'relearn') else 'delta.safetensors'
    full = load_file(str(tmp_path/'full/checkpoint-2'/name))
    resumed = load_file(str(tmp_path/'resumed/checkpoint-2'/name))
    assert full.keys() == resumed.keys()
    assert all(torch.equal(full[k], resumed[k]) for k in full)
    assert complete_checkpoint(tmp_path/'resumed/checkpoint-2', identity=training_identity(cfg), world=1, resume=True)
    for checkpoint in ('full', 'resumed'):
        counts = read(tmp_path/checkpoint/'checkpoint-2/sampling_state.json')['counts']
        assert set(counts) == {'forget', 'general'}
        assert sum(counts['forget']) == sum(counts['general']) == 16
        for record in rows(tmp_path/checkpoint/'train.jsonl'):
            assert record['specified_retain_draws'] == 0
            assert set(record['canonical_input_tokens']) == {'forget', 'general'}


def test_conrep_negative_bank_is_only_general(tmp_path, monkeypatch):
    from conrep.night import trainer
    cfg, _ = fixture(tmp_path, 'conrep')
    original = trainer.forget_loss
    sizes = []
    def observed(forget, controls, retain, **kwargs):
        sizes.append(len(retain))
        return original(forget, controls, retain, **kwargs)
    monkeypatch.setattr(trainer, 'forget_loss', observed)
    assert run(cfg) == 0
    assert sizes == [8, 8]


def test_falcon_calibration_does_not_read_specified(tmp_path, monkeypatch):
    from experiments.baselines import prepare
    cfg, _ = fixture(tmp_path, 'falcon')
    class Analyzer:
        def calculate_mutual_information(self, f, r):
            assert len(f) == len(r) == 2
            return .5, 1., 1., 1.
    monkeypatch.setattr(prepare, 'vendor_module', lambda *a: SimpleNamespace(UnifiedInformationAnalyzer=Analyzer))
    output = tmp_path/'calibration.json'
    report = prepare.falcon_layers(cfg, output)
    assert report['selected_layer'] is not None
    assert all('retain' not in r and 'general' in r for r in report['layers'].values())


def test_original_f_g_sampling_streams_and_policy_guards(tmp_path):
    from experiments.data import sample_batch
    cfg, _ = fixture(tmp_path, 'conrep')
    assert active_groups(cfg) == ('forget', 'general')
    for group, offset in [('forget', 0), ('general', 2)]:
        data = rows(Path(cfg['data']['prepared_dir'])/(group+'.jsonl'))
        assert sample_batch(data, 8, sampling_seed(42, group), 3, 0, 1) == sample_batch(data, 8, 42+offset*10007, 3, 0, 1)
    cfg['conrep']['specified_cl_weight'] = 1.
    with pytest.raises(ValueError, match='zero specified'):
        general_only(cfg)


def parent_fixture(root):
    parent, child = root/'parent', root/'child'
    source = parent/'data/T0-C0'
    source.mkdir(parents=True)
    facts = [dict(id=f'forget:{i}', relation_id=f'p{i}:age', text=f'patient p{i} is 20',
                  patient_id=f'p{i}', attribute='age', target={'answer': '20'}) for i in range(100)]
    write_rows(source/'forget.jsonl', facts)
    write_rows(source/'general.jsonl', [dict(id='wiki', text='general text')])
    write_rows(source/'forget_requests.jsonl', [dict(fact_id=r['relation_id'], prompt='age?') for r in facts])
    for part in ('validation', 'test'):
        write_rows(source/'probes'/part/'generation.jsonl', [dict(fact_id='p0:age', partition=part)])
    write(source/'task.json', dict(id='T0-C0', counts=dict(forget=100, retain_same_eval=200,
          retain_other_eval=900, max_steps=125), membership=dict(forget=[r['relation_id'] for r in facts],
          retain_train=['old'], retain_same_eval=['p0:gender'], retain_other_eval=['other:age']),
          request_seed=None, train_hashes={}))
    return parent, child


def test_data_preserves_f_bytes_and_probes_maps_legacy_ids(tmp_path):
    parent, child = parent_fixture(tmp_path)
    supplement.make_data(parent, child)
    source, target = parent/'data/T0-C0', child/'data'/supplement.TARGET
    assert file_sha(source/'forget.jsonl') == file_sha(target/'forget.jsonl')
    assert file_sha(source/'general.jsonl') == file_sha(target/'general.jsonl')
    assert not (target/'retain.jsonl').exists()
    assert not (target/'retain_requests.jsonl').exists()
    assert read(target/'task.json')['membership']['retain_train'] == []
    assert [r['fact_id'] for r in rows(target/'forget_requests.jsonl')] == [f'forget:{i}' for i in range(100)]
    for p in (source/'probes').rglob('*.jsonl'):
        assert file_sha(p) == file_sha(target/p.relative_to(source))
    supplement.make_data(parent, child)


def scheduling_fixture(root):
    parent, child = root/'parent', root/'child'
    workers = {str(i): dict(status='new', job_id=None, allocations=0, recovery_failures=0) for i in range(3)}
    write(parent/'state.json', dict(status='running', workers=copy.deepcopy(workers)))
    parent_plan = dict(project_root=str(root), peers=[], tasks=[{'id': 'old'}], source_hash='unchanged')
    write(parent/'plan.json', parent_plan)
    plan = dict(project_root=str(root), campaign=str(child), companion_parent=str(parent),
                peers=[], shell='worker.sh', walltime='12:00:00', tasks=[])
    state = dict(status='running', workers=workers, tasks={})
    for i in (1, 2):
        state['workers'][str(i)]['status'] = 'deferred_to_parent'
    for i in range(14):
        task = dict(id=f't{i}', kind='train', model='gemma', stage=10)
        plan['tasks'].append(task)
        state['tasks'][task['id']] = dict(status='pending', attempts=0)
    state['tasks']['gemma-sft-reference'] = dict(status='completed')
    write(child/'state.json', state)
    return parent, child, plan, state


def test_live_attachment_and_one_then_three_cap(tmp_path, monkeypatch):
    parent, child, plan, state = scheduling_fixture(tmp_path)
    before = read(parent/'plan.json')
    supplement.attach_reservation(child, plan)
    supplement.attach_reservation(child, plan)
    after = read(parent/'plan.json')
    assert after == dict(before, peers=[str(child)])
    active = {'parent0': {'Resource_List': {'select': '1'}}, 'parent1': {'Resource_List': {'select': '1'}}}
    calls = []
    monkeypatch.setattr(c, 'free_slots', supplement.shared_free_slots)
    monkeypatch.setattr(c.n, 'qstat', lambda: active)
    monkeypatch.setattr(c.n, 'active_jobs', lambda data, user: data)
    def submit(*a, **k):
        job = f'{100+len(calls)}.pbs'; calls.append(job)
        active[job] = {'Resource_List': {'select': '1'}}
        return SimpleNamespace(returncode=0, stdout=job, stderr='')
    monkeypatch.setattr(c.subprocess, 'run', submit)
    c.submit_available(child, plan)
    supplement.refresh_workers(child, plan)
    assert len(calls) == 1
    assert read(child/'state.json')['job_id'] == calls[0]
    assert c.reserved_peer_slots(after, active) == 0
    c.submit_available(child, plan)
    assert len(calls) == 1
    for job in ('parent0', 'parent1'):
        del active[job]
    c.submit_available(child, plan)
    assert len(calls) == 1  # two lanes remain available to the unfinished parent
    ps = read(parent/'state.json'); ps['status'] = 'completed'
    write(parent/'state.json', ps); write(parent/'COMPLETE.json', {})
    supplement.refresh_workers(child, plan)
    c.submit_available(child, plan)
    assert len(calls) == 3
    c.submit_available(child, plan)
    assert len(calls) == 3
    for path in (child/'jobs').glob('*.pbs'):
        assert '#PBS -q R9920261000' in path.read_text()
        assert '#PBS -l select=1' in path.read_text()


def test_parent_unseen_or_ambiguous_jobs_reserve_capacity(tmp_path):
    parent, child, plan, state = scheduling_fixture(tmp_path)
    ps = read(parent/'state.json')
    for i in range(3):
        ps['workers'][str(i)].update(status='queued', job_id=f'{i}.pbs')
    write(parent/'state.json', ps)
    assert supplement.shared_free_slots(plan, state, {}) == 0
    ps['workers']['2'].update(status='submitting', job_id=None)
    write(parent/'state.json', ps)
    assert supplement.shared_free_slots(plan, state, {}) == 0
    ps['workers']['2'].update(status='available')
    write(parent/'state.json', ps)
    assert supplement.shared_free_slots(plan, state, {}) == 1
    state['workers']['0'].update(status='submitting', job_id=None)
    assert supplement.shared_free_slots(plan, state, {}) == 0


def test_parent_old_dispatch_cycle_must_acknowledge_new_peer(tmp_path):
    from conrep.night.io import locked
    parent, child, plan, _ = scheduling_fixture(tmp_path)
    write(parent/'heartbeat.json', dict(pid=10, time=1))
    supplement.attach_reservation(child, plan)
    with locked(parent/'supervisor.lock'):
        assert not supplement.reservation_acknowledged(child, plan)
        write(parent/'heartbeat.json', dict(pid=10, time=2))
        assert not supplement.reservation_acknowledged(child, plan)
        assert not supplement.reservation_acknowledged(child, plan)
        write(parent/'heartbeat.json', dict(pid=10, time=3))
        assert supplement.reservation_acknowledged(child, plan)


def test_partial_export_reports_cost_without_reading_torn_final_log(tmp_path):
    from experiments.selective_export import training_costs
    cfg = tmp_path/'cfg.json'
    write(cfg, dict(run={'output_dir':str(tmp_path/'training')}))
    log = tmp_path/'training/train.jsonl'; log.parent.mkdir()
    log.write_text(json.dumps(dict(elapsed_seconds=2.,canonical_input_tokens={'forget':8,'general':32}))+'\n{"step":')
    result = training_costs({'tasks':[dict(id='t',kind='train',config=str(cfg),output=str(tmp_path))]})[0]
    assert result['logged_step_seconds'] == 2.
    assert result['canonical_input_tokens'] == {'forget':8,'general':32}
    assert result['partial_log_tail_ignored']
