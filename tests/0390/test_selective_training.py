import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import time

import pytest
import torch
from safetensors.torch import load_file
from conrep.night.io import read, write
from experiments import selective_campaign as c
from experiments.selective_prepare import base_specs
from experiments.selective_common import complete_checkpoint, training_identity, load_delta
from experiments.selective_data import write_rows
from experiments.selective_train import run

spec = importlib.util.spec_from_file_location('selective_test_helpers', Path(__file__).with_name('test_core.py'))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
torch.set_num_threads(1)


def scheduler(tmp_path):
    plan = dict(project_root=str(tmp_path),campaign=str(tmp_path),walltime='12:00:00',shell='worker.sh',peers=[],tasks=[])
    state = dict(status='running',workers={str(i):dict(status='new',job_id=None,allocations=0) for i in range(3)},tasks={})
    for i in range(6):
        task=dict(id=f't{i}',kind='reference',model=f'm{i}',stage=0)
        plan['tasks'].append(task);state['tasks'][task['id']]=dict(status='pending')
    write(tmp_path/'state.json',state)
    return plan,state


def test_two_then_three_and_idempotence(tmp_path,monkeypatch):
    plan,state=scheduler(tmp_path)
    active={'1.pbs':{'Resource_List':{'select':'1'}}};calls=[]
    monkeypatch.setattr(c.n,'qstat',lambda *a:active)
    monkeypatch.setattr(c.n,'active_jobs',lambda value,user:value)
    def submit(*args,**kwargs):
        job=f'{len(calls)+10}.pbs';calls.append(job)
        active[job]={'Resource_List':{'select':'1'}}
        return SimpleNamespace(returncode=0,stdout=job,stderr='')
    monkeypatch.setattr(c.subprocess,'run',submit)
    c.submit_available(tmp_path,plan)
    assert len(calls)==2
    c.submit_available(tmp_path,plan);assert len(calls)==2
    del active['1.pbs']
    c.submit_available(tmp_path,plan);assert len(calls)==3
    c.submit_available(tmp_path,plan);assert len(calls)==3
    for path in (tmp_path/'jobs').glob('*.pbs'):
        assert '#PBS -l select=1' in path.read_text()


def test_capacity_fails_closed_and_reserves_retrying_sft(tmp_path):
    plan,state=scheduler(tmp_path)
    assert c.free_slots(plan,state,{'sft':{'Resource_List':{'select':'2'}}})==1
    peer=tmp_path/'qwen';write(peer/'state.json',dict(status='waiting_retry',job_id='old.pbs'))
    plan['peers']=[str(peer)]
    assert c.free_slots(plan,state,{})==2
    (peer/'COMPLETE.json').touch();assert c.free_slots(plan,state,{})==3
    state['workers']['0'].update(status='submitting',job_id=None)
    assert c.free_slots(plan,state,{})==0
    state['workers']['0'].update(status='queued',job_id='unseen.pbs')
    assert c.free_slots(plan,state,{})==2


def test_qstat_failure_and_ambiguous_qsub_never_duplicate(tmp_path,monkeypatch):
    plan,_=scheduler(tmp_path);calls=[]
    def failed_qstat(*args):raise RuntimeError('PBS unavailable')
    monkeypatch.setattr(c.n,'qstat',failed_qstat)
    monkeypatch.setattr(c.subprocess,'run',lambda *a,**k:calls.append(a))
    with pytest.raises(RuntimeError):c.submit_available(tmp_path,plan)
    assert not calls
    monkeypatch.setattr(c.n,'qstat',lambda *a:{})
    monkeypatch.setattr(c.n,'active_jobs',lambda data,user:data)
    def ambiguous(*a,**k):
        calls.append(a);return SimpleNamespace(returncode=1,stdout='',stderr='connection lost after submission')
    monkeypatch.setattr(c.subprocess,'run',ambiguous)
    c.submit_available(tmp_path,plan);c.submit_available(tmp_path,plan)
    assert len(calls)==1
    assert read(tmp_path/'state.json')['workers']['0']['status']=='submitting'


def test_duplicate_arm_does_not_restart_supervisor(tmp_path,monkeypatch):
    from conrep.night.io import locked
    plan,_=scheduler(tmp_path);write(tmp_path/'plan.json',plan)
    monkeypatch.setattr(c,'verify',lambda *a:None)
    def unexpected(*a):raise AssertionError('Duplicate arm must not query or submit')
    monkeypatch.setattr(c.n,'qstat',unexpected)
    with locked(tmp_path/'supervisor.lock'):
        c.arm(tmp_path)


def test_all_fixed_tasks_precede_grid_and_cancel_requires_reconcile(tmp_path):
    assert len(base_specs())==296
    assert sum(s[5]<40 for s in base_specs())==180
    assert all(s[1]!='G256B32W2P4' for s in base_specs())
    plan,state=scheduler(tmp_path)
    plan['tasks'][1]['stage']=100
    state['tasks']['t0'].update(status='paused',job_id='ended.pbs')
    assert 't0' not in {t['id'] for t in c.ready(plan,state)}
    state['tasks']['t0']['reconciled_job']='ended.pbs'
    assert 't0' in {t['id'] for t in c.ready(plan,state)}
    assert 't1' not in {t['id'] for t in c.ready(plan,state)}


@pytest.mark.parametrize('family',['llama','gemma'])
@pytest.mark.parametrize('method',['conrep','npo','sago','rmu','falcon','lunar','relearn'])
def test_native_update_resume_exact_and_delta_reload(tmp_path,method,family):
    cfg,tok=helpers.tiny_setup(tmp_path)
    if family=='gemma':
        from transformers import Gemma2Config,Gemma2ForCausalLM
        torch.manual_seed(19)
        model=Gemma2ForCausalLM(Gemma2Config(vocab_size=len(tok),hidden_size=16,intermediate_size=32,
            num_hidden_layers=3,num_attention_heads=2,num_key_value_heads=2,head_dim=8,
            max_position_embeddings=256,sliding_window=64,pad_token_id=tok.pad_token_id))
        model.save_pretrained(cfg['unlearn']['checkpoint'])
    from experiments.selective_data import rows
    for group in ('forget','retain','general'):
        path=Path(cfg['data']['prepared_dir'])/(group+'.jsonl');original=rows(path)
        write_rows(path,[dict(original[i%len(original)],id=f'{group}:{i}') for i in range(8)])
    cfg['baseline'].update(method=method,layer=1,train_layers=[0,1],steering_coefficient=20.,
        fit_learning_rate=.001,scheduler_gamma=.9,scheduler_every_steps=10,direction_coefficient=1.,calibration_samples=2)
    cfg['unlearn']['batch_sizes']={'forget':8,'retain':8,'general':8}
    cfg['selective']={'training':{'fixture':True}}
    cfg['night']={'source_hash':'fixture','model_assets_hash':'fixture'}
    if method=='conrep':cfg['conrep'].update(specified_positive='dropout',forget_cl_weight=2.,negative_views=2)
    if method=='falcon':
        path=tmp_path/'layers.json';write(path,dict(checkpoint=cfg['unlearn']['checkpoint'],selected_layer=1))
        cfg['baseline'].update(layer_selection=str(path),optimizer='sophia',temperature=.7,conflict_weights=[.8,1.2],align_weights=[.1,1.9])
    if method=='relearn':
        path=tmp_path/'aug.jsonl'
        write_rows(path,[dict(id=f'a{i}',fact_id=f'forget:{i}',kind='qa',prompt=f'What is condition of P{i} ?',answer='general medical knowledge') for i in range(8)])
        cfg['baseline']['augmented_file']=str(path)
    if method=='lunar':
        for group in ('forget','retain'):
            path=tmp_path/(group+'-requests.jsonl')
            write_rows(path,[dict(fact_id=f'{group}:{i}',prompt=f'What is condition of P{i} ?') for i in range(8)])
            cfg['data'][group+'_requests']=str(path)
        path=tmp_path/'refusal.jsonl';write_rows(path,[dict(prompt='general medical knowledge')])
        cfg['baseline']['refusal_file']=str(path)
    cfg['run']['output_dir']=str(tmp_path/'full')
    assert run(cfg)==0
    cfg['run']['output_dir']=str(tmp_path/'resumed')
    assert run(cfg,stop_after_step=1)==75
    assert run(cfg,tmp_path/'resumed/checkpoint-1')==0
    file='adapter_model.safetensors' if method in ('conrep','npo','sago','relearn') else 'delta.safetensors'
    full=load_file(str(tmp_path/'full/checkpoint-2'/file));resumed=load_file(str(tmp_path/'resumed/checkpoint-2'/file))
    assert full.keys()==resumed.keys()
    assert all(torch.equal(full[k],resumed[k]) for k in full)
    assert complete_checkpoint(tmp_path/'resumed/checkpoint-2',identity=training_identity(cfg),world=1,resume=True)
    model=load_delta(tmp_path/'resumed/checkpoint-2','cpu',dtype='float32',attention='eager')
    if file=='delta.safetensors':
        state=dict(model.named_parameters())|dict(model.named_buffers())
        assert all(torch.equal(state[k],v) for k,v in full.items())


def test_partial_prediction_recovery_and_empty_same_group(tmp_path):
    from experiments.selective_eval import load_partial,aggregates
    path=tmp_path/'partial.jsonl';path.write_text('{"id":"a"}\n{"id":')
    assert list(load_partial(path))==['a']
    assert path.read_text()=='{"id":"a"}\n'
    records=[dict(id=str(i),kind='generation',fact_id='f',patient_id='1',attribute='age',correct=False) for i in range(3)]
    task=dict(membership=dict(forget=['f'],retain_same_eval=[],retain_other_eval=[],retain_train=[]))
    report=aggregates(records,task,{'f':{'any':True}})
    assert report['metrics']['retain_same.any_expression'] is None
    assert report['metrics']['retain_same.gap'] is None
    assert report['metrics']['whole_entity_forgetting']==1


def test_grid_is_deferred_and_has_32_6_4_runs(tmp_path):
    from experiments.selective_grid import advance
    from experiments.selective_design import REPLICATION_CANDIDATES
    from experiments.selective_common import BASELINES
    from experiments.selective_prepare import publish_task
    cfg,_=helpers.tiny_setup(tmp_path)
    cfg['baseline']['fit_learning_rate']=.001
    cfg['selective']=dict(training={'fixture':True},task_path=str(tmp_path/'T1-rq101'))
    cfg['night']=dict(source_hash='fixture',model_assets_hash='fixture')
    plan=dict(campaign=str(tmp_path),grid_phase=0,tasks=[])
    for model,variants in REPLICATION_CANDIDATES.items():
        for variant in list(variants)+list(BASELINES):
            current=copy.deepcopy(cfg);current['selective'].update(model=model,variant=variant)
            current['baseline']['method']='conrep' if variant in variants else variant
            current['baseline']['train_layers']=[0,1]
            plan['tasks'].append(publish_task(plan,current,model+'-'+variant,10))
    state=dict(tasks={t['id']:dict(status='pending') for t in plan['tasks']})
    write(tmp_path/'plan.json',plan);write(tmp_path/'state.json',state)
    assert advance(tmp_path) is False
    previous=len(plan['tasks'])
    for count in (32,6,4):
        plan=read(tmp_path/'plan.json');state=read(tmp_path/'state.json')
        for t in plan['tasks']:
            state['tasks'][t['id']]['status']='completed'
            step=read(t['config'])['unlearn']['max_steps']
            write(Path(t['output'])/'validation'/f'checkpoint-{step}'/'metrics.json',dict(metrics={
                'forget.any_expression':.2,'retain_same.any_expression':.7,'retain_other.any_expression':.8,'utility.mmlu':.6}))
        write(tmp_path/'state.json',state)
        assert advance(tmp_path)
        plan=read(tmp_path/'plan.json');assert len(plan['tasks'])-previous==count
        assert all(not t['test_terminal'] for t in plan['tasks'][previous:])
        previous=len(plan['tasks'])
    assert not advance(tmp_path)


def test_validation_selection_preserves_terminal_and_empty_same(tmp_path):
    task=dict(output=str(tmp_path));cfg=dict(unlearn=dict(max_steps=20,save_steps=10))
    for step,f,r in [(10,.4,.8),(20,.1,.5)]:
        write(tmp_path/'validation'/f'checkpoint-{step}'/'metrics.json',dict(metrics={
            'forget.any_expression':f,'retain_same.any_expression':None,'retain_other.any_expression':r}))
    c.freeze_test_selection(task,cfg)
    result=read(tmp_path/'test-selection.json')
    assert result['steps']==[10,20]
    assert all(x['step'] is None for x in result['choices'] if x['axis']=='retain_same')
    c.freeze_test_selection(task,cfg)
