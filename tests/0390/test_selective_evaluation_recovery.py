"""Run against the actual frozen v5 parser supplied by the scientific snapshot."""
import copy
import importlib.util
import os
from pathlib import Path
import pytest

frozen = os.environ.get('CONREP_FROZEN_TEST_ROOT')
if not frozen:
    pytest.skip('Set CONREP_FROZEN_TEST_ROOT to a verified campaign/code snapshot',allow_module_level=True)
import experiments
experiments.__path__.append(str(Path(frozen)/'src/experiments'))
from conrep.night.io import read, write, file_sha
from experiments import pmc_schema, pmc_scoring_v5 as scoring, selective_eval as ev, selective_recovery as repair
from experiments.selective_mcq import extract_binary
from experiments.selective_data import write_rows, rows

spec=importlib.util.spec_from_file_location('recovery_tiny_helpers',Path(__file__).with_name('test_core.py'))
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)

META=dict(ref='repair-test',files={'reviewed-extension':'sha256-fixture'})


@pytest.mark.parametrize('text,wanted',[
    ('A','A'),('B.','B'),('(A)','A'),('The answer is B.','B'),('female','B'),('male','A'),
    ('The answer is female.','B'),('A. male','A'),('B. female','B'),
    ('C',None),('D.',None),('C. male',None),('The answer is C.',None),
    ('A or B',None),('The answer is A or B.',None),('Maybe A',None),
    ('A. male\nB. female',None),('A. male\nB. female\nAnswer: B','B'),
    ('I do not know',None),('',None),
])
def test_binary_gold_blind_parser(text,wanted):
    assert extract_binary(text,{'A':'male','B':'female'})['predicted']==wanted


def hook(monkeypatch,command='evaluate'):
    from experiments import selective_calibrate
    from conrep.night import campaign as night
    # Register originals with pytest before the production installer mutates them.
    for module,name in ((scoring,'extract'),(ev,'run'),(selective_calibrate,'run'),(night,'reconcile')):
        monkeypatch.setattr(module,name,getattr(module,name))
    repair.install(META,command)


def test_four_choice_and_generation_results_are_byte_equivalent(monkeypatch):
    mappings=[({'A':'male','B':'female','C':'unknown','D':'other'},'gender'),
        ({'A':'1 year old','B':'2 years old','C':'3 years old','D':'4 years old'},'age'),
        ({'A':'123','B':'234','C':'345','D':'456'},'identifier')]
    texts=['A','B.','C.','D','The answer is B.','A or B','female','male','2 years old',
           'Patient 234','C. 345','I do not know','Not A.','A. male\nB. female','']
    before=[scoring.score_mcq(t,m,a) for m,a in mappings for t in texts]
    target=pmc_schema.target('diagnosis','asthma')
    generations=[scoring.score_answer(t,target,'123') for t in texts]
    hook(monkeypatch)
    assert [scoring.score_mcq(t,m,a) for m,a in mappings for t in texts]==before
    assert [scoring.score_answer(t,target,'123') for t in texts]==generations


def eval_fixture(tmp_path,family):
    cfg,tok=helpers.tiny_setup(tmp_path)
    if family=='gemma':
        from transformers import Gemma2Config,Gemma2ForCausalLM
        import torch
        torch.manual_seed(19)
        model=Gemma2ForCausalLM(Gemma2Config(vocab_size=len(tok),hidden_size=16,intermediate_size=32,
            num_hidden_layers=3,num_attention_heads=2,num_key_value_heads=2,head_dim=8,
            max_position_embeddings=256,sliding_window=64,pad_token_id=tok.pad_token_id))
        model.save_pretrained(cfg['unlearn']['checkpoint'])
    root=tmp_path/'task';ref=tmp_path/'reference'
    cfg['selective']=dict(task_path=str(root),reference_dir=str(ref))
    cfg['night']=dict(model_assets_hash='tiny-model')
    cfg['evaluation'].update(max_length=128,max_new_tokens=3,mmlu_test_file=None,limit=None)
    facts=[dict(id='1:gender',fact_id='1:gender',patient_id='1',attribute='gender',target=pmc_schema.target('gender','male')),
           dict(id='2:diagnosis',fact_id='2:diagnosis',patient_id='2',attribute='diagnosis',target=pmc_schema.target('diagnosis','asthma'))]
    membership=dict(forget=['1:gender'],retain_same_eval=[],retain_other_eval=['2:diagnosis'],retain_train=[])
    write(root/'task.json',dict(id='T0-General-only',membership=membership))
    for partition in ('validation','test'):
        probe=root/'probes'/partition
        write_rows(probe/'generation.jsonl',[dict(f,partition=partition,provided_fact_ids=[],probe_prompts={
            'qa':'What is condition of patient 1 ?', 'cloze':'patient 1 has condition', 'paraphrase':'What is condition of patient 2 ?'}) for f in facts])
        write_rows(probe/'mcq.jsonl',[
            dict(id='1:gender:attribute',fact_id='1:gender',patient_id='1',attribute='gender',task='attribute-2',
                 mapping={'A':'male','B':'female'},correct_letter='A',partition=partition,prompt='What is condition ? A. male B. female'),
            dict(id='2:diagnosis:attribute',fact_id='2:diagnosis',patient_id='2',attribute='diagnosis',task='attribute-4',
                 mapping={'A':'asthma','B':'diabetes','C':'fever','D':'healthy'},correct_letter='A',partition=partition,
                 prompt='What is condition ? A. asthma B. diabetes C. fever D. healthy')])
        write_rows(probe/'safe_background.jsonl',[dict(facts[1],id='2:diagnosis:safe-background',
            provided_fact_ids=[],partition=partition,prompt='general knowledge What is condition of patient 2 ?')])
        write(ref/partition/'metrics.json',dict(fact_scores={f['id']:dict(any=True) for f in facts}))
    return cfg


@pytest.mark.parametrize('family',['llama','gemma'])
def test_actual_model_eval_reproduces_crash_then_resumes_without_regeneration(tmp_path,monkeypatch,family):
    cfg=eval_fixture(tmp_path,family);output=tmp_path/'evaluation'
    with pytest.raises(ValueError,match='Four nonempty options'):
        ev.run(cfg,cfg['unlearn']['checkpoint'],output)
    prefix=(output/'predictions.jsonl').read_bytes()
    assert len(rows(output/'predictions.jsonl'))==6
    identity=read(output/'EVAL_ID.json')['identity']
    from experiments import validation_v5
    original=validation_v5.generate;seen=[]
    def generate(model,tok,prompts,options):
        seen.extend(prompts)
        return original(model,tok,prompts,options)
    monkeypatch.setattr(validation_v5,'generate',generate)
    hook(monkeypatch)
    report=ev.run(cfg,cfg['unlearn']['checkpoint'],output)
    assert len(seen)==3  # Only two MCQs and one background row remain.
    assert (output/'predictions.jsonl').read_bytes().startswith(prefix)
    assert len(rows(output/'predictions.jsonl'))==9
    assert read(output/'EVAL_ID.json')['identity']==identity
    assert ev.valid(output,cfg,cfg['unlearn']['checkpoint'])
    assert read(output/'VALIDATED.json')['execution_identity']==report['evaluation_extension']['execution_identity']
    assert read(output/'EVALUATION_EXTENSION.json')['preserved_predictions']==6
    # The test partition and four-option question follow the same repaired path.
    report=ev.run(cfg,cfg['unlearn']['checkpoint'],tmp_path/'test','test')
    assert ev.valid(tmp_path/'test',cfg,cfg['unlearn']['checkpoint'],'test')


def test_foreign_prediction_metadata_is_rejected_before_model_load(tmp_path,monkeypatch):
    cfg=eval_fixture(tmp_path,'llama');out=tmp_path/'bad-cache'
    write(out/'EVAL_ID.json',dict(identity=ev.signature(cfg,cfg['unlearn']['checkpoint'],'validation')))
    item=next(iter(repair.probe_items(cfg,'validation').values()))
    write_rows(out/'predictions.jsonl',[dict(item,prompt='foreign prompt',output='male',correct=True)])
    hook(monkeypatch)
    with pytest.raises(ValueError,match='metadata mismatch'):
        ev.run(cfg,cfg['unlearn']['checkpoint'],out)


def augmentation_fixture(tmp_path):
    cfg,_=helpers.tiny_setup(tmp_path)
    cfg['baseline'].update(method='relearn',augmented_file=str(tmp_path/'calibration/augmentation.jsonl'))
    cfg['selective']=dict(calibration_identity='original-frozen-identity')
    facts=[dict(id=f'forget:{i}',patient_id=str(i),attribute='diagnosis',target=pmc_schema.target('diagnosis','asthma')) for i in range(4)]
    write_rows(Path(cfg['data']['prepared_dir'])/'forget.jsonl',facts)
    write_rows(cfg['baseline']['augmented_file'],[dict(id='relearn:0',fact_id='forget:0',kind='qa',prompt='original question',answer='general medical knowledge')])
    return cfg


def test_relearn_extends_missing_facts_without_touching_completed_material(tmp_path,monkeypatch):
    cfg=augmentation_fixture(tmp_path)
    from experiments.baselines import prepare
    calls=[]
    def augment(temporary,output):
        requests=rows(temporary['data']['forget_requests']);calls.append(requests)
        assert temporary['baseline']==cfg['baseline']
        r=requests[0]
        write_rows(output,[dict(id='x',fact_id=r['fact_id'],kind='qa',prompt=r['prompt'],answer='general medical knowledge')])
        write(str(output)+'.audit.json',dict(accepted=1))
    monkeypatch.setattr(prepare,'relearn_augment',augment)
    repair.extend_relearn(cfg,META)
    assert [len(c) for c in calls]==[3,2,1]
    dest=Path(cfg['baseline']['augmented_file'])
    assert len(rows(dest))==4
    assert rows(dest)[0]['prompt']=='original question'
    from experiments.selective_calibrate import valid
    assert valid(cfg)
    before=dest.read_bytes()
    repair.extend_relearn(cfg,META)
    assert dest.read_bytes()==before and len(calls)==3


def test_relearn_exhaustion_never_approves_partial_coverage(tmp_path,monkeypatch):
    cfg=augmentation_fixture(tmp_path)
    from experiments.baselines import prepare
    calls=[]
    def augment(temporary,output):
        calls.append(True);write_rows(output,[]);write(str(output)+'.audit.json',dict(accepted=0))
    monkeypatch.setattr(prepare,'relearn_augment',augment)
    dest=Path(cfg['baseline']['augmented_file']);before=dest.read_bytes()
    with pytest.raises(ValueError,match='recovery_exhausted'):
        repair.extend_relearn(cfg,META)
    assert len(calls)==6 and dest.read_bytes()==before
    assert not (dest.parent/'CALIBRATED.json').exists()
    # Repeating an exhausted attempt reuses its six audited caches.
    with pytest.raises(ValueError,match='recovery_exhausted'):
        repair.extend_relearn(cfg,META)
    assert len(calls)==6


def test_requeue_is_evidence_bound_and_only_once(tmp_path,monkeypatch):
    from experiments import selective_campaign
    monkeypatch.setattr(selective_campaign,'trained',lambda *args:True)
    path=tmp_path/'config.json';write(path,dict(baseline={'method':'conrep'}))
    tasks=[dict(id=key,kind='train',method='conrep',config=str(path),output=str(tmp_path/key)) for key in ('fixed','unrelated')]
    plan=dict(tasks=tasks)
    state=dict(status='blocked',workers={},tasks={t['id']:dict(status='failed',error="RuntimeError('Validation incomplete; test selection remains blocked')") for t in tasks})
    write(tmp_path/'state.json',state)
    log=tmp_path/'fixed/validation/checkpoint-10/console.log';log.parent.mkdir(parents=True)
    log.write_text('ValueError: Four nonempty options A/B/C/D required\n')
    assert repair.requeue_supported(tmp_path,plan,META)==[('fixed','binary_mcq_parser')]
    after=read(tmp_path/'state.json');assert after['tasks']['unrelated']['status']=='failed'
    after['tasks']['fixed']['status']='failed';write(tmp_path/'state.json',after)
    assert repair.requeue_supported(tmp_path,plan,META)==[]
