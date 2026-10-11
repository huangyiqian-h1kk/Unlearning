"""Freeze real historical configs; construct the complete fixed-first task queue."""
import copy
import importlib.metadata
import json
from pathlib import Path
import sys
from conrep.night import campaign as night
from conrep.night.io import read,write,sha,file_sha,locked
from .selective_design import CANDIDATES,REPLICATION_CANDIDATES,run_counts
from .selective_common import SCHEMA,BASELINES,training_identity
from . import selective_data

ENTRY='scripts/abci/0390_selective.py'
SHELL='scripts/abci/0390_selective_worker.sh'


def historical_configs(project):
    base=project/'results/validated_v2/0390';result={};evidence={}
    for model,variants in CANDIDATES.items():
        result[model]={}
        for variant in variants:
            run_id=f'{model}-{variant}-s42'
            matches=sorted(base.glob(f'conrep-*/configs/{run_id}.json'))
            if not matches:raise FileNotFoundError('Historical candidate config missing: '+run_id)
            # Same scientific config can be copied into a later archive; insist on equal effective parameters.
            signatures={sha({k:read(p)[k] for k in ('model','unlearn','conrep','lora')}) for p in matches}
            if len(signatures)!=1:raise ValueError('Conflicting historical configurations: '+run_id)
            cfg=read(matches[0]);result[model][variant]=cfg;evidence[str(matches[0])]=file_sha(matches[0])
        parent=base/night.BASE_CAMPAIGN.split('/')[-1]/model
        for method in BASELINES:
            matches=sorted(parent.glob(f'groups/*/suites/{method}/{method}/method-config.json'))
            if len(matches)!=1:raise FileNotFoundError(f'Need the actual resolved {model}/{method} baseline config under {parent}')
            cfg=read(matches[0]);cfg=night.absolute_paths(cfg,project)
            if cfg['baseline']['method']!=method:raise ValueError('Baseline method mismatch')
            if cfg['unlearn']['checkpoint']!=result[model][variants[0]]['unlearn']['checkpoint']:
                raise ValueError('Baseline and ConRep must share the same SFT')
            result[model][method]=cfg;evidence[str(matches[0])]=file_sha(matches[0])
    return result,evidence


def task_config(base,model,method,variant,target,seed,campaign,source_hash,assets_hash,changes=None):
    cfg=copy.deepcopy(base);task=read(campaign/'data'/target/'task.json')
    root=campaign/'data'/target
    cfg.pop('storage',None);cfg.pop('diagnostics',None);cfg.pop('selection',None)
    cfg['diagnostics']={'sampling_coverage':True}
    cfg['baseline']['method']=method
    cfg['run']['seed']=seed
    cfg['unlearn'].update(max_steps=task['counts']['max_steps'],save_steps=task['counts']['forget']//10)
    if cfg['unlearn']['batch_sizes']['forget']!=8 or cfg['unlearn']['gradient_accumulation_steps']!=1:
        raise ValueError('Historical recipe must use F8/accum1 before exposure-matched migration')
    cfg['data'].update(prepared_dir=str(root),schema='selective-v1',
                       forget_requests=str(root/'forget_requests.jsonl'),retain_requests=str(root/'retain_requests.jsonl'))
    cfg['selective']=dict(task_path=str(root),reference_dir=str(campaign/'references'/model),model=model,
        variant=variant,training=dict(data=task['train_hashes'],request_seed=task['request_seed'],
            environment=environment(),task_membership=task['membership_sha256']))
    # Evaluation fields are intentionally separate from the training hash.
    for key in list(cfg['data']):
        if key.endswith('_generation') or key.endswith('_mcq'):cfg['data'].pop(key)
    cfg['evaluation'].update(partition='validation',limit=None)
    cfg['night']=dict(source_hash=source_hash,model_assets_hash=assets_hash)
    calibration=sha(dict(model=model,base=assets_hash,task=task['membership_sha256'],method=method,
                         options={k:v for k,v in cfg['baseline'].items() if k not in ('layer_selection','augmented_file','fit_learning_rate')}))
    cfg['selective']['calibration_identity']=calibration
    if method=='falcon':cfg['baseline']['layer_selection']=str(campaign/'calibration'/calibration/'layers.json')
    if method=='relearn':cfg['baseline']['augmented_file']=str(campaign/'calibration'/calibration/'augmentation.jsonl')
    for field,value in (changes or {}).items():
        section,key=field.split('.');cfg[section][key]=value
    return cfg


def environment():
    return {name:importlib.metadata.version(name) for name in ('torch','transformers','peft','numpy','accelerate')}


def publish_task(plan,cfg,task_id,stage,*,kind='train',test=True):
    campaign=Path(plan['campaign']);out=campaign/'experiments'/task_id
    if kind!='historical':cfg['run']['output_dir']=str(out/'training')
    identity=training_identity(cfg)
    path=campaign/'configs'/(task_id+'.json');write(path,cfg)
    from .selective_common import storage_forecast
    return dict(id=task_id,model=cfg['selective']['model'],method=cfg['baseline']['method'],
                variant=cfg['selective']['variant'],seed=cfg['run']['seed'],stage=stage,priority=stage,
                kind=kind,config=str(path),config_hash=file_sha(path),identity=identity,output=str(out),
                preferred_worker=len(plan['tasks'])%3,late_admission=True,test_terminal=test,
                target=Path(cfg['selective']['task_path']).name,
                storage=storage_forecast(cfg) if kind=='train' else {'additional_bytes':2*10**9})


def base_specs():
    specs=[]
    for target,stage in [('T0-C100',10),('T1-rq101',10)]+[(f'T{i}-rq101',20) for i in range(2,9)]+[('E2-rq101',30)]:
        for model,variants in CANDIDATES.items():
            specs += [(model,v,'conrep',target,42,stage,{}) for v in variants]
            specs += [(model,m,m,target,42,stage,{}) for m in BASELINES]
    for model,variants in REPLICATION_CANDIDATES.items():
        for target in ('T1-rq101','T2-rq101','T3-rq101'):
            for seed in (43,44):
                specs += [(model,v,'conrep',target,seed,40,{}) for v in variants]
                specs += [(model,m,m,target,seed,40,{}) for m in BASELINES]
        for rq in (102,103):
            target=f'T1-rq{rq}'
            specs += [(model,v,'conrep',target,42,40,{}) for v in variants]
            specs += [(model,m,m,target,42,40,{}) for m in ('sago','lunar')]
        common=next(v for v in variants if 'IB2' in v)
        specs += [(model,common,'conrep','T0-C50',42,40,{}),
                  (model,common,'conrep','T1-rq101',42,40,{'conrep.specified_cl_weight':0.})]
    return specs


def audit_tokenizers(configs,tasks,campaign):
    from conrep.v2.model import load_tokenizer
    from conrep.night.noise import audit_rows as replacement
    from conrep.night.insertion import audit_rows as insertion
    from .data import chat_example
    report={}
    for model,variants in CANDIDATES.items():
        base=configs[model][variants[0]];tok=load_tokenizer(base['unlearn']['checkpoint'],True)
        vocab=read(Path(base['unlearn']['checkpoint'])/'config.json')['vocab_size'];all_rows={}
        for target in tasks:
            for group in ('forget','retain'):
                for row in selective_data.rows(campaign/'data'/target/(group+'.jsonl')):all_rows[row['relation_id']]=row
        data=list(all_rows.values())
        for row in data:
            if len(tok.encode(row['text']))>512:raise ValueError('Fact truncated at 512: '+row['id'])
            ex=chat_example(tok,dict(row,kind='document'),512)
            if not any(t!=-100 for t in ex['labels'][1:]):raise ValueError('No supervised tokens: '+row['id'])
        checks={name:fn(tok,data,512,vocab) for name,fn in [('replacement',replacement),('insertion',insertion)]}
        if any(v['unsupported_rows'] or v['audited_rows']!=len(data) for v in checks.values()):raise ValueError('Unsupported protected noise facts')
        report[model]=checks
    write(campaign/'tokenizer-audit.json',report)


def prepare(campaign):
    campaign=Path(campaign).resolve();meta=read(campaign/'PREPARING.json');project=Path(meta['project_root'])
    if (campaign/'plan.json').exists():
        from .selective_campaign import verify
        verify(campaign,read(campaign/'plan.json'));print('Prepared campaign verified; existing state retained');return
    files=read(campaign/'code-files.json')
    for n,h in files.items():
        if file_sha(campaign/'code'/n)!=h:raise ValueError('Frozen source changed: '+n)
    configs,evidence=historical_configs(project)
    prepared={cfg['data']['prepared_dir'] for values in configs.values() for cfg in values.values()}
    if len(prepared)!=1:raise ValueError('Historical tasks use different rich prepared data')
    prepared=Path(next(iter(prepared)))
    source_hash=sha(files);assets={}
    for model,variants in CANDIDATES.items():
        start=Path(configs[model][variants[0]]['unlearn']['checkpoint'])
        if (start/'adapter_config.json').exists() or not list(start.glob('*.safetensors')):raise ValueError('Need full SFT weights')
        assets.update({str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in start.iterdir() if p.is_file()})
        teacher=Path(configs[model]['relearn']['baseline'].get('augmentation_model') or configs[model]['relearn']['model']['name_or_path'])
        if not (teacher/'config.json').is_file():raise ValueError('ReLearn needs the historical local teacher model')
        assets.update({str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in teacher.iterdir() if p.is_file()})
    with locked(campaign/'build.lock'):
        if (campaign/'plan.json').exists():
            from .selective_campaign import verify
            verify(campaign,read(campaign/'plan.json'));print('Prepared campaign verified; existing state retained');return
        if not (campaign/'data/manifest.json').exists():selective_data.prepare(prepared,campaign/'data')
        targets=read(campaign/'data/manifest.json')['tasks']
        audit_tokenizers(configs,targets,campaign)
        write(campaign/'historical-configs.json',configs);write(campaign/'historical-evidence.json',evidence)
        write(campaign/'model-assets.json',assets);write(campaign/'source.json',dict(files=files,source_hash=source_hash))
        inputs=dict(evidence)
        for p in (campaign/'data').rglob('*'):
            if p.is_file():inputs[str(p)]=file_sha(p)
        for values in configs.values():
            for cfg in values.values():
                for key in ('mmlu_file','mmlu_test_file'):
                    if cfg['evaluation'].get(key):inputs[cfg['evaluation'][key]]=file_sha(cfg['evaluation'][key])
                if cfg['baseline'].get('refusal_file'):
                    p=Path(cfg['baseline']['refusal_file']);inputs[str(p)]=file_sha(p)
        write(campaign/'inputs.json',inputs)
        plan=dict(schema=SCHEMA,campaign=str(campaign),project_root=str(project),source_hash=source_hash,
            python=sys.executable,entry=str(campaign/'code'/ENTRY),shell=str(campaign/'code'/SHELL),
            workers=3,world_size=8,queue='R9920261000',account='gcg51557',rtype='rt_HF',walltime='12:00:00',
            poll_seconds=30,reserve_bytes=100*10**9,initial_task_seconds=3600,
            recovery_policy=dict(night.RECOVERY_DEFAULTS),environment=environment(),tasks=[],
            peers=[str(project/'results/validated_v2/0390/qwen3-4b-rich-sft-20261011')],
            grid_phase=0,budget=run_counts(6),fixed_first=True)
        for model,variants in CANDIDATES.items():
            cfg=task_config(configs[model][variants[0]],model,'conrep','sft-reference','T1-rq101',42,campaign,source_hash,sha(assets))
            cfg['selective']['task_path']=str(campaign/'data')
            plan['tasks'].append(publish_task(plan,cfg,model+'-sft-reference',0,kind='reference'))
        ledger=[]
        for model,variant,method,target,seed,stage,changes in base_specs():
            cfg=task_config(configs[model][variant],model,method,variant,target,seed,campaign,source_hash,sha(assets),changes)
            suffix='-cl0' if changes else ''
            task_id=f'{model}-{target}-{variant}-s{seed}{suffix}'
            task=publish_task(plan,cfg,task_id,stage);plan['tasks'].append(task)
            ledger.append(dict(task=task_id,action='train_new',reason='New target / retain membership / training seed / objective ablation',training_identity=task['identity']))
        assert len(plan['tasks'])==298  # 296 fixed tasks plus two shared SFT references
        from .selective_common import complete_checkpoint
        for model,variants in CANDIDATES.items():
            variant=next(v for v in variants if 'IB2' in v);old=configs[model][variant]
            weight_root=Path(old['run']['output_dir'])
            points=[weight_root/f'checkpoint-{s}' for s in night.expected_checkpoints(old)]
            if not all(complete_checkpoint(p) for p in points):
                ledger.append(dict(task=model+'-T0-C0-bridge',action='unavailable',reason='Historical checkpoint weights incomplete; no retraining'))
                continue
            cfg=task_config(old,model,'conrep',variant,'T0-C0',42,campaign,source_hash,sha(assets))
            cfg['run']['output_dir']=str(weight_root)
            task=publish_task(plan,cfg,model+'-T0-C0-bridge',5,kind='historical');plan['tasks'].append(task)
            for point in points:
                inputs[str(point/'COMPLETE.json')]=file_sha(point/'COMPLETE.json')
                for p in point.iterdir():
                    if p.suffix=='.safetensors':assets[str(p)]=dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
            ledger.append(dict(task=task['id'],action='reuse_checkpoint_reevaluate',historical_config_sha256=sha(old),
                               reason='Same historical weights, selective-v1 probes; historical-v5 scores stay separate'))
        write(campaign/'inputs.json',inputs);write(campaign/'model-assets.json',assets)
        write(campaign/'reuse-ledger.json',dict(tasks=ledger,excluded=['plain G256B32W2P4','old original-task positive/noise grid'],
            historical_c0='Two common-IB variants reevaluated when all historical weights exist; missing weights reported without retraining.'))
        write(campaign/'state.json',dict(status='prepared',started_at=None,deadline=None,
            tasks={t['id']:dict(status='pending',attempts=0,failures=0) for t in plan['tasks']},
            workers={str(i):dict(job_id=None,status='new',allocations=0,recovery_failures=0) for i in range(3)}))
        write(campaign/'plan.json',plan)
        print(json.dumps(dict(fixed_trainings=296,reference_tasks=2,historical_reevaluations=len(plan['tasks'])-298,later_grid=42,max_active_jobs=3,
                              campaign=str(campaign),status='prepared; no jobs submitted')))
