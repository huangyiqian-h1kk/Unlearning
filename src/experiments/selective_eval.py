"""Resumable selective-v1 evaluation using the frozen v5 typed answer parser."""
import json
from collections import Counter, defaultdict
from pathlib import Path
from conrep.night.io import read, write, sha, file_sha
from .selective_data import rows, write_rows
from .selective_common import load_delta, complete_checkpoint


def signature(cfg, checkpoint, partition, reference=False):
    from . import validation_v5, pmc_scoring_v5, pmc_scoring, pmc_schema, answer_extraction, mmlu_robust
    root=Path(cfg['selective']['task_path'])
    probe=root/('reference' if reference else 'probes')/partition
    files=[probe/'generation.jsonl']
    if not reference: files += [probe/'mcq.jsonl',probe/'safe_background.jsonl',root/'task.json']
    mmlu=cfg['evaluation'].get('mmlu_test_file' if partition=='test' else 'mmlu_file')
    if mmlu: files.append(Path(mmlu))
    model_id=(read(Path(checkpoint)/'COMPLETE.json') if (Path(checkpoint)/'COMPLETE.json').exists()
              else cfg['night']['model_assets_hash'])
    return sha(dict(model=model_id,partition=partition,reference=reference,
        inputs={str(p):file_sha(p) for p in files},options=cfg['evaluation'],
        code={str(m.__name__):file_sha(m.__file__) for m in
              (validation_v5,pmc_scoring_v5,pmc_scoring,pmc_schema,answer_extraction,mmlu_robust)},
        implementation=file_sha(__file__)))


def load_partial(path):
    if not Path(path).exists(): return {}
    data={};offset=0
    with Path(path).open('rb+') as stream:
        while line:=stream.readline():
            try: item=json.loads(line)
            except (ValueError,UnicodeDecodeError):
                if stream.read(): raise ValueError('Corrupt prediction before final line')
                stream.truncate(offset);break
            if item['id'] in data: raise ValueError('Duplicate prediction ID')
            data[item['id']]=item;offset=stream.tell()
    return data


def aggregates(records, task, reference_scores=None):
    groups=defaultdict(list)
    for r in records:
        if r['kind']=='generation': groups[r['fact_id']].append(r)
    scores={k:dict(patient_id=v[0]['patient_id'],attribute=v[0]['attribute'],
                   any=any(x['correct'] for x in v),gen=sum(x['correct'] for x in v)/3)
            for k,v in groups.items()}
    if any(len(v)!=3 for v in groups.values()): raise ValueError('Need exactly three primary expressions')
    masks=({'all':list(scores)} if task is None else {
        'forget':task['membership']['forget'],
        'retain_same':task['membership']['retain_same_eval'],
        'retain_other':task['membership']['retain_other_eval']})
    if task:
        train=set(task['membership']['retain_train'])
        masks['retain']=masks['retain_same']+masks['retain_other']
        for key in ('retain_same','retain_other'):
            masks[key+'_covered']=[k for k in masks[key] if k in train]
            masks[key+'_unseen']=[k for k in masks[key] if k not in train]
    metrics={};counts={}
    for group,keys in list(masks.items()):
        for attr in ('age','gender','diagnosis'):
            masks[group+'.'+attr]=[k for k in keys if scores[k]['attribute']==attr]
    for group,keys in masks.items():
        counts[group]=len(keys)
        for metric in ('any','gen'):
            name=group+('.any_expression' if metric=='any' else '.generation_mean')
            metrics[name]=sum(scores[k][metric] for k in keys)/len(keys) if keys else None
        if reference_scores is not None:
            known=[k for k in keys if reference_scores[k]['any']]
            counts[group+'.sft_known']=len(known)
            metrics[group+'.known_survival']=sum(scores[k]['any'] for k in known)/len(known) if known else None
            metrics[group+'.sft_drop']=((sum(reference_scores[k]['any'] for k in keys)-sum(scores[k]['any'] for k in keys))/len(keys)) if keys else None
    if task:
        for group in ('retain_same','retain_other','retain'):
            r=metrics[group+'.any_expression'];f=metrics['forget.any_expression']
            metrics[group+'.gap']=r-f if r is not None else None
        for group in ('forget','retain_same','retain_other'):
            group_keys=set(masks[group])
            background=[r for r in records if r['kind']=='safe_background' and r['fact_id'] in group_keys]
            metrics[group+'.safe_background']=sum(r['correct'] for r in background)/len(background) if background else None
            counts[group+'.safe_background']=len(background)
            for family in ('attribute-2','attribute-4','identifier-equal','identifier-related'):
                selected=[r for r in records if r['kind']=='mcq' and r['fact_id'] in group_keys and r['task']==family]
                metrics[group+'.'+family]=sum(r['correct'] for r in selected)/len(selected) if selected else None
                counts[group+'.'+family]=len(selected)
        if reference_scores is not None:
            f=set(masks['forget']);near=set(masks['retain_same']);patients=defaultdict(list)
            for k in f|near: patients[scores[k]['patient_id']].append(k)
            eligible=[keys for keys in patients.values() if all(reference_scores[k]['any'] for k in keys)]
            successes=[all(not scores[k]['any'] for k in keys if k in f) and all(scores[k]['any'] for k in keys if k in near) for keys in eligible]
            label='joint_selective' if near else 'whole_entity_forgetting'
            metrics[label]=sum(successes)/len(successes) if successes else None
            counts[label]=len(successes)
    return dict(metrics=metrics,counts=counts,fact_scores=scores,
                scoring_statuses=dict(Counter(r.get('status','unknown') for r in records)))


def run(cfg, checkpoint, output, partition='validation', reference=False):
    from conrep.v2.model import load_tokenizer
    from . import validation_v5,pmc_scoring_v5,mmlu_robust
    import torch
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    identity=signature(cfg,checkpoint,partition,reference)
    if (output/'EVAL_ID.json').exists() and read(output/'EVAL_ID.json')['identity']!=identity:
        raise ValueError('Evaluation identity changed; use a different directory')
    write(output/'EVAL_ID.json',dict(identity=identity))
    root=Path(cfg['selective']['task_path']);probe=root/('reference' if reference else 'probes')/partition
    task=None if reference else read(root/'task.json')
    dataset=[]
    for row in rows(probe/'generation.jsonl'):
        if row['partition']!=partition or row['provided_fact_ids']: raise ValueError('Primary prompt leaks provided facts')
        for family,prompt in row['probe_prompts'].items():
            dataset.append(dict(row,id=row['fact_id']+':generation:'+family,kind='generation',task=family,prompt=prompt))
    if not reference:
        dataset += [dict(r,kind='mcq') for r in rows(probe/'mcq.jsonl')]
        for r in rows(probe/'safe_background.jsonl'):
            if set(r['provided_fact_ids']) & (set(task['membership']['forget'])|{r['fact_id']}):raise ValueError('Unsafe background leak')
            dataset.append(dict(r,kind='safe_background',task='safe_background'))
    expected={r['id'] for r in dataset}
    if len(expected)!=len(dataset):raise ValueError('Duplicate probe identity')
    done=load_partial(output/'predictions.jsonl')
    if set(done)-expected:raise ValueError('Foreign predictions in resume cache')
    mc=cfg['model'];device='cuda' if torch.cuda.is_available() else 'cpu'
    model=load_delta(checkpoint,device,local_only=mc['local_only'],dtype=mc['dtype'],attention=mc['attention'])
    model.eval();tok=load_tokenizer(cfg['unlearn']['checkpoint'],mc['local_only'])
    options=dict(cfg['evaluation']);options['partition']=partition
    todo=[r for r in dataset if r['id'] not in done];size=options['batch_size']
    with (output/'predictions.jsonl').open('a') as stream:
        for start in range(0,len(todo),size):
            chunk=todo[start:start+size]
            generated=validation_v5.generate(model,tok,[r['prompt'] for r in chunk],options)
            for row,text in zip(chunk,generated,strict=True):
                if row['kind'] in ('generation','safe_background'): scored=pmc_scoring_v5.score_answer(text,row['target'],row['patient_id'])
                else:
                    scored=pmc_scoring_v5.score_mcq(text,row['mapping'],row['attribute'])
                    scored['correct']=scored['predicted']==row['correct_letter']
                record=dict(row,output=text,**scored);done[row['id']]=record
                stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            stream.flush()
            if start%256==0:print(f'selective {partition}: {len(done)}/{len(dataset)}',flush=True)
    ref=None
    if not reference:
        refpath=Path(cfg['selective']['reference_dir'])/partition/'metrics.json'
        if not refpath.is_file():raise ValueError('SFT reference missing')
        ref=read(refpath)['fact_scores']
    report=aggregates([done[k] for k in sorted(done)],task,ref)
    mmlu=options.get('mmlu_test_file' if partition=='test' else 'mmlu_file')
    if mmlu:
        if partition=='test':
            overlap=lambda path:{(r['subject'],r['prompt']) for r in rows(path)}
            if overlap(mmlu)&overlap(options['mmlu_file']):raise ValueError('MMLU test overlaps validation')
        score,counts,diagnostics,details=mmlu_robust.evaluate(model,tok,mmlu,options)
        report['metrics']['utility.mmlu']=score;report['mmlu_diagnostics']=diagnostics
        write_rows(output/'mmlu_predictions.jsonl',details)
    report.update(checkpoint=str(checkpoint),protocol_hash=identity,partition=partition,schema='selective-v1',
                  task=None if reference else task['id'])
    write(output/'metrics.json',report)
    write(output/'VALIDATED.json',dict(identity=identity,metrics_sha256=file_sha(output/'metrics.json'),
        predictions_sha256=file_sha(output/'predictions.jsonl'),count=len(done)))
    return report


def valid(output, cfg=None, checkpoint=None, partition='validation',reference=False):
    output=Path(output)
    try:
        m=read(output/'VALIDATED.json')
        return (m['metrics_sha256']==file_sha(output/'metrics.json') and
                m['predictions_sha256']==file_sha(output/'predictions.jsonl') and
                (cfg is None or m['identity']==signature(cfg,checkpoint,partition,reference)))
    except (OSError,ValueError,KeyError):return False
