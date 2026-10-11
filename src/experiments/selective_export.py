"""Partial, deduplicated exports; weights and optimizer states are never bundled."""
from collections import Counter,defaultdict
import csv
import datetime as dt
import io
import json
from pathlib import Path
import zipfile
from conrep.night.io import read,write,file_sha
from .selective_eval import valid


def seed_statistics(records,plan):
    import statistics
    specs={t['id']:t for t in plan['tasks']};groups=defaultdict(dict)
    metrics=('forget.any_expression','retain_same.any_expression','retain_other.any_expression','retain_same.gap','utility.mmlu')
    for r in records:
        t=specs[r['experiment']]
        if t['kind']!='train' or t['stage']>=100 or r['partition']!='test':continue
        cfg=read(t['config'])
        if r['step']!=cfg['unlearn']['max_steps']:continue
        if t['target'] in ('T1-rq101','T2-rq101','T3-rq101') and not t['id'].endswith('-cl0'):
            key=('training_seed',t['model'],t['variant'],t['target'])
            groups[key][str(t['seed'])]=r
        if t['seed']==42 and t['target'].startswith('T1-rq') and not t['id'].endswith('-cl0'):
            key=('request_seed',t['model'],t['variant'],'T1');groups[key][t['target'].split('rq')[1]]=r
    out=[]
    for key,values in sorted(groups.items()):
        for metric in metrics:
            samples=[r[metric] for r in values.values() if r.get(metric) is not None]
            out.append(dict(axis=key[0],model=key[1],variant=key[2],target=key[3],metric=metric,
                seeds=sorted(values),n=len(samples),complete_three_seeds=len(samples)==3,
                mean=statistics.mean(samples) if samples else None,sd=statistics.stdev(samples) if len(samples)>1 else None))
    return out


def paired_intervals(plan):
    import numpy as np
    reports={};out=[]
    for task in plan['tasks']:
        if task['kind']!='train' or task['stage']!=10 or task['target']!='T1-rq101' or task['seed']!=42:continue
        cfg=read(task['config']);path=Path(task['output'])/'test'/f"checkpoint-{cfg['unlearn']['max_steps']}"
        if valid(path):reports[task['id']]=(task,read(path/'metrics.json'))
    for task,left in list(reports.values()):
        if task['method']!='conrep':continue
        meta=read(Path(read(task['config'])['selective']['task_path'])/'task.json')
        for other,right in list(reports.values()):
            if other['model']!=task['model'] or other['method']=='conrep':continue
            deltas=defaultdict(list)
            for key in meta['membership']['retain_same_eval']:
                a,b=left['fact_scores'][key],right['fact_scores'][key]
                deltas[a['patient_id']].append(float(a['any'])-float(b['any']))
            values=np.array([np.mean(deltas[p]) for p in sorted(deltas)])
            rng=np.random.default_rng(390101);means=[]
            for _ in range(10):means.extend(values[rng.integers(len(values),size=(1000,len(values)))].mean(1).tolist())
            out.append(dict(conrep=task['id'],baseline=other['id'],task='T1',partition='test',seed=42,
                            metric='retain_same.any_expression',difference=float(values.mean()),
                            ci95=[float(v) for v in np.quantile(means,[.025,.975])],patients=len(values),
                            caveat='Patient-cluster interval for this training seed; not a multi-seed significance claim'))
    return out


def export(campaign,predictions=False):
    campaign=Path(campaign);plan=read(campaign/'plan.json');state=read(campaign/'state.json')
    dest=campaign/'exports';dest.mkdir(exist_ok=True)
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');base='0390-selective-'+stamp
    records=[];reports=[];prediction_files=[];warnings=[];matched=[]
    from .selective_campaign import evaluations
    for task in plan['tasks']:
        cfg=read(task['config'])
        for part,checkpoint,path,reference in evaluations(task,cfg):
            if not valid(path,cfg,checkpoint,part,reference):continue
            report=read(path/'metrics.json')
            step=0 if reference else int(Path(checkpoint).name.split('-')[-1])
            records.append(dict(experiment=task['id'],model=task['model'],method=task['method'],variant=task['variant'],
                target=task['target'],stage=task['stage'],seed=task['seed'],partition=part,step=step,
                protocol_hash=report['protocol_hash'],**report['metrics']))
            relative=str(path.relative_to(campaign));reports.append((relative+'/metrics.json',{k:v for k,v in report.items() if k!='fact_scores'}))
            for name in ('predictions.jsonl','mmlu_predictions.jsonl'):
                if (path/name).is_file():prediction_files.append((relative+'/'+name,path/name))
        selection=Path(task['output'])/'test-selection.json'
        if selection.exists():
            for choice in read(selection)['choices']:
                item=dict(experiment=task['id'],**choice)
                if choice['step'] is not None:
                    path=Path(task['output'])/'test'/f"checkpoint-{choice['step']}"
                    if valid(path):
                        metrics=read(path/'metrics.json')['metrics']
                        item.update(test_forget=metrics['forget.any_expression'],test_retain=metrics[choice['axis']+'.any_expression'],
                            test_meets_ceiling=metrics['forget.any_expression']<=choice['forget_ceiling'])
                matched.append(item)
    manifest=dict(schema='0390-selective-export-v1',campaign=str(campaign),source_hash=plan['source_hash'],
        states=dict(Counter(t['status'] for t in state['tasks'].values())),validated_rows=len(records),
        warnings=warnings,predictions_separate=predictions,weights_included=False)
    buffer=io.StringIO();keys=sorted({k for r in records for k in r});writer=csv.DictWriter(buffer,fieldnames=keys);writer.writeheader();writer.writerows(records)
    summary=dest/(base+'-summary.zip')
    with zipfile.ZipFile(summary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr('checkpoint-results.csv',buffer.getvalue());z.writestr('manifest.json',json.dumps(manifest,indent=2))
        z.writestr('paired-t1-test.json',json.dumps(paired_intervals(plan),indent=2))
        z.writestr('seed-statistics.json',json.dumps(seed_statistics(records,plan),indent=2))
        z.writestr('matched-forgetting.json',json.dumps(matched,indent=2))
        for name in ('plan.json','state.json','PREPARING.json','source.json','inputs.json','model-assets.json',
                     'reuse-ledger.json','tokenizer-audit.json','grid-selection-1.json','grid-selection-2.json','BLOCKED.json'):
            if (campaign/name).is_file():z.write(campaign/name,name)
        for p in (campaign/'configs').glob('*.json'):z.write(p,str(p.relative_to(campaign)))
        for relative,report in reports:z.writestr(relative,json.dumps(report,ensure_ascii=False))
        for task in plan['tasks']:
            p=Path(task['output'])/'test-selection.json'
            if p.is_file():z.write(p,str(p.relative_to(campaign)))
            for name in ('train.jsonl','lineage.json','sampling-coverage.jsonl','augmentation-diagnostics.jsonl'):
                p=Path(read(task['config'])['run']['output_dir'])/name
                if p.is_file() and p.is_relative_to(campaign):z.write(p,str(p.relative_to(campaign)))
            p=Path(read(task['config'])['run']['output_dir'])/f"checkpoint-{read(task['config'])['unlearn']['max_steps']}"/'sampling_state.json'
            if p.is_file() and p.is_relative_to(campaign):z.write(p,str(p.relative_to(campaign)))
        for p in (campaign/'calibration').rglob('*.json'):z.write(p,str(p.relative_to(campaign)))
        z.write(campaign/'data/manifest.json','data/manifest.json')
        for p in (campaign/'data').glob('*/task.json'):z.write(p,str(p.relative_to(campaign)))
    bundles=[summary]
    if predictions:
        bundle=dest/(base+'-predictions.zip')
        with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for name,path in dict(prediction_files).items():z.write(path,name)
        bundles.append(bundle)
    for p in bundles:print(f'BUNDLE={p}\nSIZE_MiB={p.stat().st_size/1024**2:.2f}\nSHA256={file_sha(p)}')
    print(json.dumps(manifest,ensure_ascii=False))
