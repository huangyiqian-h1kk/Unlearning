"""Fixed rich-IA selective tasks and answer-free probe construction."""
import copy
from collections import Counter, defaultdict
from pathlib import Path
from conrep.night.io import read, write, sha, file_sha
from .selective_design import load_facts, build_request, build_legacy

TEMPLATES = {
 'validation': {'qa':'What is the {attribute} of patient {patient_id}?',
                'cloze':'Complete the record: Patient {patient_id} has the following {attribute}:',
                'paraphrase':'Give the recorded {attribute} for the patient whose identifier is {patient_id}.'},
 'test': {'qa':'State patient {patient_id}\'s {attribute}.',
          'cloze':'Fill in the missing value: The {attribute} recorded for patient {patient_id} is ____.',
          'paraphrase':'In the clinical records, which {attribute} is associated with patient ID {patient_id}?'}
}

def write_rows(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(__import__('json').dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))

def rows(path):
    import json
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def normalized_facts(facts):
    out={}
    for key, original in facts.items():
        row=copy.deepcopy(original)
        row.update(id=key,relation_id=key,patient_id=str(row['patient_id']),split='selective')
        out[key]=row
    return out

def gen_rows(facts, keys, partition):
    return [dict(id=k, fact_id=k, patient_id=facts[k]['patient_id'],attribute=facts[k]['attribute'],
                 target=facts[k]['target'],partition=partition,provided_fact_ids=[],
                 probe_prompts={name:template.format(**facts[k]) for name,template in TEMPLATES[partition].items()})
            for k in sorted(keys)]

def mcq_rows(facts, keys, partition):
    from .pmc_schema import target_key
    from .pmc_scoring_v5 import score_answer
    by_attr=defaultdict(list)
    for key,row in facts.items(): by_attr[row['attribute']].append(key)
    output=[];ineligible=Counter()
    for key in sorted(keys):
        row=facts[key];attr=row['attribute'];gold=target_key(row['target'])
        pool=sorted(by_attr[attr],key=lambda other:sha([partition,key,other]))
        other_values={}
        for other in pool:
            value=target_key(facts[other]['target'])
            if value!=gold: other_values.setdefault(value,other)
        wrong=[]
        for other in other_values.values():
            if any(score_answer(facts[other]['target']['answer'],facts[chosen]['target'])['correct']
                   or score_answer(facts[chosen]['target']['answer'],facts[other]['target'])['correct'] for chosen in [key]+wrong):continue
            wrong.append(other)
            if len(wrong)==3:break
        count=2 if attr=='gender' else 4
        if len(wrong)>=count-1:
            chosen=[key]+wrong[:count-1]
            ordered=sorted(chosen,key=lambda k:sha([partition,key,'options',k]))
            mapping={letter:facts[k]['target']['answer'] for letter,k in zip('ABCD',ordered)}
            correct='ABCD'[ordered.index(key)]
            question=f"What is the {attr} of patient {row['patient_id']}?"
            output.append(dict(id=key+':attribute',fact_id=key,patient_id=row['patient_id'],attribute=attr,
                task='attribute-'+str(count),mapping=mapping,correct_letter=correct,partition=partition,
                fact_ids=ordered,prompt=question+'\n'+'\n'.join(f'{k}. {v}' for k,v in mapping.items())+
                '\nRespond with exactly one option letter.'))
        else: ineligible[attr+':attribute']+=1
        # Relation diagnostics: all options must refer to unique patients.
        # Typed aliases must not turn another option into a second valid answer.
        if len(wrong)>=3:
            equal=[key]+wrong[:3]
            same=[k for k in pool if k!=key and target_key(facts[k]['target'])==gold]
            for family, chosen in [('identifier-equal',equal),('identifier-related',([same[0]]+wrong[:3]) if same else [])]:
                if not chosen: ineligible[family]+=1;continue
                answer=chosen[0];chosen=sorted(chosen,key=lambda k:sha([partition,key,family,k]))
                mapping={letter:facts[k]['patient_id'] for letter,k in zip('ABCD',chosen)}
                question=(f"Which patient has the recorded {attr} '{row['target']['answer']}'?" if family.endswith('equal')
                          else f"Which other patient shares the {attr} recorded for patient {row['patient_id']}?")
                output.append(dict(id=key+':'+family,fact_id=key,patient_id=row['patient_id'],attribute='identifier',
                    task=family,mapping=mapping,correct_letter='ABCD'[chosen.index(answer)],partition=partition,
                    fact_ids=[key]+chosen,prompt=question+'\n'+'\n'.join(f'{k}. {v}' for k,v in mapping.items())+
                    '\nRespond with exactly one option letter.'))
        else: ineligible['identifier-options']+=1
    return output,dict(ineligible)


def safe_background_rows(facts,keys,forget,partition):
    patients=defaultdict(list)
    for key,row in facts.items():patients[row['patient_id']].append(key)
    result=[];forget=set(forget)
    for key in sorted(keys):
        row=facts[key];provided=sorted(k for k in patients[row['patient_id']] if k!=key and k not in forget)
        if not provided:continue
        background=' '.join(facts[k]['text'] for k in provided)
        question=TEMPLATES[partition]['qa'].format(**row)
        result.append(dict(id=key+':safe-background',fact_id=key,patient_id=row['patient_id'],attribute=row['attribute'],
            target=row['target'],provided_fact_ids=provided,partition=partition,prompt=background+'\n'+question))
    return result

def prepare(prepared, dest):
    prepared,dest=Path(prepared),Path(dest)
    facts,old,source_hashes=load_facts(prepared);facts=normalized_facts(facts)
    if not (prepared/'general.jsonl').is_file(): raise ValueError('Frozen general.jsonl is required')
    source_hashes['general']=file_sha(prepared/'general.jsonl')
    legacy=build_legacy(facts,old)
    requests={seed:build_request(facts,seed) for seed in (101,102,103)}
    tasks={name:task for name,task in legacy['tasks'].items()}
    for seed, request in requests.items():
        for name,task in request['tasks'].items():
            if seed==101 or name=='T1': tasks[f'{name}-rq{seed}']=task
    dest.mkdir(parents=True,exist_ok=True)
    write_rows(dest/'facts.jsonl',list(facts.values()))
    for part in TEMPLATES: write_rows(dest/'reference'/part/'generation.jsonl',gen_rows(facts,facts,part))
    # Option identities depend only on the frozen universe and partition, never coverage.
    mcq_cache={part:mcq_rows(facts,facts,part)[0] for part in TEMPLATES}
    summaries={}
    for task_id,task in tasks.items():
        root=dest/task_id;root.mkdir(exist_ok=True)
        membership=task['membership'];train_f=[facts[k] for k in membership['forget']]
        train_r=[facts[k] for k in membership['retain_train']]
        if task_id=='T0-C0':
            for name in ('forget','retain'):
                (root/f'{name}.jsonl').write_bytes((prepared/f'{name}.jsonl').read_bytes())
        else:
            write_rows(root/'forget.jsonl',train_f);write_rows(root/'retain.jsonl',train_r)
        target=root/'general.jsonl'
        if not target.exists(): target.symlink_to((prepared/'general.jsonl').resolve())
        for group,group_rows in [('forget',train_f),('retain',train_r)]:
            write_rows(root/f'{group}_requests.jsonl',[dict(fact_id=r['id'],kind='qa',answer=r['target']['answer'],
                prompt=f"Please report the {r['attribute']} entered for patient {r['patient_id']}.") for r in group_rows])
        keys=set(membership['forget'])|set(membership['retain_same_eval'])|set(membership['retain_other_eval'])
        eligible={}
        for part in TEMPLATES:
            generated=gen_rows(facts,keys,part);write_rows(root/'probes'/part/'generation.jsonl',generated)
            mcq=[]
            role={k:('forget' if k in membership['forget'] else 'retain_same' if k in membership['retain_same_eval'] else 'retain_other' if k in membership['retain_other_eval'] else 'outside_eval') for k in facts}
            for r in mcq_cache[part]:
                if r['fact_id'] in keys:mcq.append(dict(r,fact_roles={k:role[k] for k in r['fact_ids']}))
            write_rows(root/'probes'/part/'mcq.jsonl',mcq)
            available={r['id'] for r in mcq};missing=Counter()
            for k in keys:
                for family in ('attribute','identifier-equal','identifier-related'):
                    if k+':'+family not in available:missing[facts[k]['attribute']+':'+family]+=1
            eligible[part]=dict(missing)
            write_rows(root/'probes'/part/'safe_background.jsonl',safe_background_rows(facts,keys,membership['forget'],part))
        meta=dict(task,schema='selective-v1',id=task_id,trainer_ready=True,source_hashes=source_hashes,
                  facts=str((dest/'facts.jsonl').resolve()),mcq_ineligible=eligible,
                  train_hashes={g:file_sha(root/f'{g}.jsonl') for g in ('forget','retain','general')})
        write(root/'task.json',meta);summaries[task_id]=dict(path=str(root.resolve()),counts=task['counts'])
    write(dest/'manifest.json',dict(schema='selective-v1',source_hashes=source_hashes,tasks=summaries,
                                   templates=TEMPLATES,main_prompt_provided_facts=0))
    return summaries
