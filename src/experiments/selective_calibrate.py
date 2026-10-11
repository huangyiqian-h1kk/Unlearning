"""Task-bound FALCON calibration and audited ReLearn augmentation."""
import copy
from pathlib import Path
from conrep.night.io import read,write,sha,file_sha,locked
from .selective_data import rows,write_rows


def valid(cfg):
    method=cfg['baseline']['method']
    if method not in ('falcon','relearn'):return True
    path=Path(cfg['baseline']['layer_selection' if method=='falcon' else 'augmented_file'])
    try:
        meta=read(path.parent/'CALIBRATED.json')
        return meta['identity']==cfg['selective']['calibration_identity'] and meta['sha256']==file_sha(path)
    except (OSError,KeyError,ValueError):return False


def run(cfg):
    from .baselines.prepare import falcon_layers,relearn_augment
    from .pmc_scoring_v5 import score_answer
    method=cfg['baseline']['method']
    if method not in ('falcon','relearn'):return
    from .runtime import initialize
    initialize(42)
    dest=Path(cfg['baseline']['layer_selection' if method=='falcon' else 'augmented_file'])
    dest.parent.mkdir(parents=True,exist_ok=True)
    with locked(dest.parent/'calibration.lock'):
        if valid(cfg):return
        if (dest.parent/'CALIBRATED.json').exists():
            raise ValueError('Previously completed calibration changed; refusing to alter training material during resume')
        if method=='falcon':
            report=falcon_layers(cfg,dest)
            if report.get('selected_layer') is None:raise ValueError('No valid FALCON layer')
        else:
            facts={r['id']:r for r in rows(Path(cfg['data']['prepared_dir'])/'forget.jsonl')}
            accepted=[];covered=set();audit=[]
            for index in range(3):
                output=dest.parent/f'round-{index}.jsonl'
                temporary=copy.deepcopy(cfg)
                if index:
                    requests=[]
                    for key in sorted(set(facts)-covered):
                        r=facts[key]
                        text=(f"In the medical records, provide the {r['attribute']} for patient {r['patient_id']}." if index==1 else
                              f"Look up patient {r['patient_id']} and report the entry for {r['attribute']}.")
                        requests.append(dict(fact_id=key,kind='qa',prompt=text,answer=r['target']['answer']))
                    request_file=dest.parent/f'round-{index}-requests.jsonl';write_rows(request_file,requests)
                    temporary['data']['forget_requests']=str(request_file)
                if not output.is_file() or not Path(str(output)+'.audit.json').is_file():relearn_augment(temporary,output)
                for r in rows(output):
                    if r['fact_id'] not in facts:raise ValueError('Foreign augmentation fact')
                    f=facts[r['fact_id']]
                    if not score_answer(r['answer'],f['target'])['correct']:
                        accepted.append(r);covered.add(r['fact_id'])
                audit.append(dict(round=index,covered=len(covered),required=len(facts)))
                if covered==set(facts):break
            # Keep only the first occurrence of an exact QA pair.
            unique={(r['fact_id'],r['prompt'],r['answer']):r for r in accepted}
            accepted=[dict(r,id=f'relearn:{i}') for i,r in enumerate(unique.values())]
            write_rows(dest,accepted)
            write(dest.parent/'coverage.json',dict(rounds=audit,missing=sorted(set(facts)-covered),accepted=len(accepted)))
            if covered!=set(facts):raise ValueError(f'blocked_augmentation: {len(facts)-len(covered)} uncovered facts')
        write(dest.parent/'CALIBRATED.json',dict(identity=cfg['selective']['calibration_identity'],sha256=file_sha(dest)))
