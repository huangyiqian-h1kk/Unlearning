"""Deferred grids: never change or replace fixed historical selective results."""
import copy
from pathlib import Path
from conrep.night.io import read,write,file_sha,locked
from .selective_design import REPLICATION_CANDIDATES
from .selective_common import BASELINES
from .selective_prepare import publish_task


def endpoint(task):
    cfg=read(task['config'])
    return read(Path(task['output'])/'validation'/f"checkpoint-{cfg['unlearn']['max_steps']}"/'metrics.json')['metrics']


def best(tasks):
    def rank(t):
        m=endpoint(t)
        return (-(m['retain_same.any_expression']-m['forget.any_expression']),-m['retain_other.any_expression'],
                -m.get('utility.mmlu',0),t['id'])
    return min(tasks,key=rank)


def advance(campaign):
    campaign=Path(campaign)
    with locked(campaign/'grid.lock'):
        plan=read(campaign/'plan.json');state=read(campaign/'state.json');phase=plan['grid_phase']
        if phase>=3 or not all(state['tasks'][t['id']]['status']=='completed' for t in plan['tasks']):return False
        candidates=[];decisions=[]
        if phase==0:
            for model,variants in REPLICATION_CANDIDATES.items():
                for variant in list(variants)+list(BASELINES):
                    control=next(t for t in plan['tasks'] if t['model']==model and t['variant']==variant and
                                 t['target']=='T1-rq101' and t['seed']==42 and t['stage']==10)
                    for factor in (.5,2.):
                        cfg=read(control['config']);section='baseline' if variant=='lunar' else 'unlearn'
                        field='fit_learning_rate' if variant=='lunar' else 'learning_rate'
                        cfg[section][field]*=factor
                        cfg['selective']['grid']=dict(phase='lr',control=control['id'],factor=factor)
                        candidates.append((cfg,control['id']+f'-grid-lr{factor}',100))
        else:
            for model in REPLICATION_CANDIDATES:
                pool=[t for t in plan['tasks'] if t['model']==model and t['method']=='conrep' and t['target']=='T1-rq101'
                      and t['seed']==42 and (t['stage']==100 or t['stage']==10 and t['variant'] in REPLICATION_CANDIDATES[model])]
                if phase==2:
                    selected=read(campaign/'grid-selection-1.json')['selected'][model]
                    pool=[t for t in plan['tasks'] if t['id']==selected or t['stage']==101 and t['model']==model]
                chosen=best(pool);decisions.append(dict(model=model,selected=chosen['id'],scores={t['id']:endpoint(t) for t in pool}))
                for value in ((1.,3.,5.) if phase==1 else (.5,2.)):
                    cfg=read(chosen['config']);field='forget_cl_weight' if phase==1 else 'specified_lm_weight'
                    cfg['conrep'][field]=value
                    cfg['selective']['grid']=dict(phase=field,control=chosen['id'],value=value)
                    candidates.append((cfg,chosen['id']+f'-grid-{field}{value}',100+phase))
        created=[]
        for cfg,name,stage in candidates:
            t=publish_task(plan,cfg,name,stage,test=False);plan['tasks'].append(t);created.append(t)
        if decisions:write(campaign/f'grid-selection-{phase}.json',dict(selected={r['model']:r['selected'] for r in decisions},decisions=decisions))
        plan['grid_phase']=phase+1
        # Append state first; task files are already immutable before publishing the new plan.
        from conrep.night.campaign import state_transaction
        with state_transaction(campaign) as current:
            for t in created:current['tasks'][t['id']]=dict(status='pending',attempts=0,failures=0)
            ledger_path=campaign/'reuse-ledger.json'
            ledger=read(ledger_path) if ledger_path.exists() else {'tasks':[]}
            known={r['task'] for r in ledger['tasks']}
            for t in created:
                if t['id'] not in known:ledger['tasks'].append(dict(task=t['id'],action='train_new',training_identity=t['identity'],
                    reason='New selective T1 hyperparameters; original setting reused from fixed main run'))
            write(ledger_path,ledger)
            write(campaign/'plan.json',plan)
        return True
