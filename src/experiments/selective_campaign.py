"""Three dynamic PBS workers; fixed historical selective experiments precede grids."""
import argparse
import copy
import getpass
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time
from conrep.night import campaign as n
from conrep.night.io import read,write,sha,file_sha,locked
from .selective_common import training_identity,complete_checkpoint,latest_checkpoint


def verify(campaign,plan):
    from .selective_prepare import environment
    if plan['workers']!=3 or plan['world_size']!=8 or not plan['fixed_first']:raise ValueError('Unexpected campaign policy')
    for name,expected in read(campaign/'source.json')['files'].items():
        if file_sha(campaign/'code'/name)!=expected:raise ValueError('Frozen code changed: '+name)
    for name,expected in read(campaign/'inputs.json').items():
        if file_sha(name)!=expected:raise ValueError('Frozen input changed: '+name)
    for task in plan['tasks']:
        if file_sha(task['config'])!=task['config_hash']:raise ValueError('Frozen config changed: '+task['id'])
    for name,expected in read(campaign/'model-assets.json').items():
        p=Path(name)
        if p.stat().st_size!=expected['size'] or p.stat().st_mtime_ns!=expected['mtime_ns']:raise ValueError('SFT asset changed')
    if environment()!=plan['environment']:raise ValueError('Environment changed from prepare')


def ready(plan,state):
    eligible=[]
    for task in plan['tasks']:
        if not n.ready_to_claim(state['tasks'][task['id']]):continue
        if task['kind']!='reference':
            reference=state['tasks'][task['model']+'-sft-reference']
            if reference['status']!='completed':continue
        eligible.append(task)
    # Stage barriers are based on pending/running work, excluding explicitly failed work.
    phases=[t['stage'] for t in plan['tasks'] if state['tasks'][t['id']]['status'] in ('running','pending','paused','interrupted')
            and (t['kind']=='reference' or state['tasks'][t['model']+'-sft-reference']['status']=='completed')]
    phase=min(phases,default=999)
    return [t for t in eligible if t['stage']==phase]


def reserved_peer_slots(plan,active):
    reserved=0
    for name in plan.get('peers',[]):
        root=Path(name)
        if not (root/'state.json').is_file() or (root/'COMPLETE.json').exists() or (root/'STOP').exists():continue
        peer=read(root/'state.json')
        if peer.get('job_id') in active:continue
        if peer.get('status') not in ('blocked','completed','stopped','failed'):
            reserved+=1
    return reserved


def free_slots(plan,state,active):
    if any(w['status']=='submitting' and not w.get('job_id') for w in state['workers'].values()):return 0
    unseen={w['job_id'] for w in state['workers'].values() if w.get('job_id') and w['job_id'] not in active and w.get('reconciled_job')!=w['job_id']}
    reserve=reserved_peer_slots(plan,active)
    return max(0,min(3-len(active)-len(unseen)-reserve,
                     3-sum(n.allocated_nodes(j) for j in active.values())-len(unseen)-reserve))


def submit_available(campaign,plan):
    with locked(Path(plan['project_root'])/'logs/0390/jobs/submit.lock'):
        active=n.active_jobs(n.qstat(),getpass.getuser());state=read(campaign/'state.json')
        available=free_slots(plan,state,active)
        busy=sum(bool(w.get('job_id') in active or w.get('job_id') and w.get('reconciled_job')!=w['job_id']) for w in state['workers'].values())
        running=sum(t['status']=='running' for t in state['tasks'].values())
        available=min(available,max(0,len(ready(plan,state))+running-busy))
        if available<=0:return
        for worker in range(3):
            state=read(campaign/'state.json');w=state['workers'][str(worker)]
            if state['status']!='running' or not ready(plan,state):return
            if w['status'] not in ('new','available') or w.get('job_id') in active or time.time()<w.get('retry_after',0):continue
            attempt=w['allocations']+1;name=f"0390_s{sha(str(campaign))[:3]}{worker}{attempt:03d}"
            path=campaign/'jobs'/(name+'.pbs');path.parent.mkdir(exist_ok=True)
            path.write_text(n.render_pbs(plan,worker,name))
            with n.state_transaction(campaign) as current:
                current['workers'][str(worker)].update(status='submitting',job_id=None,job_name=name,allocations=attempt,
                                                       submission_script=str(path),planned_stop=False)
            result=subprocess.run(['qsub',str(path)],cwd=plan['project_root'],capture_output=True,text=True,timeout=30)
            job=result.stdout.strip()
            if result.returncode or not re.fullmatch(r'\d+(?:\.[A-Za-z0-9_.-]+)?',job):
                n.event(campaign,event='qsub-ambiguous',stdout=result.stdout,stderr=result.stderr);return
            with n.state_transaction(campaign) as current:
                w=current['workers'][str(worker)]
                if w['status']=='submitting':w.update(status='queued',job_id=job)
                elif w.get('job_id')!=job:raise RuntimeError('Worker/PBS identity mismatch')
            n.event(campaign,event='submitted',worker=worker,job_id=job)
            available-=1
            if available<=0:break


def checkpoints(cfg):return n.expected_checkpoints(cfg)


def trained(task,cfg):
    try:
        root=Path(cfg['run']['output_dir']);m=read(root/'TRAINING_COMPLETE.json')
        return m['identity']==task['identity'] and m['steps']==cfg['unlearn']['max_steps'] and all(
            complete_checkpoint(root/f'checkpoint-{s}',identity=task['identity'],world=8) for s in checkpoints(cfg))
    except (OSError,ValueError,KeyError):return False


def evaluations(task,cfg):
    if task['kind']=='reference':
        return [(p,cfg['unlearn']['checkpoint'],Path(cfg['selective']['reference_dir'])/p,True) for p in ('validation','test')]
    items=[('validation',str(Path(cfg['run']['output_dir'])/f'checkpoint-{s}'),Path(task['output'])/'validation'/f'checkpoint-{s}',False) for s in checkpoints(cfg)]
    if task['test_terminal']:
        selection=Path(task['output'])/'test-selection.json'
        steps=set(read(selection)['steps']) if selection.exists() else {cfg['unlearn']['max_steps']}
        for s in sorted(steps):items.append(('test',str(Path(cfg['run']['output_dir'])/f'checkpoint-{s}'),Path(task['output'])/'test'/f'checkpoint-{s}',False))
    return items


def freeze_test_selection(task,cfg):
    """Choose all test weights using validation only, before opening test outputs."""
    source={};reports={};selected=[]
    for step in checkpoints(cfg):
        path=Path(task['output'])/'validation'/f'checkpoint-{step}'/'metrics.json'
        source[str(step)]=file_sha(path);reports[step]=read(path)['metrics']
    for axis in ('retain_same','retain_other'):
        for percent in range(0,101,5):
            eligible=[step for step,m in reports.items() if m[axis+'.any_expression'] is not None and m['forget.any_expression']<=percent/100]
            chosen=min(eligible,key=lambda step:(-reports[step][axis+'.any_expression'],reports[step]['forget.any_expression'],step)) if eligible else None
            selected.append(dict(axis=axis,forget_ceiling=percent/100,step=chosen))
    value=dict(validation_sha256=source,choices=selected,steps=sorted({cfg['unlearn']['max_steps']}|{r['step'] for r in selected if r['step'] is not None}),
               policy='Endpoint plus matched-forgetting 0..100 by 5pp; validation only')
    dest=Path(task['output'])/'test-selection.json'
    if dest.exists() and read(dest)!=value:raise ValueError('Frozen test selection changed')
    write(dest,value)


def evaluate_all(campaign,plan,task,cfg,end,interrupted):
    from .selective_eval import valid
    items=evaluations(task,cfg)
    # References have no checkpoint selection. New fixed runs finish validation first.
    todo=[item for item in items if (task['kind']=='reference' or item[0]=='validation') and not valid(item[2],cfg,item[1],item[0],item[3])]
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2,3,4,5,6,7').split(',')
    active={};failed=[]
    try:
        selected=task['kind']=='reference' or not task['test_terminal']
        while todo or active or not selected:
            if interrupted() or time.time()>end-100:return 75
            if not todo and not active and not selected:
                if failed:raise RuntimeError('Validation incomplete; test selection remains blocked')
                freeze_test_selection(task,cfg);selected=True
                todo=[item for item in evaluations(task,cfg) if item[0]=='test' and not valid(item[2],cfg,item[1],item[0],item[3])]
            for i in range(8):
                if i in active or not todo:continue
                part,checkpoint,out,reference=todo.pop(0);out.mkdir(parents=True,exist_ok=True)
                env=os.environ.copy()
                for key in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):env.pop(key,None)
                env['CUDA_VISIBLE_DEVICES']=devices[i]
                command=[plan['python'],plan['entry'],'evaluate','--config',task['config'],'--checkpoint',checkpoint,'--output',str(out),'--partition',part]
                if reference:command+=['--reference']
                stream=(out/'console.log').open('a');proc=subprocess.Popen(command,cwd=plan['project_root'],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                active[i]=(proc,stream,(part,checkpoint,out,reference))
            for i,(proc,stream,item) in list(active.items()):
                if proc.poll() is not None:
                    stream.close();del active[i]
                    if proc.returncode or not valid(item[2],cfg,item[1],item[0],item[3]):failed.append(str(item[2]))
            time.sleep(1)
        if failed:raise RuntimeError('Evaluation failed: '+', '.join(failed))
        return 0
    finally:
        for proc,stream,_ in active.values():
            n.terminate(proc)
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:n.terminate(proc,signal.SIGKILL);proc.wait()
            stream.close()


def smoke(campaign,plan,task,cfg,end,interrupted):
    # Shared shape/method smoke; protected by a lock across workers.
    key=sha([task['model'],task['variant']])[:16];root=campaign/'smoke'/key
    with locked(campaign/('smoke-'+key+'.lock')):
        if (root/'PASSED.json').exists():return
        c=copy.deepcopy(cfg);c['unlearn'].update(max_steps=2,save_steps=1);c['run']['output_dir']=str(root/'training')
        file=root/'config.json';write(file,c);identity=training_identity(c)
        marker=root/'training/TRAINING_COMPLETE.json'
        stop=root/'STOP';stop.unlink(missing_ok=True)
        previous=latest_checkpoint(root/'training',identity=identity,world=8)
        if previous is None:
            rc=n.run_child(n.train_command(plan,file,deadline=end-120,stop_file=stop,stop_after=1),root/'first.log',
                project=plan['project_root'],deadline=end,stop_file=stop,interrupted=interrupted,pause_marker=root/'training/PAUSED.json')
            if rc!=75:raise RuntimeError('Native smoke checkpoint step failed')
            previous=latest_checkpoint(root/'training',identity=identity,world=8)
        if previous is None:raise RuntimeError('Missing native smoke checkpoint')
        if not marker.exists():
            rc=n.run_child(n.train_command(plan,file,resume=previous,deadline=end-120,stop_file=stop),root/'resume.log',
                project=plan['project_root'],deadline=end,stop_file=stop,interrupted=interrupted,pause_marker=root/'training/PAUSED.json')
            if rc:raise RuntimeError('Native smoke resume failed')
        if not complete_checkpoint(root/'training/checkpoint-2',identity=identity,world=8):raise RuntimeError('Incomplete native smoke weights')
        write(root/'PASSED.json',dict(identity=identity,world_size=8,resume_from=1,final_step=2))


def worker(campaign,worker_id):
    import torch
    campaign=Path(campaign);plan=read(campaign/'plan.json');verify(campaign,plan)
    if worker_id not in range(3) or not torch.cuda.is_available() or torch.cuda.device_count()!=8:raise ValueError('Need one allocated 8-GPU node')
    stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    interrupted=lambda:stopped[0] or (campaign/'STOP').exists()
    end=time.time()+n.seconds(plan['walltime'])-60
    with n.state_transaction(campaign) as state:
        state['workers'][str(worker_id)].update(status='running',job_id=os.environ['PBS_JOBID'],pid=os.getpid(),started_at=time.time())
    while not interrupted() and time.time()<end-300:
        plan=read(campaign/'plan.json')
        if shutil.disk_usage(campaign).free<plan['reserve_bytes']:
            with n.state_transaction(campaign) as state:state['status']='storage_paused'
            break
        with n.state_transaction(campaign) as state:
            if state['status']!='running':break
            choices=sorted(ready(plan,state),key=lambda t:(t['preferred_worker']!=worker_id,t['id']))
            if not choices:break
            task=choices[0];record=state['tasks'][task['id']]
            reserved=sum(t.get('storage',{}).get('additional_bytes',0) for t in plan['tasks'] if state['tasks'][t['id']]['status']=='running')
            required=task['storage']['additional_bytes']+reserved+plan['reserve_bytes']
            free=shutil.disk_usage(campaign).free
            if free<required:
                state.update(status='storage_paused',storage=dict(free_bytes=free,required_bytes=required,task=task['id']))
                break
            record.update(status='running',worker=worker_id,job_id=os.environ['PBS_JOBID'],attempts=record['attempts']+1,claimed_at=time.time())
            record.setdefault('job_ids',[]).append(os.environ['PBS_JOBID']);state['workers'][str(worker_id)]['task']=task['id']
        cfg=read(task['config']);root=Path(task['output']);root.mkdir(parents=True,exist_ok=True);begin=time.time();rc=0;error=None
        pause=root/'PAUSE_TRAINING';pause.unlink(missing_ok=True)
        try:
            if task['kind']=='train' and not trained(task,cfg):
                from .selective_calibrate import valid as calibrated
                if not calibrated(cfg):
                    calibration_start=time.time()
                    command=[plan['python'],plan['entry'],'calibrate','--config',task['config']]
                    rc=n.run_child(command,root/'calibration.log',project=plan['project_root'],deadline=end,stop_file=pause,interrupted=interrupted)
                    write(root/'calibration-time.json',dict(seconds=time.time()-calibration_start,exit_code=rc))
                    if rc:raise RuntimeError('Calibration failed; inspect calibration.log')
                smoke(campaign,plan,task,cfg,end,interrupted)
                previous=latest_checkpoint(cfg['run']['output_dir'],identity=task['identity'],world=8)
                rc=n.run_child(n.train_command(plan,task['config'],resume=previous,deadline=end-120,stop_file=pause),root/'training.log',
                    project=plan['project_root'],deadline=end,stop_file=pause,interrupted=interrupted,
                    pause_marker=Path(cfg['run']['output_dir'])/'PAUSED.json')
                if rc==0 and not trained(task,cfg):raise RuntimeError('Trainer missing planned checkpoint outputs')
            if rc==0:rc=evaluate_all(campaign,plan,task,cfg,end,interrupted)
        except Exception as exc:rc=75 if interrupted() or time.time()>end-200 else 1;error=repr(exc)
        with n.state_transaction(campaign) as state:
            record=state['tasks'][task['id']];record.update(status=('interrupted' if stopped[0] else 'paused') if rc==75 else 'completed' if rc==0 else 'failed',
                exit_code=rc,error=error,elapsed_seconds=record.get('elapsed_seconds',0)+time.time()-begin)
            if rc==0:record['output_bytes']=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
            state['workers'][str(worker_id)]['task']=None
            if rc==0:state['workers'][str(worker_id)].update(bounded_failures=0,recovery_failures=0)
        n.event(campaign,event='task-end',task=task['id'],exit_code=rc,error=error)
        if rc==75:break
    with n.state_transaction(campaign) as state:state['workers'][str(worker_id)].update(status='interrupted' if stopped[0] else 'drained',ended_at=time.time())
    return 75 if stopped[0] else 0


def watch(campaign):
    from .selective_grid import advance
    with locked(campaign/'supervisor.lock',blocking=False):
        write(campaign/'supervisor.json',dict(pid=os.getpid(),started=time.time()))
        while not (campaign/'STOP').exists():
            try:
                plan=read(campaign/'plan.json');state=read(campaign/'state.json')
                if state['status']!='running':return
                active=n.active_jobs(n.qstat(),getpass.getuser());n.reconcile(campaign,plan,active)
                if advance(campaign):plan=read(campaign/'plan.json')
                state=read(campaign/'state.json')
                if all(t['status']=='completed' for t in state['tasks'].values()) and plan['grid_phase']==3:
                    with n.state_transaction(campaign) as current:current['status']='completed'
                    write(campaign/'COMPLETE.json',dict(time=time.time(),tasks=len(plan['tasks'])));return
                owned=any(w.get('job_id') in active or w['status']=='submitting' or w.get('job_id') and w.get('reconciled_job')!=w['job_id'] for w in state['workers'].values())
                waiting=any(t['status']=='paused' and t.get('retry_after',0)>time.time() for t in state['tasks'].values())
                blocked_workers=all(w['status'] in n.BLOCKED_WORKER_STATES for w in state['workers'].values())
                if not owned and not waiting and (not ready(plan,state) or blocked_workers):
                    with n.state_transaction(campaign) as current:current['status']='blocked'
                    write(campaign/'BLOCKED.json',dict(reason='Fixed selective task/evaluation incomplete; grid remains gated',
                        tasks={k:v for k,v in state['tasks'].items() if v['status']!='completed'}));return
                submit_available(campaign,plan)
                write(campaign/'heartbeat.json',dict(time=time.time(),pid=os.getpid(),ready=len(ready(plan,state)),
                    account_active=len(active),peer_reserved=reserved_peer_slots(plan,active)))
            except Exception as exc:n.event(campaign,event='supervisor-error',error=repr(exc))
            time.sleep(30)


def arm(campaign,resume=False,retry_failed=False,requeue_jobs=()):
    plan=read(campaign/'plan.json');verify(campaign,plan)
    with locked(campaign/'start.lock'):
        try:
            with locked(campaign/'supervisor.lock',blocking=False):pass
        except BlockingIOError:
            if requeue_jobs or retry_failed:raise ValueError('Use stop and wait for supervisor exit before explicit recovery')
            print('Supervisor already running; no duplicate submission');return
        if (campaign/'STOP').exists() and not resume:raise ValueError('STOP preserved; use resume')
        active=n.active_jobs(n.qstat(),getpass.getuser());n.reconcile(campaign,plan,active)
        if requeue_jobs:n.requeue_explicit_jobs(campaign,requeue_jobs,active)
        with n.state_transaction(campaign) as state:
            if state['status']=='completed':print('Campaign already complete');return
            if retry_failed:
                for task in state['tasks'].values():
                    if task['status']=='failed':task.update(status='pending',error=None)
            for w in state['workers'].values():
                if w['status']=='drained':w['status']='available'
            state.update(status='running',started_at=state.get('started_at') or time.time(),deadline=253402300799)
        if resume:(campaign/'STOP').unlink(missing_ok=True)
        with (campaign/'supervisor.log').open('a') as log:
            p=subprocess.Popen([plan['python'],'-I',plan['entry'],'watch','--campaign',str(campaign)],cwd=plan['project_root'],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        time.sleep(.5)
        if p.poll() is not None:raise RuntimeError('Supervisor did not start; inspect supervisor.log')
        print(f'Supervisor PID={p.pid}; dynamic selective workers 0–3; total account cap 3.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','arm','start','watch','worker','status','resume','stop','train','evaluate','calibrate','export'])
    p.add_argument('--campaign',type=Path);p.add_argument('--worker',type=int)
    p.add_argument('--config');p.add_argument('--checkpoint');p.add_argument('--output');p.add_argument('--resume')
    p.add_argument('--partition',default='validation',choices=['validation','test']);p.add_argument('--reference',action='store_true')
    p.add_argument('--deadline',type=float);p.add_argument('--stop-file');p.add_argument('--stop-after-step',type=int)
    p.add_argument('--retry-failed',action='store_true');p.add_argument('--include-predictions',action='store_true')
    p.add_argument('--requeue-job',action='append',default=[])
    a=p.parse_args();campaign=a.campaign.resolve() if a.campaign else None
    if a.command=='prepare':
        from .selective_prepare import prepare
        prepare(campaign)
    elif a.command in ('arm','start','resume'):arm(campaign,a.command=='resume',a.retry_failed,a.requeue_job)
    elif a.command=='watch':watch(campaign)
    elif a.command=='worker':return worker(campaign,a.worker)
    elif a.command=='stop':(campaign/'STOP').touch();print('Graceful stop requested; no unrelated jobs cancelled')
    elif a.command=='status':
        plan=read(campaign/'plan.json');state=read(campaign/'state.json')
        from collections import Counter
        elapsed=sum(t.get('elapsed_seconds',0) for t in state['tasks'].values())
        measured=[t['elapsed_seconds'] for t in state['tasks'].values() if t['status']=='completed' and t.get('elapsed_seconds')]
        pending=sum(t['status']!='completed' for t in state['tasks'].values())
        print(json.dumps(dict(status=state['status'],tasks=dict(Counter(t['status'] for t in state['tasks'].values())),
            workers=state['workers'],grid_phase=plan['grid_phase'],heartbeat=read(campaign/'heartbeat.json') if (campaign/'heartbeat.json').exists() else None,
            node_hours=elapsed/3600,remaining_node_hours_rough=(pending*sum(measured)/len(measured)/3600 if measured else None),
            eta_caveat='Rolling estimate; methods, calibration and target sizes differ; excludes ungenerated later grid',
            disk_free_bytes=shutil.disk_usage(campaign).free,storage_block=state.get('storage'),
            failures={k:v.get('error',v.get('failure_reason')) for k,v in state['tasks'].items() if v['status'] in ('failed','cancelled')}),indent=2))
    elif a.command=='train':
        from .selective_train import run
        result=run(read(a.config),a.resume,deadline=a.deadline,stop_file=a.stop_file,stop_after_step=a.stop_after_step)
        # torchrun treats any nonzero rank exit as failure; PAUSED.json is the pause protocol.
        return 0 if result==75 else result
    elif a.command=='evaluate':
        from .selective_eval import run
        run(read(a.config),a.checkpoint,a.output,a.partition,a.reference)
    elif a.command=='calibrate':
        from .selective_calibrate import run
        run(read(a.config))
    elif a.command=='export':
        from .selective_export import export
        export(campaign,a.include_predictions)
    return 0
