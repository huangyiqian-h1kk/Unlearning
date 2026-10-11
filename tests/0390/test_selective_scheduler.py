"""Dispatch regressions use the real 300-task topology and mocked PBS, not GPUs."""
import copy
import importlib.util
import io
import contextlib
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from conrep.night.io import read, write, sha, file_sha
from experiments import selective_campaign as c
from experiments import selective_scheduler as scheduler
from experiments.selective_prepare import base_specs

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('selective_upgrade',ROOT/'scripts/abci/0390_upgrade_selective_scheduler.py')
upgrade=importlib.util.module_from_spec(spec);spec.loader.exec_module(upgrade)


def campaign(root):
    plan=dict(project_root=str(root),campaign=str(root),walltime='12:00:00',shell='worker.sh',
              source_hash='science',peers=[],grid_phase=0,tasks=[],reserve_bytes=0)
    for model in ('gemma2_9b','llama8b'):
        plan['tasks'].append(dict(id=model+'-sft-reference',kind='reference',model=model,stage=0))
        plan['tasks'].append(dict(id=model+'-bridge',kind='historical',model=model,stage=5))
    for i,(model,variant,method,target,seed,stage,changes) in enumerate(base_specs()):
        plan['tasks'].append(dict(id=f'train-{i}',kind='train',model=model,stage=stage))
    state=dict(status='running',tasks={t['id']:dict(status='pending',attempts=0) for t in plan['tasks']},
               workers={str(i):dict(status='new',job_id=None,allocations=0) for i in range(3)})
    write(root/'plan.json',plan);write(root/'state.json',state)
    return plan,state


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_actual_300_task_topology_allows_third_training_before_references(self):
        plan,state=campaign(self.root)
        self.assertEqual(len(plan['tasks']),300)
        ready=c.ready(plan,state)
        self.assertEqual(len(ready),298)
        self.assertEqual(sum(t['kind']=='train' for t in ready),296)
        ordered=sorted(ready,key=lambda t:scheduler.task_order(t,2))
        self.assertEqual([t['kind'] for t in ordered[:3]],['reference','reference','train'])
        self.assertEqual(ordered[2]['stage'],10)

    def test_two_queued_legacy_workers_get_exactly_one_new_worker(self):
        plan,state=campaign(self.root);active={f'{i}.pbs':{'Resource_List':{'select':'1'}} for i in (10,11)}
        for i,job in enumerate(active):
            state['workers'][str(i)].update(status='queued',job_id=job,allocations=2)
        write(self.root/'state.json',state)
        calls=[]
        def submit(*args,**kwargs):
            job='12.pbs';calls.append(args);active[job]={'Resource_List':{'select':'1'}}
            return SimpleNamespace(returncode=0,stdout=job,stderr='')
        with patch.object(c.n,'qstat',return_value=active),patch.object(c.n,'active_jobs',side_effect=lambda v,u:v),patch.object(c.subprocess,'run',side_effect=submit):
            c.submit_available(self.root,plan);c.submit_available(self.root,plan)
        self.assertEqual(len(calls),1)
        self.assertEqual(read(self.root/'state.json')['workers']['2']['job_id'],'12.pbs')
        text=next((self.root/'jobs').glob('*.pbs')).read_text()
        self.assertIn('#PBS -q R9920261000',text)
        self.assertIn('#PBS -P gcg51557',text)
        self.assertIn('#PBS -l select=1',text)
        self.assertIn('#PBS -N 0390_',text)

    def test_sft_slot_released_and_account_node_caps_remain_three(self):
        plan,state=campaign(self.root)
        active={f'{i}.pbs':{'Resource_List':{'select':'1'}} for i in range(3)}
        self.assertEqual(c.free_slots(plan,state,active),0)
        del active['2.pbs'];self.assertEqual(c.free_slots(plan,state,active),1)
        active={'multi.pbs':{'Resource_List':{'select':'3'}}}
        self.assertEqual(c.free_slots(plan,state,active),0)
        peer=self.root/'qwen';write(peer/'state.json',dict(status='waiting_retry',job_id=None))
        plan['peers']=[str(peer)]
        self.assertEqual(c.free_slots(plan,state,{}),2)
        write(peer/'COMPLETE.json',{'done':True})
        self.assertEqual(c.free_slots(plan,state,{}),3)

    def test_failed_reference_blocks_its_model_but_not_the_other_model(self):
        plan,state=campaign(self.root)
        state['tasks']['gemma2_9b-sft-reference']['status']='failed'
        ready=c.ready(plan,state)
        self.assertTrue(ready)
        self.assertTrue(all(t['model']=='llama8b' for t in ready))
        state['tasks']['gemma2_9b-sft-reference']['status']='completed'
        self.assertIn('gemma2_9b-bridge',{t['id'] for t in c.ready(plan,state)})

    def test_grid_waits_for_all_fixed_results_and_each_previous_grid(self):
        plan,state=campaign(self.root)
        for stage in (100,101):
            plan['tasks'].append(dict(id=f'grid-{stage}',stage=stage,kind='train',model='llama8b'))
            state['tasks'][f'grid-{stage}']=dict(status='pending')
        self.assertFalse(any(t['stage']>=100 for t in c.ready(plan,state)))
        for t in plan['tasks']:
            if t['stage']<100:state['tasks'][t['id']]['status']='completed'
        self.assertEqual([t['id'] for t in c.ready(plan,state)],['grid-100'])
        state['tasks']['grid-100']['status']='failed'
        self.assertFalse(c.ready(plan,state))
        state['tasks']['grid-100']['status']='completed'
        self.assertEqual([t['id'] for t in c.ready(plan,state)],['grid-101'])

    def test_deferred_training_waits_for_only_its_own_reference(self):
        plan,state=campaign(self.root)
        task=next(t for t in plan['tasks'] if t['kind']=='train')
        record=state['tasks'][task['id']]
        scheduler.record_result(record,0,deferred=True,elapsed=5)
        self.assertEqual(record['status'],'pending')
        self.assertNotIn(task,c.ready(plan,state))
        state['tasks'][task['model']+'-sft-reference']['status']='completed'
        self.assertIn(task,c.ready(plan,state))
        scheduler.record_result(record,0)
        self.assertEqual(record['status'],'completed')
        self.assertFalse(record[scheduler.WAITING])

    def test_ambiguous_submission_or_qstat_failure_never_creates_extra_jobs(self):
        plan,state=campaign(self.root)
        with patch.object(c.n,'qstat',side_effect=RuntimeError('PBS down')),patch.object(c.subprocess,'run') as submit:
            with self.assertRaises(RuntimeError):c.submit_available(self.root,plan)
            submit.assert_not_called()
        with patch.object(c.n,'qstat',return_value={}),patch.object(c.n,'active_jobs',return_value={}),patch.object(c.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout='',stderr='ambiguous')) as submit:
            c.submit_available(self.root,plan);c.submit_available(self.root,plan)
            self.assertEqual(submit.call_count,1)

    def test_worker_finishes_training_then_defers_and_reuses_weights(self):
        plan,state=campaign(self.root)
        task=next(t for t in plan['tasks'] if t['kind']=='train')
        ref=next(t for t in plan['tasks'] if t['id']==task['model']+'-sft-reference')
        plan['tasks']=[ref,task]
        task.update(preferred_worker=2,storage={'additional_bytes':0},output=str(self.root/'experiment'),config=str(self.root/'config.json'),identity='fixture')
        cfg={'baseline':{'method':'conrep'},'run':{'output_dir':str(self.root/'weights')}}
        write(task['config'],cfg)
        state['tasks']={ref['id']:dict(status='running',attempts=1),task['id']:dict(status='pending',attempts=0)}
        write(self.root/'plan.json',plan);write(self.root/'state.json',state)
        training=[];evaluation=[]
        def train(*args,**kwargs):training.append(True);return 0
        fake_torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,device_count=lambda:8))
        with patch.dict(sys.modules,{'torch':fake_torch}),patch.dict(c.os.environ,{'PBS_JOBID':'third.pbs'}),patch.object(c,'verify'),patch.object(c.signal,'signal'),patch.object(c,'smoke'),patch.object(c,'latest_checkpoint',return_value=None),patch.object(c,'trained',side_effect=lambda *a:bool(training)),patch.object(c.n,'train_command',return_value=['train']),patch.object(c.n,'run_child',side_effect=train),patch.object(c,'evaluate_all',side_effect=lambda *a:evaluation.append(True) or 0):
            self.assertEqual(c.worker(self.root,2),0)
            after=read(self.root/'state.json')
            self.assertTrue(after['tasks'][task['id']][scheduler.WAITING])
            self.assertEqual(after['tasks'][task['id']]['status'],'pending')
            self.assertEqual(len(training),1);self.assertEqual(len(evaluation),0)
            after['tasks'][ref['id']]['status']='completed';write(self.root/'state.json',after)
            self.assertEqual(c.worker(self.root,2),0)
        self.assertEqual(len(training),1);self.assertEqual(len(evaluation),1)
        self.assertEqual(read(self.root/'state.json')['tasks'][task['id']]['status'],'completed')

    def test_overlay_cannot_change_training_or_capacity_code(self):
        source=(ROOT/upgrade.CONTROLLER).read_text()
        upgrade.checked_overlay(source,source)
        with self.assertRaises(ValueError):
            upgrade.checked_overlay(source,source.replace('return max(0,min(', 'return max(1,min('))
        with self.assertRaises(ValueError):
            upgrade.checked_overlay(source,source.replace("if rc==0 and not trained(task,cfg):", "if rc==0 and trained(task,cfg):"))

    def test_controller_plan_preserves_every_task_and_scientific_identity(self):
        plan,state=campaign(self.root);before=copy.deepcopy(plan)
        controller={'root':str(self.root/'controllers/v2')}
        after=upgrade.controller_plan(plan,controller)
        self.assertEqual(plan,before)
        self.assertEqual(after['tasks'],before['tasks'])
        self.assertEqual(after['source_hash'],before['source_hash'])
        self.assertEqual(after['shell'],str(self.root/'controllers/v2/code/scripts/abci/0390_selective_worker.sh'))

    def handoff_fixture(self):
        plan,state=campaign(self.root)
        base={upgrade.CONTROLLER:b'original controller',
              'scripts/abci/0390_selective.py':b'original entry',
              'scripts/abci/0390_selective_worker.sh':b'original shell',
              'src/experiments/selective_eval.py':b'original evaluator'}
        files={}
        for name,data in base.items():
            path=self.root/'code'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            files[name]=file_sha(path)
        plan['source_hash']=sha(files)
        write(self.root/'source.json',dict(files=files,source_hash=plan['source_hash']))
        folder=self.root/'controllers/fixture';updated={}
        for name in list(base)+[upgrade.POLICY_FILE]:
            data=(ROOT/name).read_bytes() if name in (upgrade.CONTROLLER,upgrade.POLICY_FILE) else base[name]
            path=folder/'code'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            updated[name]=file_sha(path)
        write(folder/'source.json',dict(files=updated,base_source_hash=plan['source_hash']))
        controller=dict(root=str(folder),policy=scheduler.POLICY,source_hash=sha(updated),ref='fixture',
            base_source_hash=plan['source_hash'],manifest_sha256=file_sha(folder/'source.json'))
        write(self.root/'plan.json',plan)
        return plan,state,controller

    def test_live_handoff_keeps_worker_progress_and_original_sources(self):
        plan,state,controller=self.handoff_fixture()
        task=plan['tasks'][0]['id'];state['tasks'][task]['status']='running'
        state['workers']['0'].update(status='running',job_id='existing.pbs',task=task)
        write(self.root/'state.json',state)
        originals={p:p.read_bytes() for p in (self.root/'code').rglob('*') if p.is_file()}
        script='''
import fcntl,json,sys,time
from pathlib import Path
root=Path(sys.argv[1]);task=sys.argv[2]
with (root/'supervisor.lock').open('a+') as supervisor:
    fcntl.flock(supervisor,fcntl.LOCK_EX)
    print('ready',flush=True)
    while json.loads((root/'state.json').read_text())['status']=='running':time.sleep(.01)
    with (root/'state.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads((root/'state.json').read_text())
        state['tasks'][task]['status']='completed'
        (root/'state-child.tmp').write_text(json.dumps(state))
        (root/'state-child.tmp').replace(root/'state.json')
'''
        proc=subprocess.Popen([sys.executable,'-c',script,str(self.root),task],stdout=subprocess.PIPE,text=True)
        try:
            self.assertEqual(proc.stdout.readline().strip(),'ready')
            with patch.object(upgrade.subprocess,'run',side_effect=AssertionError('handoff must not submit/cancel PBS')):
                upgrade.handoff(self.root,controller)
            self.assertEqual(proc.wait(timeout=5),0)
        finally:
            if proc.poll() is None:proc.kill();proc.wait()
            proc.stdout.close()
        after=read(self.root/'state.json')
        self.assertEqual(after['tasks'][task]['status'],'completed')
        self.assertEqual(after['workers']['0']['job_id'],'existing.pbs')
        self.assertEqual(after['status'],'running')
        self.assertTrue(all(p.read_bytes()==data for p,data in originals.items()))
        scheduler.verify_controller(self.root,read(self.root/'plan.json'))
        self.assertEqual(read(self.root/'plan.json')['tasks'],plan['tasks'])
        upgrade.handoff(self.root,controller)
        self.assertEqual(read(self.root/'state.json'),after)

    def test_failed_handoff_restores_state_and_does_not_install_plan(self):
        plan,state,controller=self.handoff_fixture()
        fake=SimpleNamespace(verify_controller=lambda *a:(_ for _ in ()).throw(ValueError('manifest mismatch')))
        with patch.object(upgrade,'load_policy',return_value=fake),self.assertRaisesRegex(ValueError,'manifest mismatch'):
            upgrade.handoff(self.root,controller)
        self.assertEqual(read(self.root/'state.json'),state)
        self.assertEqual(read(self.root/'plan.json'),plan)

    def test_scientific_file_change_in_controller_is_rejected(self):
        plan,state,controller=self.handoff_fixture()
        candidate=upgrade.controller_plan(plan,controller)
        scheduler.verify_controller(self.root,candidate)
        root=Path(controller['root'])
        path=root/'code/src/experiments/selective_eval.py';path.write_text('changed evaluator')
        manifest=read(root/'source.json');manifest['files']['src/experiments/selective_eval.py']=file_sha(path)
        write(root/'source.json',manifest)
        controller.update(source_hash=sha(manifest['files']),manifest_sha256=file_sha(root/'source.json'))
        with self.assertRaisesRegex(ValueError,'Scientific source changed'):
            scheduler.verify_controller(self.root,upgrade.controller_plan(plan,controller))


if __name__=='__main__':
    unittest.main()
