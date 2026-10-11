"""Scoped capacity regressions: real submission loop, synthetic PBS, no GPU."""
import copy
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from conrep.night.io import read, write
from experiments import selective_campaign as api

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('extra_node', ROOT/'scripts/abci/0390_extra_node.py')
extra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extra)


class ExtraNodeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parent, self.child = self.root/'parent', self.root/'child'
        entry = self.child/'controllers/fixture/0390_extra_node.py'
        entry.parent.mkdir(parents=True)
        entry.write_bytes((ROOT/extra.SOURCE).read_bytes())
        self.policy = dict(policy=extra.POLICY, parent=str(self.parent), companion=str(self.child),
            entry=str(entry), entry_sha256=extra.file_sha(entry), limits=dict(parent=3, companion=1, account=4),
            science=dict(parent='parent-science', companion='child-science'))
        self.plans, self.states = {}, {}
        for role, path in (('parent', self.parent), ('companion', self.child)):
            plan = dict(project_root=str(self.root), campaign=str(path), python=sys.executable,
                entry='original-'+role, shell='original-worker.sh', source_hash=role+'-science',
                peers=[], grid_phase=3, tasks=[], walltime='12:00:00')
            plan['source_hash'] = self.policy['science'][role]
            plan[extra.KEY] = self.policy
            plan['tasks'] = [dict(id=str(i), kind='train', target='T0-General-only', stage=10, model='llama8b') for i in range(14)]
            state = dict(status='running', tasks={str(i):dict(status='pending', attempts=0) for i in range(14)},
                         workers={str(i):dict(status='new', job_id=None, allocations=0) for i in range(3)})
            state['tasks']['llama8b-sft-reference'] = dict(status='completed')
            if role == 'companion':
                plan['companion_parent'] = str(self.parent)
                for i in ('1', '2'):
                    state['workers'][i]['status'] = 'deferred_to_parent'
            write(path/'plan.json', plan)
            write(path/'state.json', state)
            self.plans[role], self.states[role] = plan, state

    def persist(self):
        for role in ('parent', 'companion'):
            write(Path(self.plans[role]['campaign'])/'state.json', self.states[role])

    def own(self, role, ids):
        for i, job in enumerate(ids):
            self.states[role]['workers'][str(i)].update(status='queued', job_id=job, allocations=1)
        self.persist()

    def slots(self, role, active):
        return extra.free_slots(self.plans[role], self.states[role], active, api.n.allocated_nodes)

    def jobs(self, *ids, nodes=1):
        return {j:dict(Resource_List={'select':str(nodes)}) for j in ids}

    def test_three_parent_jobs_admit_exactly_one_extra(self):
        self.own('parent', ('1.pbs', '2.pbs', '3.pbs'))
        active = self.jobs('1.pbs', '2.pbs', '3.pbs')
        self.assertEqual(self.slots('companion', active), 1)
        self.assertEqual(self.slots('parent', active), 0)
        self.own('companion', ('4.pbs',))
        self.assertEqual(self.slots('companion', dict(active, **self.jobs('4.pbs'))), 0)

    def test_parent_replenishes_third_beside_companion(self):
        self.own('parent', ('1.pbs', '2.pbs'))
        self.own('companion', ('4.pbs',))
        active = self.jobs('1.pbs', '2.pbs', '4.pbs')
        self.assertEqual(self.slots('parent', active), 1)
        self.assertEqual(self.slots('companion', active), 0)

    def test_fourth_slot_cannot_be_borrowed_by_parent(self):
        self.assertEqual(self.slots('parent', {}), 3)
        self.assertEqual(self.slots('companion', {}), 1)
        self.states['companion']['status'] = 'completed'
        self.persist()
        self.assertEqual(self.slots('parent', {}), 3)
        active = self.jobs('external1', 'external2', 'external3')
        self.assertEqual(self.slots('parent', active), 0)
        self.assertEqual(self.slots('companion', active), 1)
        active['external4'] = self.jobs('external4')['external4']
        self.assertEqual(self.slots('companion', active), 0)

    def test_nodes_and_jobs_both_enforced(self):
        self.assertEqual(self.slots('companion', self.jobs('big', nodes=4)), 0)
        self.assertEqual(self.slots('parent', self.jobs('big', nodes=3)), 0)
        self.own('companion', ('big',))
        self.assertEqual(self.slots('parent', self.jobs('big', nodes=2)), 0)

    def test_unseen_and_ambiguous_submissions_count_on_both_sides(self):
        self.own('parent', ('1.pbs', '2.pbs', '3.pbs'))
        self.own('companion', ('4.pbs',))
        self.assertEqual(self.slots('parent', {}), 0)
        self.assertEqual(self.slots('companion', {}), 0)
        self.states['companion']['workers']['0'].update(status='submitting', job_id=None)
        self.persist()
        self.assertEqual(self.slots('parent', {}), 0)
        self.assertEqual(self.slots('companion', {}), 0)
        self.states['companion']['workers']['0'].update(status='new')
        self.states['parent']['workers']['0'].update(status='submitting', job_id=None)
        self.persist()
        self.assertEqual(self.slots('companion', {}), 0)

    def test_other_peer_reservations_count_once(self):
        peer = self.root/'qwen'
        write(peer/'state.json', dict(status='running', job_id=None))
        for role in self.plans:
            self.plans[role]['peers'] = [str(peer)]
            write(Path(self.plans[role]['campaign'])/'plan.json', self.plans[role])
        self.assertEqual(self.slots('parent', {}), 2)
        self.assertEqual(self.slots('companion', self.jobs('1', '2', '3')), 0)
        write(peer/'COMPLETE.json', {})
        self.assertEqual(self.slots('companion', self.jobs('1', '2', '3')), 1)

    def test_concurrent_real_submit_loops_with_invisible_qsub_never_exceed_four(self):
        calls = []
        def qsub(*args, **kwargs):
            calls.append(args[0])
            return SimpleNamespace(returncode=0, stdout=f'{100+len(calls)}.pbs', stderr='')
        with patch.object(api, 'free_slots', side_effect=lambda p,s,a:extra.free_slots(p,s,a,api.n.allocated_nodes)), \
             patch.object(api.n, 'qstat', return_value={}), \
             patch.object(api.n, 'active_jobs', return_value={}), \
             patch.object(api.subprocess, 'run', side_effect=qsub):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(api.submit_available, Path(p['campaign']), p) for p in self.plans.values()]
                for future in futures:
                    future.result(timeout=10)
            for _ in range(3):
                for p in self.plans.values():
                    api.submit_available(Path(p['campaign']), p)
        self.assertEqual(len(calls), 4)
        self.assertEqual(sum(w['status']=='queued' for w in read(self.parent/'state.json')['workers'].values()), 3)
        self.assertEqual(sum(w['status']=='queued' for w in read(self.child/'state.json')['workers'].values()), 1)
        for script in self.root.rglob('*.pbs'):
            self.assertIn('#PBS -l select=1', script.read_text())

    def test_parent_completion_never_expands_companion_to_three(self):
        self.states['parent']['status'] = 'completed'
        self.persist()
        write(self.parent/'COMPLETE.json', {})
        extra.refresh_companion(self.child, self.plans['companion'])
        after = read(self.child/'state.json')
        self.assertEqual(after['allocation_limit'], 1)
        self.assertEqual(after['workers']['1']['status'], 'deferred_to_parent')

    def test_different_pair_or_changed_launcher_fails_closed(self):
        plan = copy.deepcopy(self.plans['parent'])
        plan[extra.KEY]['companion'] = str(self.root/'unrelated')
        with self.assertRaises(ValueError):
            extra.free_slots(plan, self.states['parent'], {}, api.n.allocated_nodes)
        Path(self.policy['entry']).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'supervisor changed'):
            self.slots('parent', {})

    def test_handoff_preserves_concurrent_worker_progress_and_all_entry_points(self):
        for role in self.plans:
            self.plans[role].pop(extra.KEY)
            write(Path(self.plans[role]['campaign'])/'plan.json', self.plans[role])
        self.plans['parent']['peers'] = [str(self.child)]
        write(self.parent/'plan.json', self.plans['parent'])
        self.own('parent', ('existing.pbs',))
        script = '''
import fcntl,json,sys,time
from pathlib import Path
p=Path(sys.argv[1])
with (p/'supervisor.lock').open('a+') as f:
 fcntl.flock(f,fcntl.LOCK_EX)
 print('ready',flush=True)
 while json.loads((p/'state.json').read_text())['status']=='running':time.sleep(.01)
 with (p/'state.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  state=json.loads((p/'state.json').read_text())
  state['tasks']['0']['status']='completed'
  (p/'state-child.tmp').write_text(json.dumps(state))
  (p/'state-child.tmp').replace(p/'state.json')
'''
        proc = subprocess.Popen([sys.executable, '-c', script, str(self.parent)], stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(proc.stdout.readline().strip(), 'ready')
            with patch.object(extra.subprocess, 'run', side_effect=AssertionError('No PBS cancellation or submission during handoff')):
                extra.handoff_pair(self.root, self.parent, self.child, self.policy)
            self.assertEqual(proc.wait(timeout=5), 0)
        finally:
            if proc.poll() is None:
                proc.kill(); proc.wait()
            proc.stdout.close()
        for role, before in self.plans.items():
            after = read(Path(before['campaign'])/'plan.json')
            for key in before:
                if key != 'peers':
                    self.assertEqual(after[key], before[key])
        state = read(self.parent/'state.json')
        self.assertEqual(state['status'], 'running')
        self.assertEqual(state['workers']['0']['job_id'], 'existing.pbs')
        self.assertEqual(state['tasks']['0']['status'], 'completed')
        self.assertEqual(read(self.parent/'plan.json')['peers'], [])

    def test_failed_install_restores_plans_and_rearms_original_supervisors(self):
        self.own('companion', ('one.pbs', 'two.pbs'))
        before = {p:read(p/'plan.json') for p in (self.parent,self.child)}
        with patch.object(extra.subprocess, 'run') as restart:
            with self.assertRaisesRegex(ValueError, 'already has extra allocations'):
                extra.handoff_pair(self.root, self.parent, self.child, self.policy)
        self.assertEqual(restart.call_count, 2)
        for path in before:
            self.assertEqual(read(path/'plan.json'), before[path])
            self.assertEqual(read(path/'state.json')['status'], 'running')


if __name__ == '__main__':
    unittest.main()
