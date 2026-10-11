"""CPU-only regressions for the production MMLU reader failure and repair."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from conrep.night.io import read, write
from experiments.config import read_rows
from experiments.selective_data import rows, write_rows
from experiments.selective_design import read_rows as design_rows

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('reader_repair', ROOT / 'scripts/abci/0390_repair_selective_mmlu.py')
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


class ReaderRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_unicode_separators_are_not_jsonl_record_boundaries(self):
        for separator in ('\x85', '\u2028', '\u2029'):
            with self.subTest(separator=ascii(separator)):
                path = self.root / 'test.jsonl'
                original = [dict(subject='anatomy', prompt='left'+separator+'right', answer='A')]
                write_rows(path, original)
                with self.assertRaises(json.JSONDecodeError):
                    [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
                self.assertEqual(rows(path), original)
                self.assertEqual(rows(path), read_rows(path))
                self.assertEqual(design_rows(path), original)

    def test_standard_jsonl_blank_lines_and_corruption(self):
        path = self.root / 'test.jsonl'
        path.write_text('{"id":1}\n\n{"id":2}\n')
        self.assertEqual(rows(path), [{'id': 1}, {'id': 2}])
        path.write_text('{"id":1}\n{"id":')
        with self.assertRaises(json.JSONDecodeError):
            rows(path)

    def test_shared_reader_formats(self):
        path = self.root / 'test.json'
        path.write_text('[\n{"id": 1}\n]')
        self.assertEqual(rows(path), [{'id': 1}])
        path = self.root / 'test.csv'
        path.write_text('subject,prompt,answer\na,"line 1\nline 2",A\n')
        self.assertEqual(rows(path)[0]['prompt'], 'line 1\nline 2')

    def test_only_reviewed_function_edit_allowed(self):
        after = (ROOT / repair.REL).read_bytes()
        before = after.decode().replace(repair.NEW, repair.OLD).encode()
        repair.corrected_source(before, after)
        with self.assertRaises(ValueError):
            repair.corrected_source(before, after + b'\n# unrelated change\n')

    def test_mmlu_preflight_keeps_overlap_guard_and_input_bytes(self):
        validation, test = [self.root / (name+'.jsonl') for name in ('val', 'test')]
        a = dict(subject='a', prompt='v', answer='B')
        b = dict(subject='a', prompt='t\u2028continuation', answer='A')
        write_rows(validation, [a]); write_rows(test, [b])
        cfg = {'evaluation': {'mmlu_file': str(validation), 'mmlu_test_file': str(test)}}
        before = test.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            repair.check_mmlu([cfg, cfg], read_rows)
        self.assertEqual(test.read_bytes(), before)
        write_rows(test, [a])
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'overlaps'):
            repair.check_mmlu([cfg], read_rows)

    def test_guard_rejects_training_or_live_allocations(self):
        plan = dict(grid_phase=0, tasks=[dict(id='ref',kind='reference'),dict(id='train',kind='train')])
        state = dict(status='blocked', tasks={'ref':dict(status='failed',attempts=1),
            'train':dict(status='pending',attempts=0)},workers={'0':dict(status='available',job_id='1.pbs')})
        repair.guard_idle(plan,state,{})
        with self.assertRaises(ValueError):
            repair.guard_idle(plan,state,{'1.pbs':{}})
        for field, value in [('status','running'),('attempts',1)]:
            changed = copy.deepcopy(state); changed['tasks']['train'][field] = value
            with self.assertRaises(ValueError):
                repair.guard_idle(plan,changed,{})

    def test_transaction_recovers_partial_apply_without_touching_predictions(self):
        (self.root / 'a.json').write_text('old-a')
        (self.root / 'b.json').write_text('old-b')
        predictions = self.root / 'predictions.jsonl'; predictions.write_text('unchanged')
        folder = self.root / 'repairs' / 'fixture'
        repair.stage(self.root, folder, {'a.json': b'new-a', 'b.json': b'new-b'}, {'ref':'fixture'})
        (self.root / 'a.json').write_text('new-a')
        repair.apply_transaction(self.root, folder)
        repair.apply_transaction(self.root, folder)
        self.assertEqual((self.root / 'b.json').read_text(), 'new-b')
        self.assertEqual((folder / 'before/a.json').read_text(), 'old-a')
        self.assertEqual(predictions.read_text(), 'unchanged')

    def test_transaction_rejects_foreign_edits_before_writing_any_file(self):
        for name in ('a', 'b'):
            (self.root / name).write_text('old')
        folder = self.root / 'repair'
        repair.stage(self.root,folder,{'a':b'new','b':b'new'},{})
        (self.root / 'b').write_text('foreign')
        with self.assertRaises(ValueError):
            repair.apply_transaction(self.root,folder)
        self.assertEqual((self.root / 'a').read_text(),'old')


if __name__ == '__main__':
    unittest.main()
