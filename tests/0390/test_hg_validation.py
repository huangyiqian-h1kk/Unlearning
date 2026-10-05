"""Check actual HG PBS rendering and precisely scoped queue replacement, without PBS/GPU."""

import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("hg_validation", ROOT / "scripts/abci/0390_validate_hg.py")
hg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hg)


class HGValidationTests(unittest.TestCase):
    def test_presets_render_real_single_gpu_pbs_with_full_evaluation(self):
        spec = importlib.util.spec_from_file_location("hg_submit", ROOT / "scripts/abci/0390_submit.py")
        submit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(submit)
        with tempfile.TemporaryDirectory() as tmp:
            submit.__file__ = str(Path(tmp) / "scripts/abci/0390_submit.py")
            for model in hg.PRESETS:
                command = hg.submit_command(ROOT, model)
                with contextlib.redirect_stdout(io.StringIO()):
                    submit.main([*command[2:], "--dry-run"])
                run_id = hg.PRESETS[model]["run"]
                path = Path(tmp) / f"logs/0390/jobs/0390_validate-series_{model}_{run_id}.pbs"
                script = path.read_text()
                subprocess.run(["bash", "-n"], input=script, text=True, check=True)
                self.assertIn("#PBS -v RTYPE=rt_HG", script)
                self.assertIn("--nproc 1 -- validate-series", script)
                self.assertIn("--include-backbone", script)
                self.assertIn("--set evaluation.batch_size=16", script)
                self.assertIn("sft-validation-v3-hg", script)
                self.assertNotIn("evaluation.limit=", script)
                self.assertNotIn("max_new_tokens=", script)
                self.assertIn("#PBS -l walltime=" + hg.PRESETS[model]["walltime"], script)

    def make_jobs(self, root, states=("Q", "Q")):
        jobs = {}
        for index, (model, state) in enumerate(zip(hg.PRESETS, states), 1):
            run_id = hg.PRESETS[model]["old_run"]
            path = hg.marker(root, model, run_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{index}.pbs1\n")
            jobs[f"{index}.pbs1"] = {"Job_Name": "0390_" + run_id, "job_state": state,
                                     "Job_Owner": hg.getpass.getuser() + "@login"}
        return jobs

    def test_replaces_exact_two_queued_jobs_and_does_not_touch_other_users(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = self.make_jobs(root)
            jobs["other.pbs1"] = {"Job_Name": "0390_unrelated", "job_state": "R", "Job_Owner": "another-user@login"}
            self.assertEqual(hg.pending_replacements(root, list(hg.PRESETS), jobs, True), ["1.pbs1", "2.pbs1"])

    def test_running_hf_and_other_project_capacity_block_all_cancellations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = self.make_jobs(root, ("Q", "R"))
            with self.assertRaisesRegex(RuntimeError, "leaving it untouched"):
                hg.pending_replacements(root, list(hg.PRESETS), jobs, True)
            jobs["2.pbs1"]["job_state"] = "Q"
            jobs["other.pbs1"] = {"job_state": "R", "Job_Owner": hg.getpass.getuser() + "@login"}
            with self.assertRaisesRegex(RuntimeError, "Two-job limit"):
                hg.pending_replacements(root, list(hg.PRESETS), jobs, True)

    def test_replacement_requires_opt_in_and_checks_ownership(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = self.make_jobs(root)
            with self.assertRaisesRegex(RuntimeError, "replace-queued-hf"):
                hg.pending_replacements(root, list(hg.PRESETS), jobs, False)
            jobs["1.pbs1"]["Job_Owner"] = "another-user@login"
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                hg.pending_replacements(root, list(hg.PRESETS), jobs, True)

    def test_switch_calls_qdel_then_common_logging_and_submitters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = self.make_jobs(root)
            for model in hg.PRESETS:
                sft = root / f"results/validated_v2/0390/{model}/sft"
                (sft / "final").mkdir(parents=True)
                (sft / "TRAINING_COMPLETE.json").write_text("{}")
                (sft / "final/config.json").write_text("{}")
            snapshots = [jobs, jobs, jobs, {}]
            with patch.object(hg, "ROOT", root), patch.object(hg, "scheduler", side_effect=snapshots), patch.object(hg.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()):
                hg.main(["--submit", "--replace-queued-hf"])
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(commands[2:4], [["qdel", "1.pbs1"], ["qdel", "2.pbs1"]])
            self.assertIn("0390_logs.py", commands[4][1])
            self.assertEqual(commands[4][-3:], ["collect", "1.pbs1", "2.pbs1"])
            self.assertTrue(all("--dry-run" not in command for command in commands[-2:]))

    def test_job_starting_during_switch_stops_before_qdel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = self.make_jobs(root)
            for model in hg.PRESETS:
                sft = root / f"results/validated_v2/0390/{model}/sft"
                (sft / "final").mkdir(parents=True)
                (sft / "TRAINING_COMPLETE.json").write_text("{}")
                (sft / "final/config.json").write_text("{}")
            started = {key: dict(job) for key, job in jobs.items()}
            started["1.pbs1"]["job_state"] = "R"
            with patch.object(hg, "ROOT", root), patch.object(hg, "scheduler", side_effect=[jobs, started]), patch.object(hg.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "changed state"):
                    hg.main(["--submit", "--replace-queued-hf"])
            self.assertEqual(len(run.call_args_list), 2)
            self.assertTrue(all("--dry-run" in call.args[0] for call in run.call_args_list))


if __name__ == "__main__":
    unittest.main()
