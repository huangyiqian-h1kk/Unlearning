"""Exercise pipeline ordering, real checkpoint selection and PBS provenance on CPU."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from experiments import joblogs, sft_pipeline as pipeline
from experiments.cli import validation_jobs
from experiments.config import load_config, write_json


class SFTPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / "run with spaces"
        self.cfg = load_config(ROOT / "configs/0390/llama8b.yaml")
        self.cfg["run"]["output_dir"] = str(self.out / "sft")
        self.cfg["model"]["name_or_path"] = str(self.root / "backbone")
        write_json(self.root / "backbone/config.json", {})
        data = self.cfg["data"]
        data["prepared_dir"] = str(self.root / "data")
        for name in ("injection", "injection_retain_only", "forget", "retain", "mmlu"):
            write_json(self.root / f"data/{name}.jsonl", {"fixture": True})
        for split in ("forget", "retain"):
            data[f"{split}_generation"] = str(self.root / f"data/{split}.jsonl")
            data[f"{split}_mcq"] = {"attribute": str(self.root / f"data/{split}.jsonl")}
        self.cfg["evaluation"]["mmlu_file"] = str(self.root / "data/mmlu.jsonl")
        self.commands = []

    def workers(self, command, check):
        self.assertTrue(check)
        self.commands.append(command)
        config_pos = command.index("--config")
        stage = command[config_pos - 1]
        cfg = load_config(command[config_pos + 1])
        self.assertEqual(cfg, self.cfg)
        sft = Path(cfg["run"]["output_dir"])
        if stage == "sft":
            for name in ("checkpoint-50", "checkpoint-100", "final"):
                write_json(sft / name / "config.json", {})
            write_json(sft / "resolved_config.json", cfg)
            write_json(sft / "lineage.json", {"training_file": "injection"})
            write_json(sft / "TRAINING_COMPLETE.json", {"global_step": 100, "final": str(sft / "final")})
        elif stage == "validate-series":
            self.assertIn("--include-backbone", command)
            for checkpoint, dest in validation_jobs(cfg, sft, self.out / "validation", True):
                score = {"base": 0.0, "checkpoint-50": 0.648, "checkpoint-100": 0.65, "final": 0.65}[dest.name]
                write_json(dest / "metrics.json", {
                    "checkpoint": str(checkpoint), "protocol_hash": "fixed-test-protocol",
                    "metrics": {"forget.qa": score, "retain.qa": score, "utility.mmlu": 0.70},
                    "mmlu_diagnostics": {"invalid_fraction": 0.01},
                })
        else:
            self.fail(f"Unexpected worker stage: {stage}")

    def run_pipeline(self, worker=None, **kwargs):
        with patch.object(pipeline.subprocess, "run", side_effect=worker or self.workers), \
             patch.dict(os.environ, {"PBS_JOBID": "1234567.pbs1"}), \
             contextlib.redirect_stdout(io.StringIO()):
            return pipeline.run(self.cfg, self.out, nproc=8, **kwargs)

    def test_train_then_full_validation_and_real_near_best_selection(self):
        state = self.run_pipeline()
        self.assertEqual(state["status"], "completed")
        self.assertEqual(Path(state["selected"]).name, "checkpoint-50")
        self.assertEqual(len(self.commands), 2)
        for command in self.commands:
            self.assertIn("--nproc_per_node=8", command)
        self.assertIn("sft", self.commands[0])
        self.assertIn("validate-series", self.commands[1])
        self.assertEqual(len((self.out / "sft-validation-summary.tsv").read_text().splitlines()), 5)
        events = [json.loads(line) for line in (self.out / "pipeline-events.jsonl").read_text().splitlines()]
        self.assertTrue(all(event["job_id"] == "1234567.pbs1" for event in events))

    def test_worker_failure_stops_pipeline_and_preserves_failed_stage(self):
        for failed_stage in ("sft", "validate-series"):
            with self.subTest(stage=failed_stage):
                self.out = self.root / failed_stage
                self.cfg["run"]["output_dir"] = str(self.out / "sft")
                self.commands.clear()

                def fail(command, check):
                    if failed_stage == command[command.index("--config") - 1]:
                        raise subprocess.CalledProcessError(17, command)
                    self.workers(command, check)

                with self.assertRaises(subprocess.CalledProcessError):
                    self.run_pipeline(worker=fail)
                state = json.loads((self.out / "pipeline.json").read_text())
                self.assertEqual(state["failed_stage"], failed_stage)
                self.assertEqual(state["stages"][-1]["exit_code"], 17)
                self.assertFalse((self.out / "selected-sft.json").exists())

    def test_skip_training_reuses_only_matching_completed_run(self):
        self.run_pipeline()
        self.commands.clear()
        state = self.run_pipeline(skip_training=True)
        self.assertEqual(state["stages"][0]["status"], "reused")
        self.assertEqual(len(self.commands), 1)
        self.assertIn("validate-series", self.commands[0])
        self.cfg["sft"]["learning_rate"] = 9e-6
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self.run_pipeline(skip_training=True)

    def test_missing_report_is_not_silently_dropped_from_selection(self):
        def incomplete(command, check):
            self.workers(command, check)
            if "validate-series" in command:
                (self.out / "validation/checkpoint-100/metrics.json").unlink()

        with self.assertRaises(FileNotFoundError):
            self.run_pipeline(worker=incomplete)
        self.assertFalse((self.out / "selected-sft.json").exists())

    def test_no_eligible_checkpoint_keeps_results_and_rejection_report(self):
        def degraded(command, check):
            self.workers(command, check)
            if "validate-series" in command:
                for path in (self.out / "validation").glob("*/metrics.json"):
                    if path.parent.name != "base":
                        report = json.loads(path.read_text())
                        report["metrics"]["utility.mmlu"] = 0.50
                        write_json(path, report)

        with self.assertRaisesRegex(ValueError, "No checkpoint meets"):
            self.run_pipeline(worker=degraded)
        self.assertEqual(json.loads((self.out / "pipeline.json").read_text())["status"], "no_eligible_checkpoint")
        self.assertIsNone(json.loads((self.out / "selected-sft.json").read_text())["selected"])
        self.assertTrue((self.out / "sft-validation-summary.tsv").is_file())

    def test_completed_weights_cannot_be_resumed_over_existing_validation(self):
        self.run_pipeline()
        with self.assertRaisesRegex(ValueError, "Cannot change weights"):
            self.run_pipeline(resume=str(self.out / "sft/checkpoint-50"))

    def test_multirank_coordinator_is_rejected(self):
        with patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "8"}):
            with self.assertRaisesRegex(ValueError, "not inside torchrun"):
                self.run_pipeline()

    def test_submission_and_result_archive_capture_complete_workflow(self):
        self.run_pipeline()
        extra = ["--output", str(self.out)]
        details = joblogs.submission_snapshot(ROOT, "sft-pipeline", "llama8b", "l8sft42", "rt_HF", 8, "03:00:00", extra)
        self.assertNotIn("config_capture_error", details)
        self.assertEqual(details["entry_config"]["run"]["output_dir"], str(self.out / "sft"))
        self.assertEqual(details["output_dir"], str(self.out))
        dest = self.root / "archive"
        joblogs.capture_artifacts(ROOT, dest, str(self.out))
        artifacts = json.loads((dest / "artifacts.json").read_text())
        for name in ("pipeline.json", "pipeline-events.jsonl", "selected-sft.json",
                     "sft-validation-summary.tsv", "sft/TRAINING_COMPLETE.json",
                     "validation/base/metrics.json", "validation/checkpoint-100/metrics.json"):
            self.assertIn(name, artifacts)

    def test_llama8b_pbs_and_single_coordinator_launcher(self):
        spec = importlib.util.spec_from_file_location("submit_pipeline", ROOT / "scripts/abci/0390_submit.py")
        submit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(submit)
        submit.__file__ = str(self.root / "scripts/abci/0390_submit.py")
        with contextlib.redirect_stdout(io.StringIO()):
            submit.main(["sft-pipeline", "--model", "llama8b", "--run-id", "l8sft42",
                         "--output", str(self.out), "--dry-run"])
        script = next((self.root / "logs/0390/jobs").glob("*.pbs")).read_text()
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
        for value in ("#PBS -N 0390_l8sft42", "#PBS -q R9920261000", "#PBS -P gcg51557",
                      "#PBS -v RTYPE=rt_HF", "--nproc 8 -- sft-pipeline", "configs/0390/llama8b.yaml"):
            self.assertIn(value, script)
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 8),
                                version=SimpleNamespace(cuda="12.4"))
        argv = ["0390_launch.py", "--nproc", "8", "--", "sft-pipeline", "--config", "config.yaml", "--output", "out"]
        with patch.dict(sys.modules, {"torch": torch, "deepspeed": SimpleNamespace(__version__="test")}), \
             patch.object(sys, "argv", argv), patch("os.chdir"), \
             patch("subprocess.run") as run, contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(ROOT / "scripts/abci/0390_launch.py"), run_name="__main__")
        command = run.call_args.args[0]
        self.assertNotIn("torch.distributed.run", command)
        self.assertEqual(command[-2:], ["--pipeline-nproc", "8"])


if __name__ == "__main__":
    unittest.main()
