"""Real shell/process log tests plus a simulated PBS scheduler; no GPU required."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from experiments import joblogs


def make_repo(base, launch=None, compiler=True):
    repo = base / "repo with spaces"
    scripts = repo / "scripts/abci"
    scripts.mkdir(parents=True)
    for name in ("0390_run.sh", "0390_logs.py"):
        shutil.copy2(ROOT / "scripts/abci" / name, scripts / name)
    package = repo / "src/experiments"
    package.mkdir(parents=True)
    for name in ("__init__.py", "joblogs.py", "config.py", "cli.py"):
        shutil.copy2(ROOT / "src/experiments" / name, package / name)
    config = repo / "configs/0390/llama3b.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"model": {"name_or_path": "/model"},
                                  "run": {"output_dir": "results/sft"}}))
    venv = base / "venv"
    (venv / "bin").mkdir(parents=True)
    python = venv / "bin/python"
    python.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} \"$@\"\n")
    python.chmod(0o755)
    (venv / "bin/activate").write_text(f"export PATH={shlex.quote(str(venv / 'bin'))}:\"$PATH\"\n")
    (repo / "local.env").write_text(
        f"export CONREP_ENV={shlex.quote(str(venv))}\n"
        f"export CONREP_WORK_ROOT={shlex.quote(str(base / 'work'))}\n"
    )
    cuda = base / "cuda"
    (cuda / "bin").mkdir(parents=True)
    if compiler:
        nvcc = cuda / "bin/nvcc"
        nvcc.write_text("#!/bin/sh\necho simulated-CUDA\n")
        nvcc.chmod(0o755)
    init = base / "modules.sh"
    init.write_text(f"module() {{ export CUDA_HOME={shlex.quote(str(cuda))}; }}\n")
    (scripts / "0390_launch.py").write_text(launch or (
        "import sys\nfrom pathlib import Path\n"
        "print('training-stdout', flush=True)\n"
        "print('training-stderr', file=sys.stderr, flush=True)\n"
        "p = Path('results/sft'); p.mkdir(parents=True, exist_ok=True)\n"
        "(p / 'TRAINING_COMPLETE.json').write_text('{\"global_step\": 2}')\n"
        "(p / 'model.safetensors').write_bytes(b'NOT TO COPY')\n"
    ))
    env = dict(os.environ, BASH_ENV=str(init), PBS_JOBID="123.pbs1", PBS_JOBNAME="0390_test",
               PBS_QUEUE="test_queue", RTYPE="rt_HG", HF_TOKEN="test-secret-not-for-archive")
    env.pop("CONREP_LOG_ACTIVE", None)
    return repo, env


def command(repo):
    return ["bash", str(repo / "scripts/abci/0390_run.sh"), "--nproc", "1", "--", "sft",
            "--config", "configs/0390/llama3b.yaml"]


class JobLogRuntimeTests(unittest.TestCase):
    def test_real_wrapper_captures_stdout_stderr_config_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, env = make_repo(Path(tmp))
            result = subprocess.run(command(repo), env=env, text=True, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            dest = joblogs.job_dir(repo, env["PBS_JOBID"])
            log = (dest / "console.log").read_text()
            for value in ("simulated-CUDA", "training-stdout", "training-stderr"):
                self.assertIn(value, log)
                self.assertIn(value, result.stdout)
            runtime = json.loads((dest / "runtime.json").read_text())
            self.assertEqual(runtime["exit_code"], 0)
            self.assertEqual(runtime["entry_config"]["run"]["output_dir"], "results/sft")
            self.assertTrue((dest / "artifacts/TRAINING_COMPLETE.json").is_file())
            self.assertFalse((dest / "artifacts/model.safetensors").exists())
            self.assertEqual(json.loads((dest / "job.json").read_text())["status_source"], "launcher")
            self.assertTrue((repo / "logs/0390/INDEX.md").is_file())
            for path in dest.rglob("*"):
                if path.is_file():
                    self.assertNotIn("test-secret-not-for-archive", path.read_text())

    def test_failed_training_exit_is_not_hidden_by_log_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, env = make_repo(Path(tmp), launch="import sys\nprint('training failed', flush=True)\nsys.exit(7)\n")
            result = subprocess.run(command(repo), env=env, text=True, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 7, result.stderr)
            dest = joblogs.job_dir(repo, env["PBS_JOBID"])
            self.assertIn("training failed", (dest / "console.log").read_text())
            self.assertEqual(json.loads((dest / "runtime.json").read_text())["exit_code"], 7)

    def test_cuda_initialization_failure_is_archived(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, env = make_repo(Path(tmp), compiler=False)
            result = subprocess.run(command(repo), env=env, text=True, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 1)
            dest = joblogs.job_dir(repo, env["PBS_JOBID"])
            self.assertIn("CUDA compiler not found", (dest / "console.log").read_text())
            self.assertEqual(json.loads((dest / "runtime.json").read_text())["status"], "failed")

    def test_signal_reaches_child_and_records_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, env = make_repo(Path(tmp), launch=(
                "import time\nfrom pathlib import Path\n"
                "print('running-until-signal', flush=True)\n"
                "Path('ready').touch()\ntime.sleep(30)\n"
            ))
            proc = subprocess.Popen(command(repo), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 10
                while not (repo / "ready").exists() and time.monotonic() < deadline:
                    time.sleep(0.03)
                self.assertTrue((repo / "ready").exists())
                dest = joblogs.job_dir(repo, env["PBS_JOBID"])
                self.assertIn("running-until-signal", (dest / "console.log").read_text())
                proc.send_signal(signal.SIGTERM)
                _, stderr = proc.communicate(timeout=10)
                self.assertEqual(proc.returncode, 143, stderr)
                runtime = json.loads((dest / "runtime.json").read_text())
                self.assertEqual(runtime["received_signal"], signal.SIGTERM)
                self.assertEqual(runtime["status"], "failed")
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.communicate()


class JobLogArchiveTests(unittest.TestCase):
    def test_history_backfill_copies_home_logs_and_pbs_metadata_without_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, home = Path(tmp) / "repo", Path(tmp) / "home"
            home.mkdir()
            history = {"job_id": "123.pbs1", "job_name": "0390_old", "stage": "sft",
                       "model": "llama3b", "run_id": "old", "observed_exit_code": 0,
                       "output_dir": "results/old"}
            joblogs.write_json(root / "logs/0390/history.json", {"jobs": [history]})
            original = home / "0390_old.o123"
            original.write_text("old PBS training output\n")
            joblogs.write_json(root / "results/old/TRAINING_COMPLETE.json", {"global_step": 465})
            payload = {"Jobs": {"123.pbs1": {"job_state": "F", "Exit_status": 0,
                       "Job_Name": "0390_old", "resources_used": {"walltime": "00:08:48"},
                       "Variable_List": {"HF_TOKEN": "must-not-copy"}}}}
            result = subprocess.CompletedProcess([], 0, json.dumps(payload), "")
            with patch.object(Path, "home", return_value=home), patch.object(joblogs.subprocess, "run", return_value=result), contextlib.redirect_stdout(io.StringIO()):
                joblogs.collect(root, history_file="logs/0390/history.json")
            dest = joblogs.job_dir(root, "123.pbs1")
            self.assertTrue(original.is_file())
            self.assertEqual((dest / "pbs.stdout.log").read_text(), original.read_text())
            self.assertNotIn("must-not-copy", (dest / "scheduler.json").read_text())
            row = json.loads((dest / "job.json").read_text())
            self.assertEqual(row["status_source"], "PBS")
            self.assertEqual(row["walltime"], "00:08:48")
            self.assertIsNone(row["started_commit"])
            self.assertTrue((dest / "artifacts/TRAINING_COMPLETE.json").exists())
            # Output paths might later be reused. Do not rewrite the earlier artifact snapshot.
            joblogs.write_json(root / "results/old/TRAINING_COMPLETE.json", {"global_step": 999})
            with patch.object(Path, "home", return_value=home), patch.object(joblogs, "query_scheduler", return_value=(None, "expired")), contextlib.redirect_stdout(io.StringIO()):
                joblogs.collect(root, history_file="logs/0390/history.json")
            self.assertEqual(json.loads((dest / "artifacts/TRAINING_COMPLETE.json").read_text())["global_step"], 465)
            self.assertEqual(json.loads((dest / "job.json").read_text())["status_source"], "PBS")
            self.assertEqual(json.loads((dest / "scheduler-query.json").read_text())["error"], "expired")

    def test_missing_scheduler_and_logs_are_not_fabricated_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(joblogs, "query_scheduler", return_value=(None, "expired")), contextlib.redirect_stdout(io.StringIO()):
                joblogs.collect(root, job_ids=["124.pbs1"])
            row = json.loads((joblogs.job_dir(root, "124.pbs1") / "job.json").read_text())
            self.assertEqual(row["status"], "unknown")
            self.assertIsNone(row["exit_code"])

    def test_same_run_id_different_job_ids_preserve_both_submissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = {"stage": "sft", "run_id": "repeat", "git": {"commit": "first"}}
            joblogs.record_submission(root, "1.pbs1", metadata, "script one")
            joblogs.record_submission(root, "2.pbs1", {**metadata, "git": {"commit": "second"}}, "script two")
            rows = json.loads((root / "logs/0390/index.json").read_text())
            self.assertEqual([r["submitted_commit"] for r in rows], ["first", "second"])
            self.assertEqual([r["status"] for r in rows], ["submitted", "submitted"])
            self.assertEqual((joblogs.job_dir(root, "1.pbs1") / "submission.pbs").read_text(), "script one")

    def test_fast_job_start_before_submission_record_keeps_both_git_revisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = joblogs.job_dir(root, "1.pbs1")
            joblogs.write_json(dest / "runtime.json", {"git": {"commit": "actual-start"},
                               "status": "completed", "exit_code": 0})
            joblogs.record_submission(root, "1.pbs1", {"git": {"commit": "at-submit"}}, "script")
            row = json.loads((dest / "job.json").read_text())
            self.assertEqual(row["submitted_commit"], "at-submit")
            self.assertEqual(row["started_commit"], "actual-start")
            self.assertEqual(row["status"], "completed")

    def test_post_qsub_archive_error_does_not_look_like_submission_failure(self):
        spec = importlib.util.spec_from_file_location("submit_log_test", ROOT / "scripts/abci/0390_submit.py")
        submit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(submit)
        with tempfile.TemporaryDirectory() as tmp:
            submit.__file__ = str(Path(tmp) / "scripts/abci/0390_submit.py")
            calls = [subprocess.CompletedProcess([], 0, '{"Jobs": {}}', ""),
                     subprocess.CompletedProcess([], 0, "125.pbs1\n", "")]
            out, err = io.StringIO(), io.StringIO()
            with patch.object(submit.subprocess, "run", side_effect=calls), patch.object(submit, "submission_snapshot", return_value={}), patch.object(submit, "record_submission", side_effect=OSError("disk error")), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                submit.main(["sft", "--model", "llama3b", "--run-id", "repeat"])
            self.assertIn("IS SUBMITTED", err.getvalue())
            self.assertIn("125.pbs1", out.getvalue())
            marker = Path(tmp) / "logs/0390/jobs/0390_sft_llama3b_repeat.jobid"
            self.assertEqual(marker.read_text().strip(), "125.pbs1")


if __name__ == "__main__":
    unittest.main()
