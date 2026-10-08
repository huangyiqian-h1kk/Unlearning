"""Recovery, original-task continuity, and controller-only upgrade regressions."""
import argparse
import copy
import getpass
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

from conrep.night import campaign as c
from conrep.night.io import read, write, sha, file_sha, training_identity, latest_checkpoint

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("night_upgrade", ROOT / "scripts/abci/0390_recover_conrep_night.py")
upgrade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upgrade)


def queued_end():
    return {"job_state": "F", "comment": "Not Running: Placement set is too small: "
        "Insufficient amount of resource: node_group (group_a != group_b) and terminated",
        "history_timestamp": time.time() - 600}


def admin_end():
    return {"job_state": "F", "Exit_status": 271,
        "comment": "Job run and terminated by root@admin.example.invalid",
        "history_timestamp": time.time() - 600, "stime": "started"}


def fixture(root, count=4):
    tasks = []
    for i in range(count):
        cfg = {"unlearn": {"max_steps": 125, "save_steps": 10},
               "run": {"output_dir": str(root / f"training-{i}")}}
        config = root / f"config-{i}.json"
        write(config, cfg)
        tasks.append({"id": str(i), "identity": "test", "priority": 0,
                      "preferred_worker": i % 4, "config": str(config), "output": str(root / f"output-{i}")})
    plan = {"project_root": str(root), "campaign": str(root), "tasks": tasks,
            "shell": "/original/worker", "controller_shell": "/patched/worker",
            "walltime": "06:00:00", "initial_task_seconds": 3600, "source_hash": "original",
            "queue": "R9920261000", "rtype": "rt_HF", "max_attempts": 3,
            "workers": 4, "controller_ref": "test-approved-upgrade",
            "recovery_policy": dict(c.RECOVERY_DEFAULTS)}
    state = {"status": "blocked", "started_at": time.time() - 100,
             "deadline": time.time() + 36000,
             "tasks": {task["id"]: {"status": "pending", "attempts": 0, "failures": 0} for task in tasks},
             "workers": {str(i): {"status": "unknown_failure", "job_id": f"old{i}.pbs",
                 "reconciled_job": f"old{i}.pbs", "last_outcome": "unknown_failure",
                 "allocations": 1, "recovery_failures": 0} for i in range(4)}}
    write(root / "plan.json", plan)
    write(root / "state.json", state)
    return plan, state


def test_actual_four_failure_shapes_recover_and_submit_once(tmp_path, monkeypatch):
    plan, before = fixture(tmp_path, count=32)
    records = {f"old{i}.pbs": admin_end() if i < 2 else queued_end() for i in range(4)}
    active, submissions = {}, []
    def qstat(*args):
        return {"Jobs": {args[1]: records[args[1]]}} if args else {"Jobs": active}
    monkeypatch.setattr(c, "qstat", qstat)
    monkeypatch.setattr(c, "verify_snapshot", lambda *a: None)
    monkeypatch.setattr(c, "archive_job", lambda *a: None)
    starts = []
    monkeypatch.setattr(c, "start", lambda *a, **k: starts.append(True) or 0)
    assert c.recover(argparse.Namespace(campaign=str(tmp_path), hours=None)) == 0
    state = read(tmp_path / "state.json")
    assert state["status"] == "running" and state["deadline"] == before["deadline"]
    assert state["tasks"] == before["tasks"]
    assert all(w["status"] == "available" and w["recovery_failures"] == 1 for w in state["workers"].values())
    assert state["workers"]["2"]["last_reason"] == "terminated_while_queued"
    c.reconcile(tmp_path, plan, {}, reassess=True)
    assert all(w["recovery_failures"] == 1 for w in read(tmp_path / "state.json")["workers"].values())
    def submit(command, **kwargs):
        submissions.append(command)
        job = f"{100 + len(submissions)}.pbs"
        active[job] = {"Job_Owner": getpass.getuser() + "@test", "job_state": "Q"}
        return subprocess.CompletedProcess(command, 0, job + "\n", "")
    monkeypatch.setattr(c.subprocess, "run", submit)
    c.submit_available(tmp_path, plan)
    c.submit_available(tmp_path, plan)
    assert len(submissions) == 3 and len(starts) == 1
    for command in submissions:
        script = Path(command[1]).read_text()
        assert "#PBS -N 0390_" in script and "/patched/worker" in script
        assert "node_group=" not in script


def test_retry_budget_counts_allocations_once_and_preserves_task(tmp_path, monkeypatch):
    plan, state = fixture(tmp_path)
    records = {}
    monkeypatch.setattr(c, "qstat", lambda *a: {"Jobs": {a[1]: records[a[1]]}})
    monkeypatch.setattr(c, "archive_job", lambda *a: None)
    for attempt in range(1, 5):
        job = f"new{attempt}.pbs"
        records[job] = admin_end()
        records[job]["history_timestamp"] = time.time()
        with c.state_transaction(tmp_path) as current:
            current["workers"] = {"0": {**current["workers"]["0"], "status": "running", "job_id": job}}
            current["tasks"]["0"].update(status="interrupted", job_id=job)
        c.reconcile(tmp_path, plan, {})
        c.reconcile(tmp_path, plan, {})
        current = read(tmp_path / "state.json")
        assert current["workers"]["0"]["recovery_failures"] == attempt
        assert current["tasks"]["0"]["failures"] == attempt
        assert current["tasks"]["0"]["status"] == ("paused" if attempt <= 3 else "failed")
        assert current["workers"]["0"]["status"] == ("available" if attempt <= 3 else "retry_exhausted")
        assert current["workers"]["0"]["retry_after"] > time.time() + 50


def test_backoff_prevents_premature_resubmission(tmp_path, monkeypatch):
    plan, _ = fixture(tmp_path)
    with c.state_transaction(tmp_path) as state:
        state["status"] = "running"
        for worker in state["workers"].values():
            worker.update(status="available", retry_after=time.time() + 300)
    monkeypatch.setattr(c, "qstat", lambda *a: {"Jobs": {}})
    monkeypatch.setattr(c.subprocess, "run", lambda *a, **k: pytest.fail("Backoff must defer qsub"))
    c.submit_available(tmp_path, plan)


@pytest.mark.parametrize("comment", ["qdel requested by user", "terminated by someone@host",
    "deleted by user", "access denied; terminated by root@admin.example.invalid"])
def test_admin_retry_policy_does_not_revive_user_cancel_or_access_denial(comment):
    outcome, _ = c.recovery_outcome({"job_state": "F", "Exit_status": 271, "comment": comment},
                                  {"recovery_policy": dict(c.RECOVERY_DEFAULTS)})
    assert outcome in {"cancelled", "policy_denied"}


def test_queued_waiting_comment_is_not_an_interruption():
    job = queued_end()
    job["job_state"] = "Q"
    assert c.recovery_outcome(job, {"recovery_policy": dict(c.RECOVERY_DEFAULTS)})[0] != "recoverable"
    job["job_state"] = "F"
    job["comment"] = job["comment"].replace(" and terminated", "")
    assert c.recovery_outcome(job, {"recovery_policy": dict(c.RECOVERY_DEFAULTS)})[0] != "recoverable"


def test_paused_task_is_claimed_before_new_experiment(tmp_path, monkeypatch):
    plan, _ = fixture(tmp_path)
    with c.state_transaction(tmp_path) as state:
        state["status"] = "running"
        state["tasks"]["3"]["status"] = "paused"
    monkeypatch.setenv("PBS_JOBID", "replacement.pbs")
    assert c.claim(tmp_path, plan, 0, time.time() + 20000)["id"] == "3"


@pytest.mark.parametrize("interrupted", [False, True])
def test_child_exit_before_parent_observes_signal(tmp_path, monkeypatch, interrupted):
    class ExitedChild:
        returncode = 1
        def poll(self):
            return self.returncode
    monkeypatch.setattr(c.subprocess, "Popen", lambda *a, **k: ExitedChild())
    code = c.run_child(["trainer"], tmp_path / "console.log", project=tmp_path,
                       deadline=time.time() + 1000, stop_file=tmp_path / "STOP",
                       interrupted=lambda: interrupted)
    assert code == (c.PAUSE if interrupted else 1)


@pytest.mark.parametrize("exit_code", [-15, -9, 143, 137, 1])
def test_pbs_interruption_recovers_killed_child_without_reviving_training_errors(tmp_path, monkeypatch, exit_code):
    plan, _ = fixture(tmp_path)
    with c.state_transaction(tmp_path) as state:
        state["workers"] = {"0": {**state["workers"]["0"], "job_id": "new.pbs", "status": "running"}}
        state["tasks"]["0"].update(status="failed", job_id="new.pbs", exit_code=exit_code)
    monkeypatch.setattr(c, "qstat", lambda *a: {"Jobs": {"new.pbs": admin_end()}})
    monkeypatch.setattr(c, "archive_job", lambda *a: None)
    c.reconcile(tmp_path, plan, {})
    assert read(tmp_path / "state.json")["tasks"]["0"]["status"] == ("failed" if exit_code == 1 else "paused")


def make_upgrade_fixture(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    shutil.copytree(ROOT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    old = {name: subprocess.check_output(["git", "show", f"{upgrade.BASE_REF}:{name}"], cwd=ROOT)
           for name in upgrade.OWNED}
    for name, payload in old.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    def commit():
        subprocess.run(["git", "add", "."], cwd=root, check=True)
        subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                        "commit", "-qm", "fixture"], cwd=root, check=True)
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    base = commit()
    for name in upgrade.OWNED:
        (root / name).write_bytes((ROOT / name).read_bytes())
    ref = commit()
    for name, payload in old.items():
        (root / name).write_bytes(payload)
    dirty = root / "src/server_only.py"
    dirty.write_text("server_local_value = 42\n")
    campaign = root / "results/campaign"
    snapshot = campaign / "code"
    shutil.copytree(root / "src", snapshot / "src")
    for name in (c.ENTRY, c.SHELL):
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(old[name])
    hashes = {str(p.relative_to(snapshot)): file_sha(p) for p in snapshot.rglob("*") if p.is_file()}
    source_hash = sha(hashes)
    write(campaign / "source.json", {"source_hash": source_hash, "files": hashes})
    write(campaign / "inputs.json", {})
    write(campaign / "model-assets.json", {})
    cfg = {"run": {"seed": 42}, "unlearn": {"max_steps": 125, "save_steps": 10},
           "model": {}, "data": {}, "lora": {}, "conrep": {},
           "night": {"source_hash": source_hash}}
    config = campaign / "config.json"
    write(config, cfg)
    identity = training_identity(cfg)
    checkpoint = campaign / "experiments/example/training/checkpoint-10"
    checkpoint.mkdir(parents=True)
    files = {name: 1 for name in ["training_state.pt", "adapter_config.json", "adapter_model.safetensors"]
             + [f"rng-rank-{i}.pt" for i in range(8)]}
    for name in files:
        (checkpoint / name).write_bytes(b"x")
    write(checkpoint / "COMPLETE.json", {"schema": c.NAME, "identity": identity, "world_size": 8, "files": files})
    plan = {"project_root": str(root), "campaign": str(campaign), "python": sys.executable, "workers": 4,
            "source_hash": source_hash, "entry": str(snapshot / c.ENTRY), "shell": str(snapshot / c.SHELL),
            "tasks": [{"id": "example", "config": str(config), "config_hash": file_sha(config),
                       "identity": identity, "output": str(checkpoint.parent.parent)}]}
    write(campaign / "plan.json", plan)
    write(campaign / "state.json", {"status": "blocked", "deadline": time.time() + 10000,
                                    "workers": {}, "tasks": {"example": {"status": "paused"}}})
    monkeypatch.setattr(upgrade, "BASE_REF", base)
    monkeypatch.setattr(upgrade, "PREVIOUS_REFS", ())
    monkeypatch.setattr(upgrade, "assert_no_live_workers", lambda *a: None)
    return root, campaign, ref, checkpoint, plan


def test_controller_upgrade_keeps_frozen_source_dirty_server_code_and_checkpoint_identity(tmp_path, monkeypatch):
    root, campaign, ref, checkpoint, old = make_upgrade_fixture(tmp_path, monkeypatch)
    source_before = (campaign / "source.json").read_bytes()
    config_before = Path(old["tasks"][0]["config"]).read_bytes()
    state_before = (campaign / "state.json").read_bytes()
    new = upgrade.upgrade(root, campaign, ref)
    assert all(new[key] == old[key] for key in ("entry", "shell", "source_hash", "tasks"))
    assert (campaign / "source.json").read_bytes() == source_before
    assert Path(old["tasks"][0]["config"]).read_bytes() == config_before
    assert (campaign / "state.json").read_bytes() == state_before
    assert (root / "src/server_only.py").read_text() == "server_local_value = 42\n"
    assert latest_checkpoint(checkpoint.parent, identity=old["tasks"][0]["identity"], world=8) == checkpoint
    c.verify_snapshot(campaign, new)
    assert upgrade.upgrade(root, campaign, ref) == new
    assert len(list((campaign / "recovery-backups").iterdir())) == 1
    # Exercise the real checkout entry -> new frozen controller dispatch, with
    # scheduler commands unavailable so this is strictly a local status read.
    result = subprocess.run([sys.executable, str(root / c.ENTRY), "status", "--campaign", str(campaign)],
                            env={**os.environ, "PATH": "/nonexistent"},
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["workers"] == {}
    Path(new["controller_entry"]).write_text("changed")
    with pytest.raises(ValueError, match="Recovery controller changed"):
        c.verify_snapshot(campaign, new)


def test_upgrade_refuses_dirty_owned_files_before_any_replacement(tmp_path, monkeypatch):
    root, campaign, ref, _, _ = make_upgrade_fixture(tmp_path, monkeypatch)
    (root / c.SHELL).write_text("locally edited worker\n")
    before = (campaign / "plan.json").read_bytes()
    with pytest.raises(FileExistsError, match="Locally edited file preserved"):
        upgrade.upgrade(root, campaign, ref)
    assert (root / c.SHELL).read_text() == "locally edited worker\n"
    assert (campaign / "plan.json").read_bytes() == before
    assert not (campaign / "recovery-backups").exists()


def test_upgrade_refuses_live_allocations(tmp_path, monkeypatch):
    root, campaign, ref, _, _ = make_upgrade_fixture(tmp_path, monkeypatch)
    before = (campaign / "plan.json").read_bytes()
    monkeypatch.setattr(upgrade, "assert_no_live_workers", lambda *a: (_ for _ in ()).throw(RuntimeError("active PBS")))
    with pytest.raises(RuntimeError, match="active PBS"):
        upgrade.upgrade(root, campaign, ref)
    assert (campaign / "plan.json").read_bytes() == before
