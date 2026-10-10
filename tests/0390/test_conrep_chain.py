"""Handoff state transitions, scheduler uncertainty and delayed budget start."""

import argparse
import getpass
from pathlib import Path

import pytest

from conrep.night import campaign as c, chain
from conrep.night.io import read, write, file_sha


def fixture(tmp_path, monkeypatch):
    parent, successor = tmp_path / "parent", tmp_path / "successor"
    cfg = parent / "config.json"
    write(cfg, {"unlearn": {"max_steps": 125, "save_steps": 10}, "evaluation": {"mmlu_file": "mmlu"}})
    task = {"id": "first", "config": str(cfg), "output": str(parent / "experiment")}
    write(parent / "plan.json", {"project_root": str(tmp_path), "tasks": [task], "workers": 3})
    for step in c.expected_checkpoints(read(cfg)):
        target = parent / "experiment/validation" / f"checkpoint-{step}"
        write(target / "metrics.json", {"protocol_hash": "v5"})
        write(target / "NIGHT_VALIDATED.json", {"protocol_hash": "v5", "prediction_sizes": {"mmlu_predictions.jsonl": 3}})
    write(parent / "state.json", {"status": "running", "tasks": {"first": {"status": "running"}},
        "workers": {"0": {"status": "running", "job_id": "new-after-admin-kill.pbs", "job_name": "0390_parent"}}})
    plan = {"project_root": str(tmp_path), "campaign": str(successor), "workers": 3,
        "start_after_campaign": str(parent), "start_after_plan_sha256": file_sha(parent / "plan.json"),
        "poll_seconds": 60, "hours": 10, "python": "/python", "entry": "/entry"}
    write(successor / "plan.json", plan)
    write(successor / "state.json", {"status": "prepared", "started_at": None, "deadline": None,
        "workers": {"0": {"status": "new"}}})
    write(successor / "handoff.json", {"successor_plan_sha256": file_sha(successor / "plan.json")})
    monkeypatch.setattr(c, "verify_snapshot", lambda *a: None)
    monkeypatch.setattr(c, "training_done", lambda *a: True)
    monkeypatch.setattr(c, "validation_done", lambda *a: True)
    monkeypatch.setattr(c, "qstat", lambda *a: {"Jobs": {}})
    return parent, successor, plan


def finish(parent, *, settled=True):
    state = read(parent / "state.json")
    state["tasks"]["first"]["status"] = "completed"
    state["workers"]["0"]["status"] = "drained"
    if settled:
        state["workers"]["0"]["reconciled_job"] = state["workers"]["0"]["job_id"]
    write(parent / "state.json", state)


def test_waits_for_tasks_current_allocation_and_final_scheduler_evidence(tmp_path, monkeypatch):
    parent, successor, plan = fixture(tmp_path, monkeypatch)
    assert chain.parent_readiness(plan)[0] == "waiting"
    assert read(successor / "state.json")["deadline"] is None
    finish(parent, settled=False)
    assert chain.parent_readiness(plan)[0] == "waiting"
    finish(parent)
    monkeypatch.setattr(c, "qstat", lambda: {"Jobs": {"new-after-admin-kill.pbs": {
        "Job_Owner": getpass.getuser()+"@pbs", "job_state": "E", "Job_Name": "0390_parent"}}})
    assert chain.parent_readiness(plan)[0] == "waiting"
    monkeypatch.setattr(c, "qstat", lambda: {"Jobs": {}})
    before = (parent / "state.json").read_bytes()
    assert chain.parent_readiness(plan)[0] == "ready"
    assert (parent / "state.json").read_bytes() == before
    monkeypatch.setattr(c, "validation_done", lambda task, step: step != 125)
    assert chain.parent_readiness(plan)[0] == "blocked"


@pytest.mark.parametrize("status", ["stopped", "deadline_reached", "budget_paused", "storage_paused", "blocked"])
def test_terminal_parent_does_not_extend_budget_or_launch(tmp_path, monkeypatch, status):
    parent, successor, plan = fixture(tmp_path, monkeypatch)
    state = read(parent / "state.json")
    state["status"] = status
    write(parent / "state.json", state)
    monkeypatch.setattr(c, "start", lambda *a: pytest.fail("Must not launch"))
    assert chain.watch(argparse.Namespace(campaign=str(successor))) == 2
    assert read(successor / "state.json")["started_at"] is None
    assert read(parent / "state.json") == state


def test_finished_with_failed_task_is_not_success_and_source_changes_block(tmp_path, monkeypatch):
    parent, successor, plan = fixture(tmp_path, monkeypatch)
    state = read(parent / "state.json")
    state.update(status="finished")
    state["tasks"]["first"]["status"] = "failed"
    write(parent / "state.json", state)
    assert chain.parent_readiness(plan)[0] == "blocked"
    finish(parent)
    write(parent / "plan.json", dict(read(parent / "plan.json"), changed=True))
    assert chain.parent_readiness(plan)[0] == "blocked"


def test_qstat_error_and_occupied_account_are_not_free_slots(tmp_path, monkeypatch):
    parent, successor, plan = fixture(tmp_path, monkeypatch)
    finish(parent)
    def bad():
        raise RuntimeError("PBS unavailable")
    monkeypatch.setattr(c, "qstat", bad)
    with pytest.raises(RuntimeError):
        chain.parent_readiness(plan)
    monkeypatch.setattr(c, "qstat", lambda: {"Jobs": {str(i): {
        "Job_Owner": getpass.getuser()+"@pbs", "Job_Name": "other", "job_state": "Q"} for i in range(3)}})
    assert chain.parent_readiness(plan)[0] == "waiting"
    assert read(successor / "state.json")["deadline"] is None


def test_wait_then_start_once_and_restart_after_publication_crash(tmp_path, monkeypatch):
    parent, successor, plan = fixture(tmp_path, monkeypatch)
    calls = []
    def start(args):
        calls.append(args.campaign)
        state = read(successor / "state.json")
        state.update(status="running", started_at=999, deadline=999+36000)
        write(successor / "state.json", state)
    monkeypatch.setattr(c, "start", start)
    monkeypatch.setattr(chain.time, "sleep", lambda seconds: finish(parent))
    assert chain.watch(argparse.Namespace(campaign=str(successor))) == 0
    assert calls == [str(successor)]
    assert read(successor / "handoff-state.json")["status"] == "launched"
    # Re-entry adopts through the existing idempotent start API, not qsub.
    assert chain.watch(argparse.Namespace(campaign=str(successor))) == 0
    assert len(calls) == 2
    assert read(successor / "state.json")["deadline"] == 999+36000


def test_manual_start_cannot_bypass_parent_gate_and_budget_starts_late(tmp_path, monkeypatch):
    parent, successor, plan = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="cannot start"):
        c.start(argparse.Namespace(campaign=str(successor)))
    assert read(successor / "state.json")["deadline"] is None
    with pytest.raises(ValueError, match="has not started"):
        c.recover(argparse.Namespace(campaign=str(successor), hours=10, requeue_jobs=[]))
    assert read(successor / "state.json")["deadline"] is None
    finish(parent)
    popen_calls = []
    class Process:
        pid = 123
        def poll(self): return None
    monkeypatch.setattr(c.subprocess, "Popen", lambda *a, **kw: (popen_calls.append((a, kw)) or Process()))
    monkeypatch.setattr(c.time, "time", lambda: 50000.)
    monkeypatch.setattr(c.time, "sleep", lambda _: None)
    assert c.start(argparse.Namespace(campaign=str(successor))) == 0
    state = read(successor / "state.json")
    assert state["started_at"] == 50000 and state["deadline"] == 86000
    assert popen_calls[0][1]["start_new_session"] is True


def test_successor_stop_is_respected_before_launch(tmp_path, monkeypatch):
    parent, successor, plan = fixture(tmp_path, monkeypatch)
    finish(parent)
    (successor / "STOP").touch()
    monkeypatch.setattr(c, "start", lambda *a: pytest.fail("Explicit stop must be respected"))
    assert chain.watch(argparse.Namespace(campaign=str(successor))) == 0
    assert read(successor / "handoff-state.json")["status"] == "stopped"


def test_duplicate_arm_is_idempotent_and_does_not_start_budget(tmp_path, monkeypatch):
    from conrep.night.io import locked
    parent, successor, plan = fixture(tmp_path, monkeypatch)
    # arm's immutable request is created before checking the process lock.
    (successor / "handoff.json").unlink()
    monkeypatch.setattr(chain.subprocess, "Popen", lambda *a, **k: pytest.fail("Duplicate watcher"))
    with locked(successor / "handoff.lock"):
        assert chain.arm(argparse.Namespace(campaign=str(successor))) == 0
    assert read(successor / "state.json")["started_at"] is None
