"""Scheduler regressions: no GPUs, model assets, or live PBS commands."""
import argparse
import time

import pytest

from conrep.night import campaign as c
from conrep.night.io import read, write


@pytest.mark.parametrize("record,expected", [
    ({"Exit_status": 271, "comment": "Job run and terminated by root@admin.example.invalid",
      "resources_used": {"walltime": "00:00:59"},
      "Resource_List": {"walltime": "06:00:00"}}, "admin_terminated"),
    ({"comment": "Not Running: Placement set is too small: Insufficient amount of resource: "
                 "node_group (group_a != group_b) and terminated"}, "placement_failure"),
    ({"Exit_status": 271, "comment": "walltime exceeded; terminated by root@admin.example.invalid"}, "recoverable"),
    ({"Exit_status": 271, "comment": "qdel requested by user"}, "cancelled"),
    ({"Exit_status": 271}, "unknown_failure"),
    ({"Exit_status": 0}, "completed"),
])
def test_scheduler_evidence_is_not_conflated(record, expected):
    assert c.classify_end(record) == expected


def state_fixture(root):
    state = {"status": "running", "started_at": time.time(), "deadline": time.time() + 3600,
             "tasks": {"example": {"status": "pending"}},
             "workers": {str(i): {"status": status, "job_id": f"{i}.pbs",
                 "pbs_comment": "scheduler evidence", "reconciled_job": f"{i}.pbs"} for i, status in enumerate((
                 "admin_terminated", "admin_terminated", "placement_failure", "placement_failure"))}}
    write(root / "state.json", state)
    write(root / "plan.json", {"tasks": [], "workers": 3})
    return state


def test_terminal_events_are_visible_without_reading_child_logs(tmp_path, capsys):
    c.event(tmp_path, event="pbs-finished", outcome="placement_failure", comment="reason")
    console = capsys.readouterr().out
    assert '"placement_failure"' in console and '"comment": "reason"' in console
    assert (tmp_path / "events.jsonl").read_text() == console


def test_resume_cannot_claim_success_or_extend_budget_with_all_workers_blocked(tmp_path, monkeypatch):
    before = state_fixture(tmp_path)
    before["status"] = "blocked"
    write(tmp_path / "state.json", before)
    monkeypatch.setattr(c, "verify_snapshot", lambda *_: None)
    monkeypatch.setattr(c.subprocess, "Popen", lambda *a, **k: pytest.fail("Must not start a supervisor"))
    with pytest.raises(ValueError, match="All workers are blocked"):
        c.start(argparse.Namespace(campaign=str(tmp_path), hours=10), resume=True)
    assert read(tmp_path / "state.json") == before


def test_blocked_supervisor_writes_reason_and_stops(tmp_path, monkeypatch, capsys):
    state_fixture(tmp_path)
    monkeypatch.setattr(c, "qstat", lambda *a: {"Jobs": {}})
    monkeypatch.setattr(c, "reconcile", lambda *a: None)
    monkeypatch.setattr(c, "submit_available", lambda *a: pytest.fail("Must not retry administrator or placement failures"))
    monkeypatch.setattr(c.time, "sleep", lambda *a: pytest.fail("Must not silently idle"))
    assert c.supervise(argparse.Namespace(campaign=str(tmp_path))) == 0
    assert read(tmp_path / "state.json")["status"] == "blocked"
    assert read(tmp_path / "BLOCKED.json")["workers"]["2"]["status"] == "placement_failure"
    assert read(tmp_path / "summary.json")["workers"]["0"]["pbs_comment"] == "scheduler evidence"
    assert '"all-workers-blocked"' in capsys.readouterr().out


def test_reconcile_preserves_full_failure_evidence(tmp_path, monkeypatch):
    state_fixture(tmp_path)
    with c.state_transaction(tmp_path) as state:
        state["workers"] = {"0": {"status": "running", "job_id": "0.pbs", "recovery_failures": 0}}
    record = {"job_state": "F", "Exit_status": 271, "comment": "terminated by root@admin.example.invalid", "stime": "example"}
    monkeypatch.setattr(c, "qstat", lambda *a: {"Jobs": {"0.pbs": record}})
    monkeypatch.setattr(c, "archive_job", lambda *a: None)
    c.reconcile(tmp_path, {"max_attempts": 3}, {})
    worker = read(tmp_path / "state.json")["workers"]["0"]
    assert worker["status"] == "admin_terminated"
    assert worker["pbs_exit_status"] == 271 and worker["pbs_comment"] == record["comment"]
    assert worker["recovery_failures"] == 0


def test_job_name_must_match_the_established_prefix():
    plan = {"shell": "/worker", "project_root": "/project", "campaign": "/campaign", "walltime": "06:00:00"}
    with pytest.raises(ValueError, match="0390_"):
        c.render_pbs(plan, 0, "0390nabc0001")
    assert "#PBS -N 0390_nabc0001" in c.render_pbs(plan, 0, "0390_nabc0001")
