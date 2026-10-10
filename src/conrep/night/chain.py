"""Read-only parent gate and a detached, idempotent campaign handoff.

No PBS dependency on transient allocation IDs. Never stop, edit, or extend
the parent campaign. Waiting does not start the successor's time budget.
"""

import argparse
import getpass
import json
import os
from pathlib import Path
import subprocess
import time

from . import campaign as c
from .io import read, write, locked, file_sha


def parent_readiness(plan):
    parent = Path(plan["start_after_campaign"])
    if file_sha(parent / "plan.json") != plan["start_after_plan_sha256"]:
        return "blocked", "Parent plan changed after preparation; inspect before re-arming"
    parent_plan, state = read(parent / "plan.json"), read(parent / "state.json")
    if Path(parent_plan["project_root"]).resolve() != Path(plan["project_root"]).resolve():
        return "blocked", "Parent belongs to another project"
    ids = {t["id"] for t in parent_plan["tasks"]}
    if not ids or len(ids) != len(parent_plan["tasks"]) or set(state["tasks"]) != ids:
        return "blocked", "Parent task inventory differs from the pinned plan"
    statuses = {key: item["status"] for key, item in state["tasks"].items()}
    failed = [key for key, status in statuses.items() if status in {"failed", "cancelled"}]
    if failed:
        return "blocked", "Parent has failed/cancelled tasks: " + ", ".join(failed)
    if state["status"] in {"stopped", "deadline_reached", "budget_paused", "blocked", "storage_paused"} or (parent / "STOP").exists():
        return "blocked", f"Parent is {state['status']}; no implicit resume or budget extension"
    completed = sum(status == "completed" for status in statuses.values())
    if completed != len(ids):
        return "waiting", f"Parent completed {completed}/{len(ids)}; waiting for training and validation"
    jobs = c.active_jobs(c.qstat(), getpass.getuser())
    workers = list(c.all_workers(state))
    if any(w.get("job_id") in jobs or any(w.get("job_name") and w["job_name"] == j.get("Job_Name")
           for j in jobs.values()) for w in workers):
        return "waiting", "Parent results complete; PBS allocations still exiting"
    if any(w["status"] == "submitting" or (w.get("job_id") and
           w.get("reconciled_job") != w["job_id"]) for w in workers):
        return "waiting", "Waiting for parent supervisor to reconcile final PBS evidence"
    if len(jobs) >= c.MAX_NODES or sum(c.allocated_nodes(j) for j in jobs.values()) >= c.MAX_NODES:
        return "waiting", "Account slots occupied; successor budget has not started"
    # A generic 'finished' can include failures; check actual committed outputs.
    c.verify_snapshot(parent, parent_plan)
    for task in parent_plan["tasks"]:
        cfg = read(task["config"])
        if not c.training_done(task, cfg):
            return "blocked", f"Missing complete training checkpoints: {task['id']}"
        for step in c.expected_checkpoints(cfg):
            if not c.validation_done(task, step):
                return "blocked", f"Missing committed validation: {task['id']}/{step}"
            path = Path(task["output"]) / "validation" / f"checkpoint-{step}"
            marker, report = read(path / "NIGHT_VALIDATED.json"), read(path / "metrics.json")
            if (marker.get("protocol_hash") != report.get("protocol_hash") or
                (cfg["evaluation"].get("mmlu_file") and "mmlu_predictions.jsonl" not in marker["prediction_sizes"])):
                return "blocked", f"Incomplete validation evidence: {task['id']}/{step}"
    return "ready", f"All {len(ids)} parent tasks and validations committed; PBS allocations settled"


def require_parent_ready(plan):
    status, reason = parent_readiness(plan)
    if status != "ready":
        raise ValueError(f"Successor cannot start ({status}): {reason}; use chain to wait in background")


def record(campaign, status, reason, **extra):
    value = dict(status=status, reason=reason, updated_at=time.time(), pid=os.getpid(),
                 host=os.uname().nodename, **extra)
    write(campaign / "handoff-state.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def watch(args):
    campaign = Path(args.campaign).resolve()
    plan = read(campaign / "plan.json")
    with locked(campaign / "handoff.lock", blocking=False):
        request = read(campaign / "handoff.json")
        if request["successor_plan_sha256"] != file_sha(campaign / "plan.json"):
            raise ValueError("Successor plan changed after arming")
        c.verify_snapshot(campaign, plan)
        last = None
        while not (campaign / "HANDOFF_STOP").exists():
            try:
                successor = read(campaign / "state.json")
                if (campaign / "STOP").exists() or successor["status"] == "stopped":
                    record(campaign, "stopped", "Successor was explicitly stopped; no automatic launch")
                    return 0
                if successor["started_at"] is not None:
                    if successor["status"] == "running":
                        # Re-entering after a crash between start and publication is safe.
                        c.start(argparse.Namespace(campaign=str(campaign)))
                        record(campaign, "launched", "Successor supervisor active; existing jobs adopted")
                        return 0
                    record(campaign, "done", "Successor already started; no automatic restart",
                           successor_status=successor["status"])
                    return 0
                status, reason = parent_readiness(plan)
                if (status, reason) != last:
                    record(campaign, status, reason)
                    last = status, reason
                else:
                    value = read(campaign / "handoff-state.json")
                    write(campaign / "handoff-state.json", dict(value, updated_at=time.time()))
                if status == "blocked":
                    return 2
                if status == "ready":
                    if (campaign / "HANDOFF_STOP").exists():
                        break
                    c.start(argparse.Namespace(campaign=str(campaign)))
                    record(campaign, "launched", "Successor supervisor started; budget begins now",
                           deadline=read(campaign / "state.json")["deadline"])
                    return 0
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                # A broken scheduler query or partial external state is never readiness.
                record(campaign, "waiting_error", repr(exc))
                last = None
            time.sleep(min(60, plan["poll_seconds"]))
        record(campaign, "stopped", "Handoff stopped; parent and successor jobs were not cancelled")
    return 0


def arm(args):
    campaign = Path(args.campaign).resolve()
    plan = read(campaign / "plan.json")
    if not plan.get("start_after_campaign"):
        raise ValueError("Campaign does not declare a pinned predecessor")
    c.verify_snapshot(campaign, plan)
    if (campaign / "STOP").exists() or read(campaign / "state.json")["status"] == "stopped":
        raise ValueError("Successor explicitly stopped; inspect before an explicit resume")
    request = {"parent_campaign": plan["start_after_campaign"],
        "parent_plan_sha256": plan["start_after_plan_sha256"],
        "successor_plan_sha256": file_sha(campaign / "plan.json")}
    with locked(campaign / "handoff-start.lock"):
        path = campaign / "handoff.json"
        if path.exists() and read(path) != request:
            raise ValueError("Existing handoff has different provenance")
        write(path, request)
        try:
            with locked(campaign / "handoff.lock", blocking=False):
                pass
        except BlockingIOError:
            print("Handoff already active; no duplicate watcher or PBS submission.")
            return 0
        if read(campaign / "state.json")["started_at"] is not None:
            print("Successor already started; use status/recover, not another handoff.")
            return 0
        (campaign / "HANDOFF_STOP").unlink(missing_ok=True)
        record(campaign, "armed", "Waiting in background; successor budget not started")
        with (campaign / "handoff.log").open("a") as log:
            process = subprocess.Popen([plan["python"], plan.get("controller_entry", plan["entry"]),
                "chain-watch", "--campaign", str(campaign)], cwd=plan["project_root"],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(.5)
        if process.poll() is not None and process.returncode != 0:
            raise RuntimeError("Handoff could not arm; inspect handoff.log and handoff-state.json")
        print(f"Handoff armed, PID {process.pid}; safe to disconnect SSH. Parent jobs unchanged.")
        status(argparse.Namespace(campaign=str(campaign)))
    return 0


def status(args):
    campaign = Path(args.campaign).resolve()
    value = read(campaign / "handoff-state.json") if (campaign / "handoff-state.json").exists() else {"status": "not_armed"}
    try:
        with locked(campaign / "handoff.lock", blocking=False):
            value["watcher_active"] = False
    except BlockingIOError:
        value["watcher_active"] = True
    current = read(campaign / "state.json")
    value.update(successor_status=current["status"], successor_started_at=current["started_at"],
                 successor_deadline=current["deadline"])
    print(json.dumps(value, indent=2, ensure_ascii=False))
    return value
