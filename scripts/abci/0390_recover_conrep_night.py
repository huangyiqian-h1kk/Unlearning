#!/usr/bin/env python3
"""Upgrade a stopped campaign's controller; preserve its frozen training identity.

Run from a fetched ref with python -I. No checkout/reset, model deletion, or PBS
submission occurs unless --restart is requested. --hours explicitly extends the
budget; by default the original campaign deadline is retained.
"""
import argparse
import contextlib
import copy
import datetime as dt
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

BASE_REF = "33d5f9c22106d43bbaa88cc3bc57ee23afbc4ac9"
PREVIOUS_REFS = ("da02594fdf973b4ef26d522c11ec128dc2c5b6aa",
                 "07cb3864f3874d3be7ec8d2cf956511ffb80aa2f")
ENTRY = "scripts/abci/0390_conrep_night.py"
SHELL = "scripts/abci/0390_conrep_night_worker.sh"
CONTROLLER = "src/conrep/night/campaign.py"
OWNED = (ENTRY, SHELL, CONTROLLER)
MAX_WORKERS = 3
PREFERRED = {v: worker for worker, values in enumerate(("ADGJ", "BEHK", "CFIL")) for v in values}
POLICY = {"version": "0390-night-recovery-v3-20min", "retry_admin_termination": True,
          "retry_terminated_queued": True, "interruption_retry_seconds": 1200,
          "interruption_max_retries": None, "other_max_retries": 3,
          "backoff_seconds": 60, "backoff_max_seconds": 300}


def read(path):
    return json.loads(Path(path).read_text())


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temp.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        temp.chmod(path.stat().st_mode & 0o777)
    os.replace(temp, path)


def write(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


@contextlib.contextmanager
def lock(path):
    with Path(path).open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root)


def verify_original(campaign, plan):
    source = read(campaign / "source.json")
    if source["source_hash"] != plan["source_hash"]:
        raise ValueError("Original source hash changed")
    for name, expected in source["files"].items():
        if digest(campaign / "code" / name) != expected:
            raise ValueError(f"Original frozen source changed: {name}")
    for name, expected in read(campaign / "inputs.json").items():
        if digest(name) != expected:
            raise ValueError(f"Input changed: {name}")
    for task in plan["tasks"]:
        if digest(task["config"]) != task["config_hash"]:
            raise ValueError(f"Task configuration changed: {task['id']}")
    for name, expected in read(campaign / "model-assets.json").items():
        stat = Path(name).stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"SFT asset changed: {name}")
    for name, expected in plan.get("controller_files", {}).items():
        if digest(name) != expected:
            raise ValueError(f"Installed controller changed: {name}")


def assert_no_live_workers(state):
    output = subprocess.run(["qstat", "-f", "-F", "json"], text=True,
                            capture_output=True, check=True, timeout=30)
    jobs = json.loads(output.stdout).get("Jobs") or {}
    workers = [*state["workers"].values(), *state.get("retired_workers", {}).values()]
    ids = {w.get("job_id") for w in workers}
    names = {w.get("job_name") for w in workers}
    live = [name for name, record in jobs.items()
            if record.get("Job_Owner", "").split("@")[0] == getpass.getuser()
            and record.get("job_state") in {"Q", "R", "H", "T", "W", "S", "E", "B"}
            and (name in ids or record.get("Job_Name") in names)]
    if live:
        raise RuntimeError(f"Campaign still has active PBS allocations; no code changed: {live}")
    if any(w["status"] == "submitting" and not w.get("job_id") for w in workers):
        raise RuntimeError("Unresolved submission must be reconciled before upgrading")


def migrate_three_workers(plan, state):
    """Change scheduling metadata only; preserve every task and allocation record."""
    revised, migrated = copy.deepcopy(plan), copy.deepcopy(state)
    revised["workers"] = MAX_WORKERS
    revised["recovery_policy"] = dict(POLICY)
    for task in revised["tasks"]:
        task["preferred_worker"] = PREFERRED.get(task.get("variant"), task.get("preferred_worker", 0) % MAX_WORKERS)
    for key in list(migrated["workers"]):
        if key not in {str(i) for i in range(MAX_WORKERS)}:
            retired = migrated.setdefault("retired_workers", {})
            if key in retired and retired[key] != migrated["workers"][key]:
                raise ValueError(f"Retired worker history collision: {key}")
            retired[key] = migrated["workers"].pop(key)
    for i in range(MAX_WORKERS):
        migrated["workers"].setdefault(str(i), {"job_id": None, "status": "new",
                                               "allocations": 0, "recovery_failures": 0})
    return revised, migrated


def upgrade(root, campaign, ref):
    root, campaign = Path(root).resolve(), Path(campaign).resolve()
    commit = git(root, "rev-parse", "--verify", ref + "^{commit}").decode().strip()
    payloads = {name: git(root, "show", f"{commit}:{name}") for name in OWNED}
    plan = read(campaign / "plan.json")
    if Path(plan["project_root"]).resolve() != root:
        raise ValueError("Campaign belongs to another checkout")
    verify_original(campaign, plan)
    if (plan.get("controller_ref") == commit and plan.get("workers") == MAX_WORKERS
            and set(read(campaign / "state.json")["workers"]) == {str(i) for i in range(MAX_WORKERS)}
            and all((root / n).read_bytes() == p for n, p in payloads.items())):
        print("Controller already installed; all original source/configuration checks passed.")
        return plan
    with lock(campaign / "start.lock"), lock(campaign / "supervisor.lock"), lock(campaign / "state.lock"):
        state = read(campaign / "state.json")
        assert_no_live_workers(state)
        plan = read(campaign / "plan.json")
        verify_original(campaign, plan)
        revised, migrated = migrate_three_workers(plan, state)
        # Only these package-owned files can be replaced, and only when their
        # content matches an actual installed release. Dirty server files stop
        # the entire upgrade before any checkout or plan mutation.
        accepted_refs = {BASE_REF, commit, *PREVIOUS_REFS}
        if plan.get("controller_ref"):
            accepted_refs.add(plan["controller_ref"])
        for name in OWNED:
            current = (root / name).read_bytes()
            if current not in {git(root, "show", f"{version}:{name}") for version in accepted_refs}:
                raise FileExistsError(f"Locally edited file preserved: {name}; no upgrade performed")
        destination = campaign / "controllers" / commit
        expected = dict(read(campaign / "source.json")["files"])
        expected.update({name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()})
        if not destination.exists():
            staging = destination.with_name(commit + ".incomplete-" + uuid.uuid4().hex)
            staging.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(campaign / "code", staging,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            for name, payload in payloads.items():
                atomic_bytes(staging / name, payload)
            staging.rename(destination)
        for name, expected_hash in expected.items():
            if digest(destination / name) != expected_hash:
                raise ValueError(f"Recovery controller snapshot collision: {name}")
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        backup = campaign / "recovery-backups" / stamp
        write(backup / "plan.json", plan)
        write(backup / "state.json", state)
        for name in OWNED:
            atomic_bytes(backup / "checkout" / name, (root / name).read_bytes())
        revised.update(controller_entry=str(destination / ENTRY),
                       controller_shell=str(destination / SHELL), controller_ref=commit,
                       controller_files={str(destination / n): h for n, h in expected.items()},
                       recovery_policy=dict(POLICY))
        # entry, shell, source_hash, task/config identities remain the originals.
        for name, payload in payloads.items():
            atomic_bytes(root / name, payload)
        write(campaign / "plan.json", revised)
        write(campaign / "state.json", migrated)
        write(backup / "upgrade.json", {"ref": commit, "controller": str(destination),
              "original_source_hash": plan["source_hash"], "policy": POLICY,
              "previous_workers": plan.get("workers"), "workers": MAX_WORKERS})
        print(json.dumps({"controller": str(destination), "backup": str(backup),
              "original_deadline": state.get("deadline"), "training_source_unchanged": True,
              "workers": MAX_WORKERS, "interruption_retry_seconds": 1200,
              "interruption_max_retries": None,
              "task_count": len(plan["tasks"]), "status": "upgraded; no PBS submitted"}, indent=2))
        return revised


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--hours", type=float)
    args = parser.parse_args()
    if args.hours is not None and (args.hours <= 0 or not args.restart):
        parser.error("A positive --hours requires --restart")
    plan = upgrade(args.root, args.campaign, args.ref)
    if args.restart:
        command = [plan["python"], plan["controller_entry"], "recover", "--campaign", str(args.campaign.resolve())]
        if args.hours is not None:
            command += ["--hours", str(args.hours)]
        subprocess.run(command, cwd=args.root, check=True)
        subprocess.run([plan["python"], plan["controller_entry"], "status", "--campaign",
                        str(args.campaign.resolve())], cwd=args.root, check=True)


if __name__ == "__main__":
    main()
