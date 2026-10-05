"""PBS job provenance and local log archives; no GPU or training imports."""

from __future__ import annotations

import argparse
import contextlib
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time


ARTIFACT_NAMES = {
    "TRAINING_COMPLETE.json", "COMPLETE.json", "metrics.json", "report.json",
    "resolved_config.json", "lineage.json", "trainer_state.json", "train.jsonl",
}
SCHEDULER_FIELDS = {
    "Job_Name", "Job_Owner", "job_state", "queue", "Resource_List",
    "resources_used", "Exit_status", "comment", "qtime", "stime", "mtime",
    "obittime", "Output_Path", "Error_Path", "exec_host", "exec_vnode",
}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_root(root):
    return Path(root) / "logs/0390"


def job_dir(root, job_id):
    if not re.fullmatch(r"[0-9]+(?:\.[A-Za-z0-9_-]+)+", job_id):
        raise ValueError(f"Expected a full PBS job ID, e.g. 1234567.pbs1: {job_id!r}")
    return log_root(root) / "runs" / job_id


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".capture-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode())


def git_state(root):
    def git(*args):
        try:
            result = subprocess.run(
                ["git", *args], cwd=root, text=True, capture_output=True, timeout=15,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "tracked_changes": git("status", "--porcelain", "--untracked-files=no"),
    }


def experiment_args(launch_args):
    return launch_args[launch_args.index("--") + 1:] if "--" in launch_args else launch_args


def entry_details(root, launch_args):
    """Snapshot the entry configuration; trainers can make further stage-specific changes."""
    argv = experiment_args(list(launch_args))
    details = {"argv": argv, "stage": argv[0] if argv else None}
    if "--nproc" in launch_args:
        details["nproc"] = int(launch_args[launch_args.index("--nproc") + 1])
    try:
        from .cli import parser, resolve
        # Suppress argparse diagnostics here; the actual command still diagnoses failures.
        with contextlib.redirect_stderr(io.StringIO()):
            args = parser().parse_args(argv)
        args.config = str(Path(root) / args.config)
        cfg = resolve(args)
        details.update(
            entry_config=cfg, config_path=args.config, method=args.method,
            checkpoint=args.checkpoint, checkpoint_root=args.checkpoint_root,
            resume=args.resume, retain_only=args.retain_only,
            model=Path(args.config).stem,
        )
        output = args.output
        if not output and args.stage in {"sft", "smoke", "unlearn", "baseline"}:
            output = cfg["run"]["output_dir"]
        if not output and args.stage == "falcon-layers":
            output = cfg.get("baseline", {}).get("layer_selection")
        if not output and args.stage == "relearn-augment":
            output = cfg.get("baseline", {}).get("augmented_file")
        details["output_dir"] = str(Path(root) / output) if output else None
    except (Exception, SystemExit) as exc:
        details["config_capture_error"] = str(exc)
    return details


def submission_snapshot(root, stage, model, run_id, rtype, nproc, walltime, extra):
    launch_args = ["--nproc", str(nproc), "--", stage,
                   "--config", f"configs/0390/{model}.yaml", *extra]
    if stage == "preflight":
        launch_args.append("--distributed")
    return {
        **entry_details(root, launch_args), "captured_at": now(), "git": git_state(root),
        "stage": stage, "model": model, "run_id": run_id,
        "job_name": ("0390_" + run_id)[:15], "rtype": rtype, "nproc": nproc,
        "requested_walltime": walltime, "checkout": str(root), "launch_args": launch_args,
    }


def record_submission(root, job_id, metadata, script):
    dest = job_dir(root, job_id)
    # This namespace is separate from runtime records: the job can start before qsub returns.
    metadata = dict(metadata)
    for field, option in (("queue", "q"), ("project", "P")):
        match = re.search(rf"^#PBS -{option} (\S+)$", script, re.M)
        if match:
            metadata[field] = match[1]
    write_json(dest / "submission.json", {**metadata, "job_id": job_id})
    atomic_bytes(dest / "submission.pbs", script.encode())
    if metadata.get("entry_config") is not None:
        write_json(dest / "submission-config.json", metadata["entry_config"])
    rebuild_index(root)


def snapshot_file(source, dest):
    """Atomic streaming copy. Original logs and checkpoints are never moved or removed."""
    source, dest = Path(source), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".capture-", dir=dest.parent)
    checksum = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(fd, "wb") as out, source.open("rb") as src:
            # Bounded by the size at open: a running log is an explicitly partial snapshot.
            remaining = os.fstat(src.fileno()).st_size
            while remaining:
                block = src.read(min(1024 * 1024, remaining))
                if not block:
                    break
                out.write(block)
                checksum.update(block)
                size += len(block)
                remaining -= len(block)
        os.replace(temporary, dest)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"source": str(source), "bytes": size, "sha256": checksum.hexdigest(),
            "captured_at": now()}


def capture_artifacts(root, dest, output, *, replace=False):
    if not output:
        return
    source = Path(root) / output
    if not source.exists() or ((dest / "artifacts.json").is_file() and not replace):
        return
    files = [source] if source.is_file() else (
        path for path in source.rglob("*") if path.is_file() and path.name in ARTIFACT_NAMES
    )
    records = {}
    for path in files:
        relative = path.name if source.is_file() else str(path.relative_to(source))
        # Generation dumps and model/optimizer weights are linked via output_dir, not copied.
        if path.stat().st_size > 20 * 1024 * 1024:
            continue
        records[relative] = snapshot_file(path, dest / "artifacts" / relative)
    write_json(dest / "artifacts.json", records)


def summarize(dest):
    records = {key: read_json(dest / (key + ".json"), {})
               for key in ("history", "submission", "runtime", "scheduler")}
    history, submitted, runtime, scheduler = (records[k] for k in records)
    recovered = read_json(dest / "recovered-entry.json", {})
    fields = ("job_name", "stage", "model", "method", "run_id", "queue", "rtype", "nproc",
              "output_dir", "checkpoint", "checkpoint_root", "requested_walltime")
    row = {"job_id": dest.name}
    for source in (recovered, history, submitted, runtime):
        row.update({key: source[key] for key in fields if source.get(key) is not None})
    for field, key in (("job_name", "Job_Name"), ("queue", "queue")):
        if scheduler.get(key):
            row[field] = scheduler[key]
    row["submitted_commit"] = submitted.get("git", {}).get("commit") or history.get("submitted_commit")
    row["started_commit"] = runtime.get("git", {}).get("commit")
    row["started_at"] = runtime.get("started_at")
    row["ended_at"] = runtime.get("ended_at")
    row.update(status="unknown", exit_code=None, status_source="unknown")
    if submitted:
        row.update(status="submitted", status_source="qsub")
    if history.get("observed_exit_code") is not None:
        row.update(status="completed" if history["observed_exit_code"] == 0 else "failed",
                   exit_code=history["observed_exit_code"], status_source="conversation")
    if scheduler.get("job_state"):
        row.update(status=scheduler["job_state"], exit_code=None, status_source="PBS")
    if runtime:
        row.update(status=runtime.get("status", "started"), exit_code=runtime.get("exit_code"),
                   status_source="launcher")
    if scheduler.get("job_state") == "F":
        code = scheduler.get("Exit_status")
        row.update(status=("completed" if code == 0 else "failed") if code is not None else "finished",
                   exit_code=code, status_source="PBS")
    elapsed = runtime.get("elapsed_seconds")
    elapsed_text = None
    if elapsed is not None:
        seconds = round(elapsed)
        elapsed_text = f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
    row["walltime"] = scheduler.get("resources_used", {}).get("walltime") or elapsed_text or history.get("observed_walltime")
    row["note"] = history.get("note", "")
    row["recovered_entry_notice"] = recovered.get("notice")
    row["files"] = [str(path.relative_to(dest)) for path in (
        dest / "console.log", dest / "pbs.stdout.log", dest / "pbs.stderr.log",
        dest / "artifacts.json", dest / "submission.pbs", dest / "recovered.pbs",
    ) if path.is_file()]
    row["records"] = {key: key + ".json" for key, value in records.items() if value}
    return row


def rebuild_index(root):
    base = log_root(root)
    base.mkdir(parents=True, exist_ok=True)
    with (base / "index.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = []
        for dest in sorted((base / "runs").glob("*")):
            if dest.is_dir():
                row = summarize(dest)
                write_json(dest / "job.json", row)
                rows.append(row)
        write_json(base / "index.json", rows)
        keys = ["job_id", "job_name", "stage", "model", "run_id", "rtype", "status",
                "exit_code", "status_source", "walltime", "submitted_commit", "started_commit", "output_dir"]
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=keys, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        atomic_bytes(base / "index.tsv", stream.getvalue().encode())
        lines = ["# 0390 job log index", "", f"Updated (UTC): {now()}", "",
                 "Status source distinguishes PBS records, launcher exit, and conversation evidence.", "",
                 "| PBS job | Stage / model / run | Resource | Status (source) | Exit | Walltime | Logs |",
                 "| --- | --- | --- | --- | --- | --- | --- |"]
        for row in rows:
            job = row["job_id"]
            links = " ".join(f"[{Path(p).name}](runs/{job}/{p})" for p in row["files"] if p.endswith(".log"))
            stage = " / ".join(str(row.get(k) or "?") for k in ("stage", "model", "run_id"))
            lines.append(f"| [{job}](runs/{job}/job.json) | {stage} | {row.get('rtype', '?')} | "
                         f"{row['status']} ({row['status_source']}) | {row['exit_code']} | {row.get('walltime', '?')} | {links} |")
        if (base / "setup/manifest.json").is_file():
            lines += ["", "Login-node logs (no PBS job ID): [manifest](setup/manifest.json)."]
        atomic_bytes(base / "INDEX.md", ("\n".join(lines) + "\n").encode())
        return rows


def run_logged(root, launch_args):
    """One supervisor outside torchrun: merge/tee all ranks and preserve the child's exit code."""
    job_id = os.environ["PBS_JOBID"]
    dest = job_dir(root, job_id)
    dest.mkdir(parents=True, exist_ok=True)
    # Keep previous runtime metadata if PBS reruns the same job ID; console.log is appended.
    previous = read_json(dest / "runtime.json")
    if previous:
        with (dest / "attempts.jsonl").open("a") as stream:
            stream.write(json.dumps(previous) + "\n")
    metadata = {
        **entry_details(root, launch_args), "job_id": job_id,
        "job_name": os.environ.get("PBS_JOBNAME"), "queue": os.environ.get("PBS_QUEUE"),
        "rtype": os.environ.get("RTYPE"), "launch_args": launch_args,
        "started_at": now(), "host": socket.gethostname(), "git": git_state(root),
        "python": sys.executable, "status": "running", "checkout": str(root),
        "versions": {},
    }
    for package in ("torch", "transformers", "peft", "accelerate", "deepspeed", "PyYAML"):
        try:
            metadata["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    write_json(dest / "runtime.json", metadata)
    if metadata.get("entry_config") is not None:
        write_json(dest / "runtime-config.json", metadata["entry_config"])
    rebuild_index(root)
    command = ["bash", str(Path(root) / "scripts/abci/0390_run.sh"), *launch_args]
    env = dict(os.environ, CONREP_LOG_ACTIVE="1", PYTHONUNBUFFERED="1")
    started = time.monotonic()
    child = None
    received_signal = None
    saved_handlers = {}

    def forward(signum, frame):
        nonlocal received_signal
        received_signal = signum
        if child is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signum)

    code = 1
    try:
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            saved_handlers[signum] = signal.signal(signum, forward)
        with (dest / "console.log").open("ab", buffering=0) as stream:
            header = f"[0390] job={job_id} start={metadata['started_at']} archive={dest}\n".encode()
            stream.write(header)
            print(header.decode(), end="", flush=True)
            child = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            if received_signal:
                forward(received_signal, None)
            with child.stdout:
                for block in iter(lambda: child.stdout.read1(65536), b""):
                    stream.write(block)
                    try:
                        sys.stdout.buffer.write(block)
                        sys.stdout.buffer.flush()
                    except BrokenPipeError:
                        pass
            code = child.wait()
            if code < 0:
                code = 128 - code
            if received_signal and code == 0:
                code = 128 + received_signal
            footer = f"\n[0390] job={job_id} exit_code={code} end={now()}\n".encode()
            stream.write(footer)
            print(footer.decode(), end="", flush=True)
    finally:
        if child is not None and child.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
        for signum, handler in saved_handlers.items():
            signal.signal(signum, handler)
        metadata.update(ended_at=now(), elapsed_seconds=round(time.monotonic() - started, 3),
                        exit_code=code, status="completed" if code == 0 else "failed",
                        received_signal=received_signal)
        # Archiving failures must not turn a failed training run into a successful job (or vice versa).
        try:
            write_json(dest / "runtime.json", metadata)
            capture_artifacts(root, dest, metadata.get("output_dir"), replace=True)
            rebuild_index(root)
        except Exception as exc:
            print(f"[0390] archive warning for {job_id}: {exc}", file=sys.stderr)
    return code


def query_scheduler(job_id):
    try:
        result = subprocess.run(["qstat", "-fx", "-F", "json", job_id],
                                text=True, capture_output=True, timeout=20)
        if result.returncode:
            return None, result.stderr.strip() or f"qstat exit {result.returncode}"
        jobs = json.loads(result.stdout).get("Jobs", {})
        value = jobs.get(job_id)
        if value is None:
            return None, "Job is no longer in scheduler history"
        filtered = {key: value[key] for key in SCHEDULER_FIELDS if key in value}
        if "Exit_status" in filtered:
            filtered["Exit_status"] = int(filtered["Exit_status"])
        return filtered, None
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)


def recover_script(root, job_id, dest):
    for marker in (log_root(root) / "jobs").glob("*.jobid"):
        if marker.read_text().strip() == job_id and marker.with_suffix(".pbs").is_file():
            snapshot_file(marker.with_suffix(".pbs"), dest / "recovered.pbs")
            # A recovered script is evidence collected now, not a guaranteed submission snapshot.
            script = marker.with_suffix(".pbs").read_text()
            identity = re.match(r"0390_(.+)_(llama3b|qwen7b)_(.+)\.pbs$", marker.with_suffix(".pbs").name)
            name = re.search(r"^#PBS -N (\S+)$", script, re.M)
            for line in script.splitlines():
                if not line.startswith("bash "):
                    continue
                tokens = shlex.split(line)
                if len(tokens) > 2 and tokens[0] == "bash" and tokens[1].endswith("0390_run.sh"):
                    details = entry_details(root, tokens[2:])
                    if identity:
                        details.update(stage=identity[1], model=identity[2], run_id=identity[3])
                    if name:
                        details["job_name"] = name[1]
                    return details
    return {}


def collect_job(root, job_id, history=None):
    dest = job_dir(root, job_id)
    dest.mkdir(parents=True, exist_ok=True)
    if history:
        write_json(dest / "history.json", history)
    recovered = recover_script(root, job_id, dest)
    if recovered:
        write_json(dest / "recovered-entry.json", {
            **recovered, "captured_at": now(),
            "notice": "Resolved against today's files; not a historical config snapshot",
        })
    scheduler, error = query_scheduler(job_id)
    write_json(dest / "scheduler-query.json", {"queried_at": now(), "error": error})
    if scheduler:
        write_json(dest / "scheduler.json", scheduler)
    else:
        scheduler = read_json(dest / "scheduler.json", {})
    row = summarize(dest)
    name = scheduler.get("Job_Name") or row.get("job_name")
    if not name:
        pbs = dest / "recovered.pbs"
        match = re.search(r"^#PBS -N (\S+)$", pbs.read_text(), re.M) if pbs.is_file() else None
        name = match.group(1) if match else None
    captured = read_json(dest / "captured-logs.json", {})
    number = job_id.split(".")[0]
    for channel, letter, key in (("stdout", "o", "Output_Path"), ("stderr", "e", "Error_Path")):
        candidates = []
        if name and re.fullmatch(r"[A-Za-z0-9_-]+", name):
            candidates += [Path.home() / f"{name}.{letter}{number}", Path(root) / f"{name}.{letter}{number}"]
        if scheduler.get(key):
            # PBS paths have a hostname prefix. Only inspect the local path; never scp.
            value = scheduler[key].split(":", 1)[-1]
            if value.startswith("/"):
                candidates.append(Path(value))
        source = next((path for path in candidates if path.is_file()), None)
        if source:
            captured[channel] = snapshot_file(source, dest / f"pbs.{channel}.log")
        elif channel not in captured:
            captured[channel] = {"missing": True, "searched": [str(p) for p in candidates]}
    write_json(dest / "captured-logs.json", captured)
    output = row.get("output_dir") or recovered.get("output_dir")
    if row["status"] not in {"Q", "R", "H", "T", "W", "S", "E", "B", "running", "submitted"}:
        capture_artifacts(root, dest, output)
    has_log = (dest / "console.log").is_file() or (dest / "pbs.stdout.log").is_file()
    print(f"{job_id}: log={'found' if has_log else 'MISSING'}; "
          f"status={row['status']} ({row['status_source']}); "
          f"PBS={'captured' if not error else 'unavailable'}")


def collect(root, history_file=None, job_ids=()):
    histories = {}
    if history_file:
        history_path = Path(root) / history_file
        if not history_path.is_file():
            raise FileNotFoundError(f"Private history file not found: {history_path}. "
                                    "Omit --history to discover existing .jobid files.")
        seed = read_json(history_path, {})
        histories = {row["job_id"]: row for row in seed.get("jobs", [])}
    identifiers = set(job_ids)
    if not job_ids:
        identifiers.update(histories)
        identifiers.update(path.name for path in (log_root(root) / "runs").glob("*") if path.is_dir())
        for marker in (log_root(root) / "jobs").glob("*.jobid"):
            identifiers.add(marker.read_text().strip())
    errors = {}
    for identifier in sorted(identifiers):
        try:
            collect_job(root, identifier, histories.get(identifier))
        except Exception as exc:
            errors[identifier] = str(exc)
            print(f"{identifier}: collection error: {exc}", file=sys.stderr)
    # Login-node setup/data logs have no PBS job ID. Keep them visibly separate.
    setup = log_root(root) / "setup"
    records = read_json(setup / "manifest.json", {})
    for source in sorted(log_root(root).glob("*.log")):
        records[source.name] = snapshot_file(source, setup / source.name)
    if records:
        write_json(setup / "manifest.json", records)
    rows = rebuild_index(root)
    write_json(log_root(root) / "collection-report.json", {
        "collected_at": now(), "requested_job_ids": sorted(identifiers), "errors": errors,
        "jobs_without_stdout": [row["job_id"] for row in rows
                                if not {"console.log", "pbs.stdout.log"}.intersection(row["files"])],
        "setup_logs": sorted(records),
    })
    print(f"Indexed {len(rows)} jobs: {log_root(root) / 'INDEX.md'}")
    return 1 if errors else 0


def main(root, argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Internal PBS wrapper; automatically used by 0390_run.sh")
    run.add_argument("args", nargs=argparse.REMAINDER)
    capture = sub.add_parser("collect", help="Copy PBS logs, query status, and refresh lightweight artifacts")
    capture.add_argument("job_ids", nargs="*")
    capture.add_argument("--history", nargs="?", const="logs/0390/history.json",
                         metavar="PRIVATE_JSON", help="Import a private history ledger; default: logs/0390/history.json")
    sub.add_parser("list", help="Print the local index; does not query PBS")
    show = sub.add_parser("show", help="Print one job's manifest")
    show.add_argument("job_id")
    args = p.parse_args(argv)
    if args.command == "run":
        launch_args = args.args[1:] if args.args[:1] == ["--"] else args.args
        return run_logged(root, launch_args)
    if args.command == "collect":
        return collect(root, args.history, args.job_ids)
    elif args.command == "list":
        rows = rebuild_index(root)
        keys = ["job_id", "stage", "model", "run_id", "rtype", "status", "exit_code", "status_source"]
        print("\t".join(keys))
        for row in rows:
            print("\t".join(str(row.get(key, "?")) for key in keys))
    elif args.command == "show":
        dest = job_dir(root, args.job_id)
        if not dest.is_dir():
            p.error(f"Job not archived: {args.job_id}; run collect first")
        print(json.dumps(summarize(dest), ensure_ascii=False, indent=2))
    return 0
