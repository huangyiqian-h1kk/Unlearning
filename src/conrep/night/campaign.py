"""Three single-node PBS workers, a persistent supervisor, and a durable task queue.

The entry point is additive: existing trainers, validators and submitters are
not overwritten. Prepare freezes the *actual* server source tree, including
uncommitted changes, and all workers import that frozen snapshot.
"""

import argparse
import contextlib
import copy
import csv
import datetime as dt
import getpass
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

from .io import read, write, sha, file_sha, locked, latest_checkpoint, complete_checkpoint, training_identity

NAME = "0390-conrep-night-v1"
MAX_NODES = 3
PAUSE = 75
ENTRY = "scripts/abci/0390_conrep_night.py"
SHELL = "scripts/abci/0390_conrep_night_worker.sh"
BLOCKED_WORKER_STATES = {"cancelled", "unknown_failure", "retry_exhausted",
                         "admin_terminated", "placement_failure", "policy_denied",
                         "admin_stopped", "resource_rejected"}
RECOVERY_VERSION = "0390-night-recovery-v3-20min"
RECOVERY_DEFAULTS = {"version": RECOVERY_VERSION, "retry_admin_termination": True,
                     "retry_terminated_queued": True, "interruption_retry_seconds": 1200,
                     "interruption_max_retries": None, "other_max_retries": 3,
                     "backoff_seconds": 60, "backoff_max_seconds": 300}
UNLIMITED_INTERRUPTION_REASONS = {"admin_terminated", "terminated_while_queued"}
BASE_CAMPAIGN = "results/validated_v2/0390/final-unlearn-v5-parallel3-seed42"
VARIANTS = {
    "A": {},
    "B": {"conrep.forget_cl_weight": 5.0},
    "C": {"conrep.protected_positive": True},
    "D": {"conrep.forget_cl_weight": 5.0, "conrep.protected_positive": True},
    "E": {"conrep.forget_cl_weight": 2.0},
    "F": {"conrep.forget_cl_weight": 5.0, "conrep.views": 8},
    "G": {"conrep.forget_cl_weight": 5.0, "conrep.specified_lm_weight": 0.0},
    "H": {"conrep.forget_cl_weight": 5.0, "conrep.protected_positive": True,
          "conrep.specified_lm_weight": 0.0},
    "I": {"conrep.forget_cl_weight": 5.0, "lora.lora_dropout": 0.1},
    "J": {"conrep.forget_cl_weight": 5.0, "lora.r": 64, "lora.lora_alpha": 128},
    "K": {"conrep.forget_cl_weight": 5.0, "lora.r": 128, "lora.lora_alpha": 256},
    "L": {"conrep.forget_cl_weight": 5.0, "conrep.corruption_rate": 0.5},
}
PREFERRED = {v: worker for worker, values in enumerate(("ADGJ", "BEHK", "CFIL")) for v in values}


def specs():
    result = []
    groups = [("gemma2_9b", 42, "ABCD", 0), ("gemma2_9b", 42, "EFGH", 1),
              ("gemma2_9b", 42, "IJKL", 2), ("gemma2_9b", 43, "ABCDEFGH", 3),
              ("gemma2_9b", 44, "ABCD", 4), ("llama8b", 42, "ABCD", 5),
              ("llama8b", 43, "ABCD", 6)]
    for model, seed, variants, priority in groups:
        for variant in variants:
            result.append({"id": f"{model}-{variant}-s{seed}", "model": model,
                           "variant": variant, "seed": seed, "priority": priority,
                           "preferred_worker": PREFERRED[variant]})
    return result


def apply_variant(base, variant, seed, output):
    cfg = copy.deepcopy(base)
    cfg["run"].update(seed=seed, output_dir=str(output))
    cfg["unlearn"].update(max_steps=125, save_steps=10)
    cfg["lora"].update(r=32, lora_alpha=64, lora_dropout=0.05)
    cfg["conrep"].update(forget_cl_weight=1.0, views=4, negative_views=4,
        corruption_rate=0.7, specified_cl_weight=1.0, general_cl_weight=1.0,
        specified_lm_weight=1.0, general_lm_weight=1.0,
        protected_positive=False, protected_positive_probability=0.5)
    for path, value in VARIANTS[variant].items():
        node = cfg
        keys = path.split(".")
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
    # The selected-checkpoint pruner is never invoked by this entry point.
    cfg.pop("storage", None)
    return cfg


def absolute_paths(value, project):
    if isinstance(value, dict):
        return {key: absolute_paths(v, project) for key, v in value.items()}
    if isinstance(value, list):
        return [absolute_paths(v, project) for v in value]
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        candidate = project / expanded
        if candidate.exists():
            return str(candidate.resolve())
        return expanded
    return value


def input_files(cfg):
    files = [Path(cfg["data"]["prepared_dir"]) / (name + ".jsonl")
             for name in ("forget", "retain", "general")]
    for split in ("forget", "retain"):
        files.append(Path(cfg["data"][split + "_generation"]))
        files += [Path(v) for v in cfg["data"][split + "_mcq"].values() if v]
    if cfg["evaluation"].get("mmlu_file"):
        files.append(Path(cfg["evaluation"]["mmlu_file"]))
    return sorted(set(files))


def source_snapshot(project, campaign):
    dest = campaign / "code"
    shutil.copytree(project / "src", dest / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in (ENTRY, SHELL):
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, target)
    files = {str(p.relative_to(dest)): file_sha(p) for p in sorted(dest.rglob("*")) if p.is_file()}
    provenance = {"files": files, "source_hash": sha(files)}
    for key, command in (("commit", ["git", "rev-parse", "HEAD"]),
                         ("status", ["git", "status", "--short"]),
                         ("diff", ["git", "diff", "--binary", "--", "src", "scripts"])):
        result = subprocess.run(command, cwd=project, text=True, capture_output=True)
        provenance[key] = result.stdout if result.returncode == 0 else result.stderr
    write(campaign / "source.json", provenance)
    return provenance["source_hash"]


def prepare(args):
    from experiments.config import load_config, read_rows
    from .positives import audit
    project = Path(args.project_root).resolve()
    campaign = Path(args.campaign).resolve()
    if campaign.exists() and any(campaign.iterdir()):
        raise FileExistsError(f"Campaign already exists; use status/start/resume: {campaign}")
    if not (project / "src/experiments/validation.py").is_file():
        raise ValueError("Run prepare in the server's current Unlearning checkout")
    configs, data_hashes, audits, model_assets = {}, {}, {}, {}
    for model, path in (("gemma2_9b", args.gemma_config), ("llama8b", args.llama_config)):
        cfg = absolute_paths(load_config(project / path), project)
        checkpoint = Path(cfg["unlearn"]["checkpoint"])
        if not (checkpoint / "config.json").exists() or (checkpoint / "adapter_config.json").exists():
            raise ValueError(f"Expected the approved full SFT final, got {checkpoint}")
        if not list(checkpoint.glob("*.safetensors")) and not list(checkpoint.glob("pytorch_model*.bin")):
            raise ValueError(f"SFT checkpoint has no weights: {checkpoint}")
        model_assets.update({str(p): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                             for p in checkpoint.iterdir() if p.is_file()})
        if cfg["unlearn"]["batch_sizes"] != {"forget": 8, "retain": 16, "general": 32}:
            raise ValueError("Expected unchanged global contrastive batches 8/16/32")
        if cfg["conrep"].get("specified_positive") != "dropout":
            raise ValueError("This matrix expects the archived specified_positive=dropout setup")
        if cfg["evaluation"].get("partition") != "validation":
            raise ValueError("Overnight tuning must use the frozen validation partition")
        if cfg["evaluation"].get("limit"):
            raise ValueError("Full checkpoint validation is required; evaluation.limit must be null")
        hashes = {str(p): file_sha(p) for p in input_files(cfg)}
        data_hashes.update(hashes)
        for group in ("forget", "retain", "general"):
            if len(read_rows(Path(cfg["data"]["prepared_dir"]) / f"{group}.jsonl")) < cfg["unlearn"]["batch_sizes"][group]:
                raise ValueError(f"{model}/{group} has fewer rows than its global batch")
        rows = read_rows(Path(cfg["data"]["prepared_dir"]) / "retain.jsonl")
        audits[model] = audit(rows)
        configs[model] = cfg
    # Never silently run C/D/H as identical clean positives.
    if any(x["eligible_rows"] == 0 for x in audits.values()):
        raise ValueError("Protected-positive coverage is zero. No jobs submitted; inspect retain text structure.")
    campaign.mkdir(parents=True, exist_ok=True)
    source_hash = source_snapshot(project, campaign)
    write(campaign / "positive-audit.json", audits)
    write(campaign / "inputs.json", data_hashes)
    write(campaign / "model-assets.json", model_assets)
    tasks = []
    for spec in specs():
        task_dir = campaign / "experiments" / spec["id"]
        cfg = apply_variant(configs[spec["model"]], spec["variant"], spec["seed"], task_dir / "training")
        cfg["night"] = {"source_hash": source_hash, "data_hashes": data_hashes,
                        "model_assets_hash": sha(model_assets),
                        "campaign": str(campaign), "experiment": spec["id"]}
        config_file = campaign / "configs" / (spec["id"] + ".json")
        write(config_file, cfg)
        tasks.append(dict(spec, config=str(config_file), output=str(task_dir),
                          identity=training_identity(cfg), config_hash=file_sha(config_file)))
    plan = {"schema": NAME, "project_root": str(project), "campaign": str(campaign),
            "python": sys.executable, "entry": str(campaign / "code" / ENTRY),
            "shell": str(campaign / "code" / SHELL), "workers": MAX_NODES, "world_size": 8,
            "queue": "R9920261000", "account": "gcg51557", "rtype": "rt_HF",
            "hours": args.hours, "walltime": "06:00:00", "poll_seconds": 60,
            "max_attempts": 3, "recovery_policy": dict(RECOVERY_DEFAULTS),
            "source_hash": source_hash, "tasks": tasks,
            "initial_task_seconds": 3600, "reserve_bytes": int(args.reserve_gb * 10**9)}
    write(campaign / "plan.json", plan)
    write(campaign / "state.json", {"status": "prepared", "started_at": None, "deadline": None,
          "tasks": {t["id"]: {"status": "pending", "attempts": 0, "failures": 0} for t in tasks},
          "workers": {str(i): {"job_id": None, "status": "new", "allocations": 0,
                               "recovery_failures": 0} for i in range(MAX_NODES)}})
    print(json.dumps({"campaign": str(campaign), "experiments": len(tasks),
                      "workers": MAX_NODES,
                      "protected_coverage": {m: a["eligible_fraction"] for m, a in audits.items()},
                      "status": "prepared; no PBS jobs submitted"}, ensure_ascii=False, indent=2))


@contextlib.contextmanager
def state_transaction(campaign):
    with locked(campaign / "state.lock"):
        state = read(campaign / "state.json")
        yield state
        write(campaign / "state.json", state)


def event(campaign, **values):
    row = {"time": dt.datetime.now(dt.timezone.utc).isoformat(), **values}
    line = json.dumps(row, ensure_ascii=False)
    with locked(campaign / "events.lock"):
        with (campaign / "events.jsonl").open("a") as stream:
            stream.write(line + "\n")
    # PBS console and supervisor.log must show progress and terminal failures,
    # even when the child trainer's detailed output lives in a separate file.
    print(line, flush=True)


def check_worker_limit(plan):
    if plan.get("workers") != MAX_NODES:
        raise ValueError(f"Campaign worker limit is {plan.get('workers')}; expected {MAX_NODES}. "
                         "Prepare a new three-node campaign or use the controller-only recovery upgrader.")


def verify_snapshot(campaign, plan):
    check_worker_limit(plan)
    for name, expected in read(campaign / "source.json")["files"].items():
        if file_sha(campaign / "code" / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    # A controller-only repair has separate provenance. Training still runs
    # through the original entry/source hash, so old checkpoints remain valid.
    for name, expected in plan.get("controller_files", {}).items():
        if file_sha(name) != expected:
            raise ValueError(f"Recovery controller changed: {name}")
    for name, expected in read(campaign / "inputs.json").items():
        if file_sha(name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    for task in plan["tasks"]:
        if file_sha(task["config"]) != task["config_hash"]:
            raise ValueError(f"Frozen configuration changed: {task['id']}")
    for name, expected in read(campaign / "model-assets.json").items():
        current = Path(name).stat()
        if current.st_size != expected["size"] or current.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"SFT asset changed since prepare: {name}")


def claim(campaign, plan, worker_id, allocation_end):
    now = time.time()
    with state_transaction(campaign) as state:
        if state["status"] != "running" or (campaign / "STOP").exists():
            return None
        remaining = min(state["deadline"], allocation_end) - now
        completed = [t.get("elapsed_seconds", 0) for t in state["tasks"].values()
                     if t["status"] == "completed" and t.get("elapsed_seconds", 0) > 0]
        estimate = max(2700, max(completed[-8:], default=plan["initial_task_seconds"]))
        candidates = sorted(plan["tasks"], key=lambda t: (
            state["tasks"][t["id"]]["status"] != "paused", t["priority"],
            t["preferred_worker"] != worker_id, t["id"]))
        for task in candidates:
            record = state["tasks"][task["id"]]
            if not ready_to_claim(record):
                continue
            # Allow incomplete validation to finish near the end, but never
            # knowingly begin a new hour-long training+validation run there.
            cfg = read(task["config"])
            trained = training_done(task, cfg)
            needed = 1200 if trained else 1.15 * estimate + 300
            if remaining < needed:
                continue
            if task["priority"] >= 2 and now >= state["deadline"] - 5400 and not trained:
                continue
            record.update(status="running", worker=worker_id, job_id=os.environ.get("PBS_JOBID"),
                          attempts=record["attempts"] + 1, claimed_at=now, stage="starting")
            record.setdefault("job_ids", []).append(os.environ.get("PBS_JOBID"))
            state["workers"][str(worker_id)]["task"] = task["id"]
            return task
    return None


def ready_to_claim(record):
    # A worker may checkpoint and return before PBS records a user's qdel.
    # Wait for final scheduler evidence before any worker resumes that task.
    return record["status"] == "pending" or (record["status"] == "paused"
        and time.time() >= record.get("retry_after", 0)
        and (not record.get("job_id") or record.get("reconciled_job") == record["job_id"]))


def expected_checkpoints(cfg):
    last, every = cfg["unlearn"]["max_steps"], cfg["unlearn"]["save_steps"]
    return sorted(set(range(every, last + 1, every)) | {last})


def training_done(task, cfg):
    root = Path(cfg["run"]["output_dir"])
    try:
        marker = read(root / "TRAINING_COMPLETE.json")
        if marker["identity"] != task["identity"] or marker["steps"] != cfg["unlearn"]["max_steps"]:
            return False
        return all(complete_checkpoint(root / f"checkpoint-{step}", identity=task["identity"], world=8)
                   for step in expected_checkpoints(cfg))
    except (OSError, ValueError, KeyError):
        return False


def validation_done(task, step):
    root = Path(task["output"]) / "validation" / f"checkpoint-{step}"
    try:
        marker = read(root / "NIGHT_VALIDATED.json")
        return (marker["identity"] == task["identity"]
                and marker["metrics_sha256"] == file_sha(root / "metrics.json")
                and "predictions.jsonl" in marker["prediction_sizes"]
                and all((root / name).is_file() and (root / name).stat().st_size == size
                        for name, size in marker["prediction_sizes"].items()))
    except (OSError, KeyError, ValueError):
        return False


def validate_one(config, checkpoint, output):
    from experiments.validation import run
    cfg = read(config)
    identity = training_identity(cfg)
    if not complete_checkpoint(checkpoint, identity=identity, world=8):
        raise ValueError("Refusing validation of incomplete/incompatible weights")
    report = run(cfg, checkpoint, output)
    if not (Path(output) / "predictions.jsonl").is_file():
        raise ValueError("Validator did not preserve per-example predictions")
    if cfg["evaluation"].get("mmlu_file") and not (Path(output) / "mmlu_predictions.jsonl").is_file():
        raise ValueError("Validator did not preserve MMLU predictions")
    write(Path(output) / "NIGHT_VALIDATED.json", {"identity": identity,
          "metrics_sha256": file_sha(Path(output) / "metrics.json"),
          "prediction_sizes": {name: (Path(output) / name).stat().st_size
                               for name in ("predictions.jsonl", "mmlu_predictions.jsonl")
                               if (Path(output) / name).is_file()},
          "protocol_hash": report.get("protocol_hash")})


def terminate(process, sig=signal.SIGTERM):
    if process.poll() is None:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass


def run_child(command, log, *, project, deadline, stop_file, interrupted, env=None, pause_marker=None):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open("a") as stream:
        stream.write("\nCOMMAND " + shlex.join(map(str, command)) + "\n")
        stream.flush()
        launch_time = time.time()
        process = subprocess.Popen(list(map(str, command)), cwd=project, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        requested = None
        while process.poll() is None:
            if interrupted() or time.time() >= deadline - 90:
                Path(stop_file).touch()
                requested = requested or time.time()
                if time.time() - requested > 45:
                    terminate(process)
                if time.time() - requested > 60 or time.time() >= deadline - 5:
                    terminate(process, signal.SIGKILL)
            time.sleep(1)
        # PBS can terminate the child before this loop observes the parent's
        # signal. Do not turn that ordering into a deterministic task failure.
        if requested is not None or interrupted():
            return PAUSE
        if pause_marker and Path(pause_marker).exists():
            marker = read(pause_marker)
            if marker.get("created_at", 0) >= launch_time and process.returncode == 0:
                return PAUSE
        return process.returncode


def train_command(plan, config, *, resume=None, deadline=None, stop_file=None, stop_after=None):
    command = [plan["python"], "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               "--nproc_per_node=8", plan["entry"], "train", "--config", str(config)]
    for key, value in (("--resume", resume), ("--deadline", deadline), ("--stop-file", stop_file),
                       ("--stop-after-step", stop_after)):
        if value is not None:
            command += [key, str(value)]
    return command


def gpu_smoke(campaign, plan, worker_id, deadline, interrupted):
    event(campaign, event="smoke-start", worker=worker_id,
          job_id=os.environ.get("PBS_JOBID"))
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        raise RuntimeError("Each worker requires one allocated HF node with 8 visible GPUs")
    root = campaign / "smoke" / f"worker-{worker_id}-{os.environ.get('PBS_JOBID', 'local')}"
    cfg = read(plan["tasks"][worker_id]["config"])
    cfg["unlearn"].update(max_steps=2, save_steps=1)
    cfg["run"]["output_dir"] = str(root / "training")
    config = root / "config.json"
    write(config, cfg)
    stop_file = root / "STOP"
    command = train_command(plan, config, deadline=deadline - 30, stop_file=stop_file, stop_after=1)
    event(campaign, event="smoke-save", worker=worker_id, log=str(root / "first.log"))
    code = run_child(command, root / "first.log", project=plan["project_root"], deadline=deadline,
                     stop_file=stop_file, interrupted=interrupted,
                     pause_marker=root / "training/PAUSED.json")
    if code != PAUSE or interrupted():
        raise RuntimeError(f"GPU checkpoint smoke failed in first stage: {code}")
    checkpoint = latest_checkpoint(root / "training", identity=training_identity(cfg), world=8)
    if checkpoint is None:
        raise RuntimeError("GPU smoke did not produce a complete all-rank checkpoint")
    command = train_command(plan, config, resume=checkpoint, deadline=deadline - 30, stop_file=stop_file)
    event(campaign, event="smoke-resume", worker=worker_id, log=str(root / "resume.log"))
    code = run_child(command, root / "resume.log", project=plan["project_root"], deadline=deadline,
                     stop_file=stop_file, interrupted=interrupted,
                     pause_marker=root / "training/PAUSED.json")
    if code != 0 or not (root / "training/TRAINING_COMPLETE.json").exists():
        raise RuntimeError(f"GPU resume smoke failed: {code}")
    write(root / "PASSED.json", {"world_size": 8, "resume_from_step": 1, "final_step": 2})
    event(campaign, event="smoke-passed", worker=worker_id, path=str(root / "PASSED.json"))


def validate_all(campaign, plan, task, cfg, *, deadline, interrupted):
    pending = [step for step in expected_checkpoints(cfg) if not validation_done(task, step)]
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
    if len(devices) < 8:
        raise RuntimeError("Validation requires 8 distinct allocated visible GPUs")
    active, failed = {}, []
    stopping_at = None
    try:
        while pending or active:
            stopping = interrupted() or time.time() >= deadline - 90
            if stopping:
                stopping_at = stopping_at or time.time()
                for process, stream, step in active.values():
                    terminate(process, signal.SIGKILL if time.time() - stopping_at > 20 else signal.SIGTERM)
            for index in range(8):
                if index in active or not pending or stopping:
                    continue
                step = pending.pop(0)
                output = Path(task["output"]) / "validation" / f"checkpoint-{step}"
                output.mkdir(parents=True, exist_ok=True)
                stream = (output / "console.log").open("a")
                env = os.environ.copy()
                for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
                    env.pop(key, None)
                env["CUDA_VISIBLE_DEVICES"] = devices[index]
                command = [plan["python"], plan["entry"], "validate-one", "--config", task["config"],
                           "--checkpoint", str(Path(cfg["run"]["output_dir"]) / f"checkpoint-{step}"),
                           "--output", str(output)]
                process = subprocess.Popen(command, cwd=plan["project_root"], env=env,
                                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                active[index] = (process, stream, step)
            for index, (process, stream, step) in list(active.items()):
                code = process.poll()
                if code is not None:
                    stream.close()
                    del active[index]
                    if code != 0 or not validation_done(task, step):
                        failed.append(step)
            if stopping and not active:
                return PAUSE
            time.sleep(1)
        return 1 if failed else 0
    finally:
        for process, stream, step in active.values():
            terminate(process)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                terminate(process, signal.SIGKILL)
                process.wait()
            stream.close()


def worker(args):
    campaign = Path(args.campaign).resolve()
    plan = read(campaign / "plan.json")
    verify_snapshot(campaign, plan)
    worker_id = args.worker
    if worker_id not in range(plan["workers"]):
        raise ValueError("Worker ID is not present in this campaign")
    signalled = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: signalled.__setitem__(0, True))
    start = time.time()
    state = read(campaign / "state.json")
    if state["status"] != "running" or (campaign / "STOP").exists():
        return 0
    allocation_end = min(state["deadline"], start + seconds(plan["walltime"]) - 60)
    def interrupted():
        return signalled[0] or (campaign / "STOP").exists()
    with state_transaction(campaign) as current:
        current["workers"][str(worker_id)].update(status="running", pid=os.getpid(),
            host=os.uname().nodename, started_at=start, job_id=os.environ.get("PBS_JOBID"))
    event(campaign, worker=worker_id, event="worker-start", job_id=os.environ.get("PBS_JOBID"))
    try:
        gpu_smoke(campaign, plan, worker_id, min(allocation_end, time.time() + 900), interrupted)
    except Exception as exc:
        with state_transaction(campaign) as current:
            current["workers"][str(worker_id)].update(stage="smoke-failed", error=repr(exc))
        event(campaign, event="smoke-error", worker=worker_id,
              job_id=os.environ.get("PBS_JOBID"), error=repr(exc))
        raise
    while not interrupted() and time.time() < allocation_end - 90:
        if shutil.disk_usage(campaign).free < plan["reserve_bytes"]:
            event(campaign, worker=worker_id, event="storage-reserve-reached")
            with state_transaction(campaign) as current:
                current["status"] = "storage_paused"
            break
        task = claim(campaign, plan, worker_id, allocation_end)
        if task is None:
            break
        cfg, task_start = read(task["config"]), time.time()
        root = Path(task["output"])
        root.mkdir(parents=True, exist_ok=True)
        stop_file = root / "PAUSE_TRAINING"
        stop_file.unlink(missing_ok=True)
        code = 0
        try:
            if not training_done(task, cfg):
                checkpoint = latest_checkpoint(cfg["run"]["output_dir"], identity=task["identity"], world=8)
                with state_transaction(campaign) as current:
                    current["tasks"][task["id"]]["stage"] = "training"
                event(campaign, task=task["id"], event="train", resume=str(checkpoint) if checkpoint else None,
                      job_id=os.environ.get("PBS_JOBID"))
                code = run_child(train_command(plan, task["config"], resume=checkpoint,
                                 deadline=allocation_end - 120, stop_file=stop_file),
                                 root / "training-console.log", project=plan["project_root"],
                                 deadline=allocation_end, stop_file=stop_file, interrupted=interrupted,
                                 pause_marker=Path(cfg["run"]["output_dir"]) / "PAUSED.json")
                if code == 0 and not training_done(task, cfg):
                    raise RuntimeError("Trainer exited successfully without all planned complete checkpoints")
            if code == 0:
                with state_transaction(campaign) as current:
                    current["tasks"][task["id"]]["stage"] = "validation"
                event(campaign, task=task["id"], event="validation", job_id=os.environ.get("PBS_JOBID"))
                code = validate_all(campaign, plan, task, cfg, deadline=allocation_end, interrupted=interrupted)
        except Exception as exc:
            code = 1
            event(campaign, task=task["id"], event="task-error", error=repr(exc))
        if code != 0 and interrupted():
            code = PAUSE
        with state_transaction(campaign) as current:
            record = current["tasks"][task["id"]]
            record.update(exit_code=code, elapsed_seconds=record.get("elapsed_seconds", 0) + time.time() - task_start)
            record["status"] = ("interrupted" if signalled[0] else "paused") if code == PAUSE else (
                "completed" if code == 0 else "failed")
            current["workers"][str(worker_id)]["task"] = None
            if code == 0:
                current["workers"][str(worker_id)]["recovery_failures"] = 0
                current["workers"][str(worker_id)]["bounded_failures"] = 0
        event(campaign, task=task["id"], event="task-end", exit_code=code)
        summarize(campaign)
        if code == PAUSE:
            break
    with state_transaction(campaign) as current:
        current["workers"][str(worker_id)].update(
            status="interrupted" if signalled[0] else "drained", ended_at=time.time())
    return PAUSE if signalled[0] else 0


def seconds(value):
    parts = str(value).split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected HH:MM:SS, got {value}")
    h, m, s = map(int, parts)
    return h * 3600 + m * 60 + s


def active_jobs(payload, user):
    return {identifier: job for identifier, job in (payload.get("Jobs") or {}).items()
            if job.get("Job_Owner", "").split("@")[0] == user
            and job.get("job_state") in {"Q", "R", "H", "T", "W", "S", "E", "B"}}


def allocated_nodes(job):
    selection = job.get("Resource_List", {}).get("select", "1")
    try:
        return sum(int(chunk.split(":")[0]) for chunk in str(selection).split("+"))
    except ValueError:
        # Unknown resource requests never create artificial free slots.
        return MAX_NODES


def qstat(*args):
    result = subprocess.run(["qstat", *args, "-f", "-F", "json"], text=True,
                            capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f"qstat failed; refusing to infer free slots: {result.stderr.strip()}")
    return json.loads(result.stdout)


def classify_end(job):
    """Do not infer walltime from a launcher SIGTERM/exit 143 alone."""
    text = " ".join(str(job.get(k, "")) for k in ("comment", "reason", "Exit_reason", "description")).lower()
    if re.search(r"qdel|delete request|user request|cancelled by|canceled by", text):
        return "cancelled"
    if re.search(r"permission denied|unauthori[sz]ed|not authorized|access denied|"
                 r"policy violation|quota violation|reservation (?:expired|ended)|queue.*(?:disabled|not enabled)", text):
        return "policy_denied"
    admin = bool(re.search(r"(?:terminated|killed|deleted) by root(?:@|\b)", text))
    if not admin and re.search(r"(?:deleted|terminated|killed) by (?:user\b|[a-z0-9_-]+@)", text):
        return "cancelled"
    if re.search(r"walltime|wall time|time limit|node fail|node down|mom.*lost|preempt", text):
        return "recoverable"
    if re.search(r"placement set is too small|insufficient.*node_group", text):
        return "placement_failure"
    if admin:
        return "admin_terminated"
    try:
        code = int(job.get("Exit_status"))
    except (TypeError, ValueError):
        code = -99999
    if code == 0:
        return "completed"
    try:
        used = seconds(job.get("resources_used", {}).get("walltime", "0:0:0"))
        requested = seconds(job.get("Resource_List", {}).get("walltime", "0:0:0"))
        if requested > 0 and used >= requested and code in (271, 143, 137, -11):
            return "recoverable"
    except (ValueError, TypeError):
        pass
    return "unknown_failure"


def recovery_outcome(job, plan):
    """Apply this campaign's approved interruption policy to PBS evidence."""
    reason = classify_end(job)
    policy = plan.get("recovery_policy", {})
    if reason == "admin_terminated" and policy.get("retry_admin_termination"):
        return "recoverable", reason
    text = str(job.get("comment", "")).lower()
    if (reason in {"placement_failure", "unknown_failure"} and policy.get("retry_terminated_queued")
            and job.get("job_state") in {"F", "C"} and "terminated" in text
            and not job.get("stime") and not job.get("exec_host")
            and job.get("Exit_status") is None):
        # PBS can retain its last placement comment when a queued job is
        # terminated. Retry the same reservation/resources at the approved interval;
        # never change node_group or pretend this proves a permanent mismatch.
        return "recoverable", "terminated_while_queued"
    return reason, reason


def retry_limit(plan, reason):
    policy = plan.get("recovery_policy", {})
    if reason in UNLIMITED_INTERRUPTION_REASONS and "interruption_max_retries" in policy:
        return policy["interruption_max_retries"]
    return int(policy.get("other_max_retries", policy.get("max_retries", plan.get("max_attempts", 3))))


def all_workers(state):
    """Keep retired allocation evidence without making a fourth submission slot."""
    return [*state["workers"].values(), *state.get("retired_workers", {}).values()]


def pbs_exit_code(job):
    try:
        return int(job.get("Exit_status"))
    except (TypeError, ValueError):
        return None


def render_pbs(plan, worker_id, job_name):
    if not re.fullmatch(r"0390_[A-Za-z0-9_-]{1,10}", job_name):
        raise ValueError("PBS job names must use the established 0390_ prefix and fit 15 characters")
    command = ["bash", plan.get("controller_shell", plan["shell"]),
               plan["project_root"], plan["campaign"], str(worker_id)]
    return ("#!/bin/bash\n#PBS -P gcg51557\n#PBS -q R9920261000\n#PBS -v RTYPE=rt_HF\n"
            f"#PBS -l select=1\n#PBS -l walltime={plan['walltime']}\n#PBS -N {job_name}\n"
            "#PBS -j oe\n#PBS -k oe\nset -euo pipefail\n" + shlex.join(command) + "\n")


def archive_job(campaign, plan, job_id, pbs=None):
    archive = Path(plan["project_root"]) / "logs/0390/runs" / job_id
    archive.mkdir(parents=True, exist_ok=True)
    if pbs is not None:
        write(archive / "pbs-final.json", pbs)
    state = read(campaign / "state.json")
    ids = [name for name, task in state["tasks"].items() if job_id in task.get("job_ids", [])]
    write(archive / "job.json", {"job_id": job_id, "stage": "conrep-night-worker",
          "queue": plan["queue"], "rtype": plan["rtype"], "nproc": 8,
          "campaign": str(campaign), "tasks": ids, "source_hash": plan["source_hash"],
          "updated_at": time.time(), "pbs": pbs,
          "status": "completed" if pbs and pbs_exit_code(pbs) == 0 else "recorded"})
    write(archive / "artifacts" / "campaign-state.json", state)
    shutil.copy2(campaign / "plan.json", archive / "artifacts" / "plan.json")
    for task in plan["tasks"]:
        if task["id"] not in ids:
            continue
        target = archive / "artifacts" / task["id"]
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(task["config"], target / "config.json")
        for relative in ("training/train.jsonl", "training/TRAINING_COMPLETE.json"):
            source = Path(task["output"]) / relative
            if source.is_file():
                shutil.copy2(source, target / source.name)
        for source in (Path(task["output"]) / "validation").glob("checkpoint-*/metrics.json"):
            dest = target / "validation" / source.parent.name
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest / "metrics.json")
    index = Path(plan["project_root"]) / "logs/0390/INDEX.md"
    with locked(index.with_suffix(".lock")):
        existing = index.read_text() if index.exists() else "# 0390 jobs\n"
        if f"runs/{job_id}/" not in existing:
            with index.open("a") as stream:
                if not existing.strip():
                    stream.write("# 0390 jobs\n")
                stream.write(f"\n- [{job_id}](runs/{job_id}/job.json): ConRep night, {campaign.name}\n")


def reconcile(campaign, plan, active, *, reassess=False):
    snapshot = read(campaign / "state.json")
    entries = [(group, key, value) for group in ("workers", "retired_workers")
               for key, value in snapshot.get(group, {}).items()]
    for group, worker_id, old in entries:
        job_id = old.get("job_id")
        if not job_id:
            if old["status"] == "submitting":
                matches = [identifier for identifier, job in active.items()
                           if job.get("Job_Name") == old["job_name"]]
                if len(matches) == 1:
                    with state_transaction(campaign) as state:
                        state[group][worker_id].update(job_id=matches[0], status="queued")
                    event(campaign, event="adopt-submission", worker=worker_id, job_id=matches[0])
                # An ambiguous submission stays blocked rather than duplicating qsub.
            continue
        if job_id in active:
            continue
        already = old.get("reconciled_job") == job_id
        changed_policy = old.get("reconciled_policy") != RECOVERY_VERSION
        if already and not (reassess and (old.get("last_outcome") in {
                "unknown_failure", "admin_terminated", "placement_failure", "admin_stopped", "resource_rejected"}
                or (changed_policy and old.get("last_outcome") == "recoverable"))):
            continue
        try:
            payload = qstat("-x", job_id)
            records = payload.get("Jobs") or {}
            record = records.get(job_id)
            if record is None:
                matches = [value for name, value in records.items()
                           if name.split(".")[0] == job_id.split(".")[0]]
                record = matches[0] if len(matches) == 1 else None
            if not record:
                raise RuntimeError("Finished PBS record missing")
            if record.get("job_state") not in {"F", "C"}:
                continue
        except Exception as exc:
            cached = Path(plan["project_root"]) / "logs/0390/runs" / job_id / "pbs-final.json"
            if reassess and cached.is_file():
                record = read(cached)
                if record.get("job_state") not in {"F", "C"}:
                    raise ValueError(f"Archived job has no final state: {job_id}")
            else:
                event(campaign, event="pbs-status-unknown", job_id=job_id, error=str(exc))
                continue
        outcome, reason = recovery_outcome(record, plan)
        with state_transaction(campaign) as state:
            worker = state[group][worker_id]
            # status and supervisor may reconcile concurrently. Count each
            # finished allocation once, including migration from the old policy.
            counted = worker.get("reconciled_job") == job_id and worker.get("last_outcome") == "recoverable"
            if worker.get("reconciled_job") == job_id and not reassess:
                continue
            if worker.get("planned_stop"):
                outcome = "planned_stop"
            elif outcome == "unknown_failure" and worker.get("status") == "drained" and pbs_exit_code(record) == 0:
                outcome = "completed"
            worker.update(reconciled_job=job_id, last_outcome=outcome,
                          last_reason=reason, reconciled_policy=RECOVERY_VERSION,
                          pbs_exit_status=record.get("Exit_status"),
                          pbs_comment=record.get("comment"),
                          pbs_started_at=record.get("stime"))
            limit = retry_limit(plan, reason)
            if outcome == "recoverable":
                if not counted:
                    worker["recovery_failures"] = worker.get("recovery_failures", 0) + 1
                    if limit is not None:
                        worker["bounded_failures"] = worker.get("bounded_failures", 0) + 1
                policy = plan.get("recovery_policy", {})
                delay = (policy.get("interruption_retry_seconds", 1200) if limit is None else
                         min(policy.get("backoff_max_seconds", 300), policy.get("backoff_seconds", 60)
                             * 2 ** min(max(worker.get("bounded_failures", 1) - 1, 0), 10)))
                try:
                    ended = float(record.get("history_timestamp"))
                except (TypeError, ValueError):
                    ended = worker.get("interruption_ended_at", time.time()) if counted else time.time()
                worker["interruption_ended_at"] = ended
                worker["retry_after"] = max(ended, 0) + delay
            worker["status"] = ("available" if outcome in {"completed", "recoverable", "planned_stop"}
                                and (outcome != "recoverable" or limit is None
                                     or worker.get("bounded_failures", 0) <= limit) else
                                "retry_exhausted" if outcome == "recoverable" else outcome)
            for name, task in state["tasks"].items():
                legacy_failure = reassess and task.get("failure_reason") in {
                    "unknown_failure", "admin_terminated", "placement_failure", "admin_stopped", "resource_rejected"}
                old_retry_exhausted = (reassess and counted and limit is None
                    and task.get("reconciled_job") == job_id and task.get("failures", 0) > 0)
                # A killed child can exit before the worker receives SIGTERM.
                # These exits become retryable only with independent final PBS
                # evidence; ordinary training errors remain failed.
                killed_child = outcome == "recoverable" and task.get("exit_code") in {
                    -signal.SIGTERM, -signal.SIGKILL, 128 + signal.SIGTERM, 128 + signal.SIGKILL}
                if task.get("job_id") != job_id or not (
                        task["status"] in {"running", "interrupted", "paused"}
                        or (task["status"] == "failed" and (legacy_failure or killed_child or old_retry_exhausted))):
                    continue
                task["reconciled_job"] = job_id
                if outcome == "recoverable":
                    if not counted:
                        task["failures"] = task.get("failures", 0) + 1
                        if limit is not None:
                            task["bounded_failures"] = task.get("bounded_failures", 0) + 1
                    task["retry_after"] = worker["retry_after"] if limit is None else 0
                    task["status"] = "paused" if limit is None or task.get("bounded_failures", 0) <= limit else "failed"
                    if task["status"] == "failed":
                        task["failure_reason"] = "retry_exhausted"
                    else:
                        task.pop("failure_reason", None)
                elif outcome in {"completed", "planned_stop"}:
                    task["status"] = "paused"
                else:
                    task["status"] = "cancelled" if outcome == "cancelled" else "failed"
                    task["failure_reason"] = outcome
        event(campaign, event="pbs-finished", job_id=job_id, outcome=outcome, reason=reason,
              exit_status=record.get("Exit_status"), comment=record.get("comment"))
        archive_job(campaign, plan, job_id, record)


def admissible_pending(campaign, plan, state, *, count=False):
    now = time.time()
    remaining = state["deadline"] - now
    elapsed = [x["elapsed_seconds"] for x in state["tasks"].values()
               if x["status"] == "completed" and x.get("elapsed_seconds", 0) > 0]
    needed = 1.15 * max(2700, max(elapsed[-8:], default=plan["initial_task_seconds"])) + 300
    eligible = 0
    for spec in plan["tasks"]:
        task = state["tasks"][spec["id"]]
        if not ready_to_claim(task):
            continue
        trained = training_done(spec, read(spec["config"]))
        allowed = (trained and remaining > 1500) or (
            not trained and remaining > needed + 300 and (spec["priority"] < 2 or remaining > 5400))
        if allowed:
            eligible += 1
            if not count:
                return True
    return eligible if count else False


def submit_available(campaign, plan):
    check_worker_limit(plan)
    # Same lock path as the existing single-job submitter. All current user's
    # active jobs count, including other projects and held/queued jobs.
    with locked(Path(plan["project_root"]) / "logs/0390/jobs/submit.lock"):
        jobs = active_jobs(qstat(), getpass.getuser())
        state = read(campaign / "state.json")
        if any(w["status"] == "submitting" and not w.get("job_id")
               for w in all_workers(state)):
            # An accepted submission can briefly be absent from qstat. Until
            # reconcile adopts it, its account-slot usage is unknown.
            return
        # A successful qsub can be invisible to qstat for a short time. A known
        # ID holds its slot until PBS history confirms that allocation ended.
        unseen = {w["job_id"] for w in all_workers(state)
                  if w.get("job_id") and w["job_id"] not in jobs
                  and w.get("reconciled_job") != w["job_id"]}
        free = min(MAX_NODES - len(jobs) - len(unseen),
                   MAX_NODES - sum(allocated_nodes(job) for job in jobs.values()) - len(unseen))
        active_owned = sum(w.get("job_id") in jobs or w.get("job_id") in unseen
                           for w in all_workers(state))
        already_claimed = sum(t["status"] == "running" for t in state["tasks"].values())
        free = min(free, admissible_pending(campaign, plan, state, count=True)
                   + already_claimed - active_owned)
        if free <= 0:
            return
        for worker_id in range(plan["workers"]):
            state = read(campaign / "state.json")
            if state["status"] != "running" or not admissible_pending(campaign, plan, state):
                return
            old = state["workers"][str(worker_id)]
            if old.get("job_id") in jobs or old["status"] not in {"new", "available"}:
                continue
            if time.time() < old.get("retry_after", 0):
                continue
            allocation = old["allocations"] + 1
            job_name = f"0390_n{sha(str(campaign))[:3]}{worker_id}{allocation:03d}"
            script = render_pbs(plan, worker_id, job_name)
            path = campaign / "jobs" / f"{job_name}.pbs"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(script)
            with state_transaction(campaign) as state:
                state["workers"][str(worker_id)].update(status="submitting", job_id=None,
                    job_name=job_name, allocations=allocation, submission_script=str(path),
                    planned_stop=False)
            result = subprocess.run(["qsub", str(path)], text=True, capture_output=True,
                                    cwd=plan["project_root"], timeout=30)
            if result.returncode:
                # A failed/ambiguous qsub is not automatically retried: a response
                # can be lost after PBS accepts a job. Reconcile by unique name.
                event(campaign, event="qsub-ambiguous", worker=worker_id,
                      stdout=result.stdout, stderr=result.stderr)
                return
            job_id = result.stdout.strip()
            if not re.fullmatch(r"\d+(?:\.[A-Za-z0-9_.-]+)?", job_id):
                event(campaign, event="qsub-ambiguous", worker=worker_id, stdout=job_id)
                return
            with state_transaction(campaign) as state:
                worker_state = state["workers"][str(worker_id)]
                # PBS may start the worker before qsub returns. Never overwrite
                # its running/drained state with the older queued observation.
                if worker_state["status"] == "submitting":
                    worker_state.update(job_id=job_id, status="queued")
                elif worker_state.get("job_id") != job_id:
                    raise RuntimeError("PBS submission and worker job IDs disagree")
            archive = Path(plan["project_root"]) / "logs/0390/runs" / job_id
            archive.mkdir(parents=True, exist_ok=True)
            (archive / "submission.pbs").write_text(script)
            write(archive / "submission.json", {"job_id": job_id, "job_name": job_name,
                  "worker": worker_id, "campaign": str(campaign), "submitted_at": time.time(),
                  "source_hash": plan["source_hash"], "queue": plan["queue"], "nproc": 8})
            event(campaign, event="submitted", worker=worker_id, job_id=job_id)
            free -= 1
            if free <= 0:
                break


def summarize(campaign):
    plan = read(campaign / "plan.json")
    rows = []
    for task in plan["tasks"]:
        cfg = read(task["config"])
        for step in expected_checkpoints(cfg):
            if not validation_done(task, step):
                continue
            report = read(Path(task["output"]) / "validation" / f"checkpoint-{step}" / "metrics.json")
            row = {"experiment": task["id"], "model": task["model"], "variant": task["variant"],
                   "seed": task["seed"], "step": step, "protocol_hash": report.get("protocol_hash"),
                   "checkpoint": report.get("checkpoint"), **report.get("metrics", {})}
            for key, value in report.get("mmlu_diagnostics", {}).items():
                if isinstance(value, (int, float, str, bool)):
                    row["mmlu_diagnostic." + key] = value
            rows.append(row)
    with locked(campaign / "summary.lock"):
        path = campaign / "checkpoint-results.csv"
        temporary = path.with_suffix(".csv.tmp")
        keys = sorted({key for row in rows for key in row})
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
        state = read(campaign / "state.json")
        counts = {}
        for task in state["tasks"].values():
            counts[task["status"]] = counts.get(task["status"], 0) + 1
        write(campaign / "summary.json", {"state": state["status"], "task_counts": counts,
              "validated_checkpoints": len(rows), "deadline": state["deadline"],
              "worker_count": plan.get("workers"), "max_account_nodes": MAX_NODES,
              "recovery_policy": plan.get("recovery_policy", {}),
              "workers": {key: {field: worker[field] for field in (
                  "job_id", "status", "task", "stage", "last_outcome", "pbs_exit_status",
                  "pbs_comment", "error", "last_reason", "recovery_failures", "retry_after") if field in worker}
                  for key, worker in state["workers"].items()},
              "updated_at": time.time(), "selection": "disabled", "test": "not run"})


def supervise(args):
    campaign = Path(args.campaign).resolve()
    plan = read(campaign / "plan.json")
    check_worker_limit(plan)
    # Held for the lifetime of this supervisor; start/resume cannot duplicate it.
    with locked(campaign / "supervisor.lock", blocking=False):
        write(campaign / "supervisor.json", {"pid": os.getpid(), "host": os.uname().nodename,
                                             "started_at": time.time()})
        while True:
            state = read(campaign / "state.json")
            if state["status"] != "running" or (campaign / "STOP").exists():
                break
            if time.time() >= state["deadline"]:
                (campaign / "STOP").touch()
                with state_transaction(campaign) as state:
                    state["status"] = "deadline_reached"
                    owned = []
                    for worker in all_workers(state):
                        worker["planned_stop"] = True
                        if worker.get("job_id"):
                            owned.append(worker["job_id"])
                # Only jobs belonging to this campaign are cancelled.
                for job_id in owned:
                    subprocess.run(["qdel", job_id], text=True, capture_output=True)
                event(campaign, event="deadline-reached")
                break
            try:
                active = active_jobs(qstat(), getpass.getuser())
                reconcile(campaign, plan, active)
                current = read(campaign / "state.json")
                owned_active = any(w.get("job_id") in active for w in all_workers(current))
                owned_unsettled = any(w["status"] == "submitting" or (
                    w.get("job_id") and w.get("reconciled_job") != w["job_id"])
                    for w in all_workers(current))
                pending = any(t["status"] in {"pending", "paused", "running", "interrupted"}
                              for t in current["tasks"].values())
                if not pending and not owned_active and not owned_unsettled:
                    with state_transaction(campaign) as state:
                        state["status"] = "finished"
                    break
                if not owned_active and not owned_unsettled and all(w["status"] in BLOCKED_WORKER_STATES
                        for w in current["workers"].values()):
                    with state_transaction(campaign) as state:
                        state["status"] = "blocked"
                    detail = {key: {field: w.get(field) for field in (
                        "job_id", "status", "pbs_exit_status", "pbs_comment")}
                        for key, w in current["workers"].items()}
                    write(campaign / "BLOCKED.json", {"time": time.time(), "workers": detail})
                    event(campaign, event="all-workers-blocked", workers=detail,
                          detail="No eligible workers: inspect BLOCKED.json and retry counts")
                    break
                waiting_retry = any(t["status"] == "paused" and t.get("retry_after", 0) > time.time()
                                    for t in current["tasks"].values())
                if not owned_active and not owned_unsettled and not waiting_retry and not admissible_pending(campaign, plan, current):
                    with state_transaction(campaign) as state:
                        state["status"] = "budget_paused"
                    break
                submit_available(campaign, plan)
                summarize(campaign)
                write(campaign / "heartbeat.json", {"time": time.time(), "pid": os.getpid()})
                if time.time() - state.get("last_progress_at", 0) >= 300:
                    print(json.dumps(read(campaign / "summary.json"), ensure_ascii=False), flush=True)
                    with state_transaction(campaign) as state:
                        state["last_progress_at"] = time.time()
            except Exception as exc:
                # Scheduler query/parse failures must never become duplicate submissions.
                event(campaign, event="supervisor-error", error=repr(exc))
                print(f"Supervisor error (no unsafe retry): {exc}", flush=True)
            time.sleep(min(plan["poll_seconds"], max(0, read(campaign / "state.json")["deadline"] - time.time())))
        summarize(campaign)
        event(campaign, event="supervisor-ended", status=read(campaign / "state.json")["status"])
    return 0


def start(args, *, resume=False):
    campaign = Path(args.campaign).resolve()
    plan = read(campaign / "plan.json")
    verify_snapshot(campaign, plan)
    if resume and plan.get("recovery_policy"):
        reconcile(campaign, plan, active_jobs(qstat(), getpass.getuser()), reassess=True)
    state = read(campaign / "state.json")
    if state["workers"] and all(w["status"] in BLOCKED_WORKER_STATES
                                 for w in state["workers"].values()):
        reasons = {key: {field: w.get(field) for field in (
            "job_id", "status", "pbs_comment")} for key, w in state["workers"].items()}
        raise ValueError("All workers are blocked; no jobs submitted or deadline changed. "
                         "Resolve the scheduler/administrator failure before retrying: "
                         + json.dumps(reasons, ensure_ascii=False))
    if resume and (not args.hours or args.hours <= 0):
        raise ValueError("Explicit resume requires positive --hours")
    with locked(campaign / "start.lock"):
        # Serialize the liveness check with launch; two simultaneous starts
        # must not both decide that a supervisor is absent.
        try:
            with locked(campaign / "supervisor.lock", blocking=False):
                pass
        except BlockingIOError:
            if resume:
                with state_transaction(campaign) as state:
                    state["deadline"] = time.time() + args.hours * 3600
                    state["status"] = "running"
                (campaign / "STOP").unlink(missing_ok=True)
                print("Extended the existing supervisor; current PBS jobs remain adopted.")
                return 0
            print("Supervisor is already running; existing jobs are adopted, no duplicate start.")
            return 0
        # A login-node scheduler failure does not consume the ten-hour budget.
        qstat()
        with state_transaction(campaign) as state:
            if state["started_at"] is None:
                state["started_at"] = time.time()
                state["deadline"] = state["started_at"] + plan["hours"] * 3600
            elif state["status"] != "running" and not resume:
                raise ValueError("Campaign was stopped/finished. Use resume --hours to explicitly extend it.")
            if resume:
                state["deadline"] = time.time() + args.hours * 3600
                for worker in state["workers"].values():
                    if worker["status"] in {"drained", "available"}:
                        worker["status"] = "available"
            state["status"] = "running"
        if resume:
            (campaign / "STOP").unlink(missing_ok=True)
        with (campaign / "supervisor.log").open("a") as stream:
            process = subprocess.Popen([plan["python"], plan.get("controller_entry", plan["entry"]),
                                        "supervise", "--campaign", str(campaign)],
                                       cwd=plan["project_root"], stdout=stream, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        time.sleep(0.5)
        if process.poll() is not None:
            raise RuntimeError("Supervisor failed to start; inspect supervisor.log")
        print(f"Supervisor PID {process.pid}; campaign {campaign}; at most {MAX_NODES} active account jobs/nodes.")
    return 0


def recover(args):
    """Reassess old scheduler failures and start without resetting experiments."""
    campaign = Path(args.campaign).resolve()
    plan = read(campaign / "plan.json")
    verify_snapshot(campaign, plan)
    hours = getattr(args, "hours", None)
    if hours is not None and hours <= 0:
        raise ValueError("--hours must be positive")
    try:
        with locked(campaign / "supervisor.lock", blocking=False):
            pass
    except BlockingIOError:
        return start(args, resume=hours is not None)
    with locked(campaign / "start.lock"), locked(campaign / "supervisor.lock", blocking=False):
        before = read(campaign / "state.json")
        if before["status"] == "stopped" or ((campaign / "STOP").exists()
                and before["status"] != "deadline_reached"):
            raise ValueError("Explicitly stopped campaign preserved; use resume for an intentional restart")
        if hours is None and before.get("deadline") is not None and time.time() >= before["deadline"]:
            raise ValueError("Original deadline passed; recover --hours HOURS explicitly grants more time")
        jobs = active_jobs(qstat(), getpass.getuser())
        reconcile(campaign, plan, jobs, reassess=True)
        with state_transaction(campaign) as state:
            eligible = any(w["status"] in {"new", "available"} or w.get("job_id") in jobs
                           for w in state["workers"].values())
            if not eligible:
                raise ValueError("No recoverable worker; inspect final PBS reasons and retry counts")
            if hours is not None:
                state["deadline"] = time.time() + hours * 3600
            state["status"] = "running"
        if before["status"] == "deadline_reached" and hours is not None:
            (campaign / "STOP").unlink(missing_ok=True)
        event(campaign, event="recovery-enabled", policy=plan.get("recovery_policy"),
              deadline=read(campaign / "state.json")["deadline"],
              detail="Original task configs, trainer snapshot and checkpoints retained")
    return start(args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--project-root", default=str(Path.cwd()))
    p.add_argument("--campaign", required=True)
    p.add_argument("--gemma-config", default=BASE_CAMPAIGN + "/gemma2_9b/config.json")
    p.add_argument("--llama-config", default=BASE_CAMPAIGN + "/llama8b/config.json")
    p.add_argument("--hours", type=float, default=10)
    p.add_argument("--reserve-gb", type=float, default=100)
    for name in ("start", "resume", "recover", "supervise", "worker", "status", "stop", "summarize"):
        p = commands.add_parser(name)
        p.add_argument("--campaign", required=True)
        if name == "resume":
            p.add_argument("--hours", type=float, required=True)
        if name == "recover":
            p.add_argument("--hours", type=float, help="Explicitly extend the budget; default keeps the original deadline")
        if name == "worker":
            p.add_argument("--worker", type=int, choices=range(MAX_NODES), required=True)
        if name == "stop":
            p.add_argument("--cancel-jobs", action="store_true")
    p = commands.add_parser("train")
    p.add_argument("--config", required=True)
    p.add_argument("--resume")
    p.add_argument("--deadline", type=float)
    p.add_argument("--stop-file")
    p.add_argument("--stop-after-step", type=int)
    p = commands.add_parser("validate-one")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if args.hours <= 0 or args.reserve_gb < 0:
            parser.error("hours must be positive and reserve-gb nonnegative")
        prepare(args)
    elif args.command == "train":
        from .trainer import run
        result = run(read(args.config), args.resume, deadline=args.deadline, stop_file=args.stop_file,
                     stop_after_step=args.stop_after_step)
        # torchrun converts any nonzero rank exit into ChildFailedError. A
        # deliberate pause is communicated by the atomic PAUSED.json marker.
        return 0 if result == PAUSE else result
    elif args.command == "validate-one":
        validate_one(args.config, args.checkpoint, args.output)
    elif args.command in {"start", "resume"}:
        return start(args, resume=args.command == "resume")
    elif args.command == "recover":
        return recover(args)
    elif args.command == "supervise":
        return supervise(args)
    elif args.command == "worker":
        return worker(args)
    elif args.command == "stop":
        campaign = Path(args.campaign).resolve()
        (campaign / "STOP").touch()
        with state_transaction(campaign) as state:
            state["status"] = "stopped"
            jobs = []
            for worker_state in all_workers(state):
                worker_state["planned_stop"] = True
                if worker_state.get("job_id"):
                    jobs.append(worker_state["job_id"])
        if args.cancel_jobs:
            for job_id in jobs:
                subprocess.run(["qdel", job_id], text=True, capture_output=True)
        print("Stopped. No automatic resubmission; resume requires an explicit command.")
    else:
        campaign = Path(args.campaign).resolve()
        if args.command == "status" and shutil.which("qstat"):
            try:
                reconcile(campaign, read(campaign / "plan.json"), active_jobs(qstat(), getpass.getuser()))
            except Exception as exc:
                print(f"PBS status unavailable; displaying persisted state: {exc}", file=sys.stderr)
        summarize(campaign)
        print(json.dumps(read(campaign / "summary.json"), ensure_ascii=False, indent=2))
    return 0
