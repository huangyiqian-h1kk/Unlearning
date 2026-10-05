"""One PBS allocation: full SFT, backbone/checkpoint validation, then selection.

The coordinator never loads a model. Separate child processes release all training
state before the allocation is reused for independent validation workers.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from .config import output_dir, write_json
from .selection import select


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def worker_command(stage, config, nproc, extra=()):
    entry = Path(__file__).resolve().parents[2] / "scripts/0390_experiment.py"
    command = [sys.executable]
    if nproc > 1:
        command += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                    f"--nproc_per_node={nproc}"]
    return [*command, str(entry), stage, "--config", str(config), *extra]


def completed_training(cfg, retain_only):
    root = output_dir(cfg)
    marker = json.loads((root / "TRAINING_COMPLETE.json").read_text())
    if Path(marker["final"]).resolve() != root / "final":
        raise ValueError("SFT completion marker points to a different output directory")
    if not (root / "final/config.json").is_file() or marker["global_step"] < 1:
        raise ValueError("SFT final model or optimizer steps missing")
    if json.loads((root / "resolved_config.json").read_text()) != cfg:
        raise ValueError("Completed SFT configuration differs from this pipeline")
    lineage = json.loads((root / "lineage.json").read_text())
    expected = "injection_retain_only" if retain_only else "injection"
    if lineage["training_file"] != expected:
        raise ValueError("Completed SFT used a different injection split")
    return marker


def validate_inputs(cfg, retain_only):
    model = Path(cfg["model"]["name_or_path"])
    if cfg["model"]["local_only"] and not (model / "config.json").is_file():
        raise FileNotFoundError(f"Download the backbone before submission: {model}")
    data = cfg["data"]
    injection = "injection_retain_only" if retain_only else "injection"
    paths = [Path(data["prepared_dir"]) / f"{injection}.jsonl",
             cfg["evaluation"]["mmlu_file"]]
    for split in ("forget", "retain"):
        paths += [data[f"{split}_generation"], *data[f"{split}_mcq"].values()]
    for value in paths:
        path = Path(value)
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Pipeline input missing or empty: {path}")
        with path.open("rb") as stream:
            if stream.read(80).startswith(b"version https://git-lfs.github.com/spec"):
                raise ValueError(f"Materialize the Git LFS input before submission: {path}")


def summarize_validation(jobs, output):
    reports = []
    for checkpoint, dest in jobs:
        path = dest / "metrics.json"
        report = json.loads(path.read_text())
        if Path(report["checkpoint"]).resolve() != checkpoint.resolve():
            raise ValueError(f"Validation checkpoint identity mismatch: {path}")
        reports.append((dest.name, report))
    base = reports[0][1]
    if reports[0][0] != "base":
        raise ValueError("Original-backbone validation is required")
    columns = ["checkpoint", "forget_qa_pct", "retain_qa_pct", "qa_mean_pct",
               "mmlu_pct", "mmlu_change_pp", "mmlu_invalid_pct"]
    with Path(output).open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(columns)
        print("\t".join(columns), flush=True)
        for name, report in reports:
            if report["protocol_hash"] != base["protocol_hash"]:
                raise ValueError("Validation series contains different protocols")
            metrics = report["metrics"]
            values = [metrics["forget.qa"], metrics["retain.qa"],
                      (metrics["forget.qa"] + metrics["retain.qa"]) / 2,
                      metrics["utility.mmlu"],
                      metrics["utility.mmlu"] - base["metrics"]["utility.mmlu"],
                      report["mmlu_diagnostics"]["invalid_fraction"]]
            row = [name, *(f"{100 * value:.2f}" for value in values)]
            writer.writerow(row)
            print("\t".join(row), flush=True)


def run(cfg, output, nproc=8, resume=None, retain_only=False, skip_training=False):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or "RANK" in os.environ:
        raise ValueError("Run sft-pipeline once, not inside torchrun")
    if nproc not in {1, 2, 4, 8}:
        raise ValueError("nproc must be 1, 2, 4 or 8")
    if resume and skip_training:
        raise ValueError("Use either --resume or --skip-training")
    root = output_dir({"run": {"output_dir": output}})
    sft_root, validation = output_dir(cfg), root / "validation"
    if sft_root != root / "sft":
        raise ValueError("Pipeline SFT output must be --output/sft")
    validate_inputs(cfg, retain_only)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "sft-pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = root / "sft-pipeline-config.json"
        if snapshot.exists() and json.loads(snapshot.read_text()) != cfg:
            raise ValueError("Pipeline configuration changed; use a new --output directory")
        if skip_training:
            completed_training(cfg, retain_only)
        elif not resume and sft_root.exists() and any(sft_root.iterdir()):
            raise FileExistsError("SFT output exists: use --resume or --skip-training")
        if not skip_training and (validation.exists() or (root / "selected-sft.json").exists()):
            raise ValueError("Cannot change weights after validation; use a new output directory")
        write_json(snapshot, cfg)
        state = {"job_id": os.environ.get("PBS_JOBID"), "started_at": now(),
                 "status": "running", "nproc": nproc, "retain_only": retain_only,
                 "output_dir": str(root), "stages": [], "selected": None}
        selection_path = root / "selected-sft.json"

        def event(stage, status, **extra):
            record = {"time": now(), "job_id": state["job_id"],
                      "stage": stage, "status": status, **extra}
            state["stages"].append(record)
            write_json(root / "pipeline.json", state)
            with (root / "pipeline-events.jsonl").open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[0390 pipeline] {stage}: {status}", flush=True)

        def execute(stage, extra=()):
            command = worker_command(stage, snapshot, nproc, extra)
            event(stage, "running", command=command)
            print(shlex.join(command), flush=True)
            started = time.monotonic()
            # Inherit stdout/stderr: the outer PBS log supervisor captures all stages.
            subprocess.run(command, check=True)
            event(stage, "completed", elapsed_seconds=round(time.monotonic() - started, 3))

        stage = "sft"
        try:
            if skip_training:
                event(stage, "reused", training_complete=str(sft_root / "TRAINING_COMPLETE.json"))
            else:
                extra = ["--retain-only"] if retain_only else []
                if resume:
                    extra += ["--resume", resume]
                execute(stage, extra)
            marker = completed_training(cfg, retain_only)
            state["global_step"] = marker["global_step"]
            stage = "validate-series"
            execute(stage, ["--checkpoint-root", str(sft_root), "--output", str(validation),
                            "--include-backbone"])
            # Only select after every validation worker exits successfully, and
            # require a report for every discovered checkpoint plus the backbone.
            from .cli import validation_jobs

            jobs = validation_jobs(cfg, sft_root, validation, include_backbone=True)
            stage = "select"
            event(stage, "running", candidates=len(jobs) - 1)
            summarize_validation(jobs, root / "sft-validation-summary.tsv")
            selected = select(
                cfg, [str(dest / "metrics.json") for _, dest in jobs[1:]],
                str(validation / "base/metrics.json"),
                "retain-only" if retain_only else "sft", str(selection_path),
            )
            state.update(status="completed", selected=selected["selected"], ended_at=now())
            event(stage, "completed", selected=selected["selected"])
            return state
        except Exception as exc:
            state.update(status="failed", failed_stage=stage, ended_at=now(), error=str(exc))
            if stage == "select" and selection_path.is_file():
                selection = json.loads(selection_path.read_text())
                if selection.get("selected") is None:
                    state["status"] = "no_eligible_checkpoint"
            event(stage, "failed", error=str(exc),
                  exit_code=exc.returncode if isinstance(exc, subprocess.CalledProcessError) else 1)
            raise
