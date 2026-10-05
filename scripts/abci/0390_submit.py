#!/usr/bin/env python3
"""Render a standalone PBS file and submit at most two active jobs per user."""

import argparse
import fcntl
import getpass
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from experiments.joblogs import record_submission, submission_snapshot

PROFILES = {
    "preflight": ("00:20:00", 8),
    "smoke": ("00:20:00", 8),
    "sft": ("06:00:00", 8),
    "sft-pipeline": ("03:00:00", 8),
    "unlearn": ("02:00:00", 8),
    "baseline": ("03:00:00", 8),
    "validate": ("02:00:00", 1),
    "validate-series": ("04:00:00", 8),
    "audit-mmlu": ("00:30:00", 1),
    "analyze": ("01:00:00", 1),
    "falcon-layers": ("02:00:00", 1),
    "relearn-augment": ("02:00:00", 1),
}

SINGLE_GPU_STAGES = {"validate", "audit-mmlu", "analyze", "falcon-layers", "relearn-augment"}


def render(stage, model, run_id, root, walltime, nproc, extra, rtype="rt_HF"):
    if rtype not in {"rt_HF", "rt_HG"}:
        raise ValueError("rtype must be rt_HF or rt_HG")
    if rtype == "rt_HG" and nproc != 1:
        raise ValueError("rt_HG allocates one GPU; use --nproc 1")
    if stage in SINGLE_GPU_STAGES and nproc != 1:
        raise ValueError(f"{stage} has no multi-rank output sharding; use --nproc 1")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run-id may contain only letters, digits, '_' and '-'")
    if not re.fullmatch(r"\d{1,3}:[0-5]\d:[0-5]\d", walltime):
        raise ValueError("walltime must be HH:MM:SS")
    name = ("0390_" + run_id)[:15]
    argv = [
        "bash",
        str(root / "scripts/abci/0390_run.sh"),
        "--nproc",
        str(nproc),
        "--",
        stage,
        "--config",
        f"configs/0390/{model}.yaml",
        *extra,
    ]
    if stage == "preflight":
        argv.append("--distributed")
    return (
        f"#!/bin/bash\n#PBS -P gcg51557\n#PBS -q R9920261000\n#PBS -v RTYPE={rtype}\n"
        f"#PBS -l select=1\n#PBS -l walltime={walltime}\n#PBS -N {name}\n#PBS -j oe\n#PBS -k oe\n"
        f"set -euo pipefail\ncd {shlex.quote(str(root))}\n{shlex.join(argv)}\n"
    )


def active_jobs(payload, user):
    return [
        identifier
        for identifier, job in (payload.get("Jobs") or {}).items()
        if job.get("Job_Owner", "").split("@")[0] == user
        and job.get("job_state") in {"Q", "R", "H", "T", "W", "S", "E", "B"}
    ]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=PROFILES)
    p.add_argument("--model", choices=["qwen7b", "llama3b", "llama8b"], required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--walltime")
    p.add_argument(
        "--rtype",
        choices=["rt_HF", "rt_HG"],
        help="Default: rt_HG for single-GPU evaluation/preparation, rt_HF otherwise",
    )
    p.add_argument("--nproc", type=int, choices=[1, 2, 4, 8])
    p.add_argument("--dry-run", action="store_true")
    args, extra = p.parse_known_args(argv)
    root = Path(__file__).resolve().parents[2]
    walltime, nproc = PROFILES[args.stage]
    if args.stage == "sft" and args.model == "llama3b":
        walltime = "03:00:00"
    if args.stage == "unlearn" and args.model == "llama3b":
        walltime = "01:00:00"
    rtype = args.rtype or ("rt_HG" if args.stage in SINGLE_GPU_STAGES else "rt_HF")
    nproc = args.nproc or (1 if rtype == "rt_HG" else nproc)
    script = render(
        args.stage,
        args.model,
        args.run_id,
        root,
        args.walltime or walltime,
        nproc,
        extra,
        rtype=rtype,
    )
    path = root / "logs/0390/jobs" / f"0390_{args.stage}_{args.model}_{args.run_id}.pbs"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != script:
        raise FileExistsError(f"Different job already uses run-id: {path}")
    path.write_text(script)
    if args.dry_run:
        print(script)
        print(f"Written: {path}")
        return
    with (path.parent / "submit.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = subprocess.run(
            ["qstat", "-f", "-F", "json"], check=True, text=True, capture_output=True
        )
        jobs = active_jobs(json.loads(result.stdout), getpass.getuser())
        if len(jobs) >= 2:
            raise RuntimeError(
                f"Two-job limit: already {len(jobs)} active/queued jobs. No job submitted."
            )
        snapshot = submission_snapshot(
            root, args.stage, args.model, args.run_id, rtype, nproc,
            args.walltime or walltime, extra,
        )
        result = subprocess.run(
            ["qsub", str(path)], check=True, text=True, capture_output=True, cwd=root
        )
        job_id = result.stdout.strip()
        (path.with_suffix(".jobid")).write_text(job_id + "\n")
        print(f"Submitted {job_id}: {path}")
        try:
            record_submission(root, job_id, snapshot, script)
            print(f"Job archive: {root / 'logs/0390/runs' / job_id}")
        except Exception as exc:
            # qsub already succeeded. Do not encourage an accidental duplicate submission.
            print(f"WARNING: {job_id} IS SUBMITTED, but archive failed: {exc}. "
                  "Do not resubmit; use 0390_logs.py collect later.", file=sys.stderr)


if __name__ == "__main__":
    main()
