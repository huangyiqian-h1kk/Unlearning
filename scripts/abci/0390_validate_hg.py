#!/usr/bin/env python3
"""Render/submit single-GPU SFT validation, optionally replacing its queued HF jobs."""

import argparse
import getpass
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ACTIVE = {"Q", "R", "H", "T", "W", "S", "E", "B"}
PRESETS = {
    "llama3b": {"old_run": "l3sftv3", "run": "l3sftv3hg", "walltime": "02:00:00"},
    "qwen7b": {"old_run": "q7sftv3", "run": "q7sftv3hg", "walltime": "03:00:00"},
}


def submit_command(root, model):
    preset = PRESETS[model]
    results = f"results/validated_v2/0390/{model}"
    return [
        sys.executable, str(root / "scripts/abci/0390_submit.py"), "validate-series",
        "--model", model, "--run-id", preset["run"],
        "--rtype", "rt_HG", "--nproc", "1", "--walltime", preset["walltime"],
        "--include-backbone", "--checkpoint-root", f"{results}/sft",
        "--output", f"{results}/sft-validation-v3-hg",
        "--set", "evaluation.batch_size=16",
    ]


def marker(root, model, run_id):
    return root / "logs/0390/jobs" / f"0390_validate-series_{model}_{run_id}.jobid"


def scheduler():
    result = subprocess.run(["qstat", "-f", "-F", "json"], check=True,
                            capture_output=True, text=True)
    return json.loads(result.stdout).get("Jobs") or {}


def owned(job):
    return job.get("Job_Owner", "").split("@")[0] == getpass.getuser()


def pending_replacements(root, models, jobs, replace_queued):
    cancel = []
    for model in models:
        preset = PRESETS[model]
        for kind, run_id in (("HG", preset["run"]), ("HF", preset["old_run"])):
            path = marker(root, model, run_id)
            if not path.is_file():
                continue
            job_id = path.read_text().strip()
            job = jobs.get(job_id, {})
            if job.get("job_state") not in ACTIVE:
                continue
            if not owned(job) or job.get("Job_Name") != ("0390_" + run_id)[:15]:
                raise RuntimeError(f"Job identity mismatch for {path}: {job_id}; no cancellation")
            if kind == "HG":
                raise RuntimeError(f"HG job {job_id} is already active; do not submit it again")
            if job["job_state"] != "Q":
                raise RuntimeError(f"HF job {job_id} is {job['job_state']}, not queued; leaving it untouched")
            if not replace_queued:
                raise RuntimeError(f"HF job {job_id} is queued; use --replace-queued-hf to replace it")
            cancel.append(job_id)
    remaining = sum(owned(job) and job.get("job_state") in ACTIVE
                    for job_id, job in jobs.items() if job_id not in cancel)
    if remaining + len(models) > 2:
        raise RuntimeError(f"Two-job limit: {remaining} other jobs would remain active; "
                           f"cannot submit {len(models)} HG jobs. No cancellation performed.")
    return cancel


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["both", *PRESETS], default="both")
    p.add_argument("--submit", action="store_true", help="Without this flag, only render PBS files")
    p.add_argument("--replace-queued-hf", action="store_true",
                   help="Cancel only the matching old HF jobs observed in Q state before submitting")
    args = p.parse_args(argv)
    if args.replace_queued_hf and not args.submit:
        p.error("--replace-queued-hf requires --submit")
    models = list(PRESETS) if args.model == "both" else [args.model]
    commands = [submit_command(ROOT, model) for model in models]
    # Render both scripts first: catch conflicting run labels before changing queued jobs.
    for model, command in zip(models, commands):
        print(f"[0390] {model}: HG / 1 GPU / batch 16 / {PRESETS[model]['walltime']}", flush=True)
        subprocess.run([*command, "--dry-run"], cwd=ROOT, check=True)
    if not args.submit:
        print("[0390] Render only: no qstat, qdel, or qsub was executed.")
        return 0
    for model in models:
        sft = ROOT / f"results/validated_v2/0390/{model}/sft"
        for required in (sft / "TRAINING_COMPLETE.json", sft / "final/config.json"):
            if not required.is_file():
                raise FileNotFoundError(f"SFT input missing: {required}; no cancellation performed")
    cancel = pending_replacements(ROOT, models, scheduler(), args.replace_queued_hf)
    for job_id in cancel:
        # Recheck just before qdel; a job may have started while the plan was rendered.
        current = scheduler().get(job_id)
        if current is None or current.get("job_state") not in ACTIVE:
            continue
        if current["job_state"] != "Q" or not owned(current):
            raise RuntimeError(f"Job {job_id} changed state/owner; stop without cancelling it")
        print(f"[0390] Cancelling queued HF job {job_id}", flush=True)
        subprocess.run(["qdel", job_id], check=True)
    if cancel:
        for _ in range(20):
            jobs = scheduler()
            if not any(jobs.get(job_id, {}).get("job_state") in ACTIVE for job_id in cancel):
                break
            time.sleep(1)
        else:
            raise RuntimeError("PBS has not released the cancelled jobs yet; retry once they finish exiting")
        subprocess.run([sys.executable, str(ROOT / "scripts/abci/0390_logs.py"),
                        "collect", *cancel], cwd=ROOT, check=True)
    # The common submitter rechecks the user's two-job limit, and records each new job ID.
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
