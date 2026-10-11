#!/usr/bin/env python3
"""Check the Qwen3 training stack or repair only its unpinned Arrow dependency.

For an existing campaign, run with its own Python and --campaign PATH --resume.
The frozen code, plan, input hashes, weights and evaluation caches are not edited.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import uuid

ARROW_VERSION = "19.0.1"


def installed():
    return {re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower(): d.version
            for d in importlib.metadata.distributions() if d.metadata.get("Name")}


def check_stack(smoke=False):
    expected = {"numpy": "1.26.4", "pyarrow": ARROW_VERSION, "datasets": "3.2.0",
                "pandas": "2.2.3", "transformers": "4.51.3", "peft": "0.14.0",
                "accelerate": "1.3.0", "deepspeed": "0.16.2"}
    found = {name: importlib.metadata.version(name) for name in expected}
    if found != expected:
        raise ValueError(f"Qwen3 training dependency mismatch: expected={expected}; found={found}")
    if importlib.metadata.version("torch").split("+")[0] != "2.5.1":
        raise ValueError("Expected torch 2.5.1")
    import numpy as np
    import pyarrow as pa
    import pandas as pd
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments, Qwen3Config, Qwen3ForCausalLM, set_seed
    import peft
    import deepspeed
    import torch

    values = np.array([1, 2, 3], dtype=np.int64)
    table = pa.table({"value": values})
    if table.to_pandas()["value"].tolist() != [1, 2, 3]:
        raise RuntimeError("Arrow/NumPy/Pandas roundtrip failed")
    if Dataset.from_dict({"value": values}).with_format("numpy")[:]["value"].tolist() != [1, 2, 3]:
        raise RuntimeError("Datasets/Arrow roundtrip failed")
    if smoke:
        # Tiny randomly initialized CPU model; no pretrained assets or GPUs.
        torch.set_num_threads(1)
        set_seed(42)
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, use_sliding_window=False, pad_token_id=0, eos_token_id=2)
        model = Qwen3ForCausalLM(config)
        original = {k: v.detach().clone() for k, v in model.named_parameters()}
        data = [dict(input_ids=[1, 4, 5, 2], attention_mask=[1]*4,
                     labels=[-100, -100, 5, 2]) for _ in range(2)]
        with tempfile.TemporaryDirectory(prefix="0390-qwen3-smoke-") as directory:
            args = TrainingArguments(output_dir=directory, max_steps=1,
                per_device_train_batch_size=2, learning_rate=1e-4, use_cpu=True,
                save_strategy="no", report_to=[], disable_tqdm=True,
                dataloader_pin_memory=False)
            trainer = Trainer(model=model, args=args, train_dataset=data)
            result = trainer.train()
            if trainer.state.global_step != 1 or not np.isfinite(result.training_loss):
                raise RuntimeError("Tiny Qwen3 Trainer smoke did not complete one finite step")
            if not any(not torch.equal(original[k], v) for k, v in model.named_parameters()):
                raise RuntimeError("Tiny Qwen3 Trainer did not update any parameter")
    print("QWEN3_TRAINING_PREFLIGHT_PASS " + json.dumps(dict(versions=found, cpu_smoke=smoke)), flush=True)


def require_finished(state, scheduler):
    """Only a confirmed final PBS record permits changing a campaign environment."""
    if state.get("status") == "submitting":
        raise ValueError("Resolve the ambiguous qsub outcome before repairing this environment")
    job = state.get("job_id")
    if not job:
        if state.get("status") not in {"ready", "blocked", "waiting_retry"}:
            raise ValueError("Cannot establish that this campaign is idle")
        return None
    active = scheduler.active_jobs(scheduler.qstat(), getpass.getuser())
    if job in active:
        raise ValueError(f"PBS job {job} is still active; wait until it has finished, then rerun")
    record = (scheduler.qstat("-x", job).get("Jobs") or {}).get(job)
    if not record or record.get("job_state") not in {"F", "C"}:
        raise ValueError(f"Wait for confirmed final PBS history for {job}, then rerun")
    return record


def repair(campaign, resume=False):
    campaign = Path(campaign).resolve()
    plan = json.loads((campaign / "plan.json").read_text())
    if plan.get("schema") != "0390-qwen3-rich-v1" or Path(plan["campaign"]).resolve() != campaign:
        raise ValueError("Expected an existing Qwen3 rich SFT campaign")
    env = Path(plan["python"]).parent.parent
    if Path(sys.prefix).resolve() != env.resolve() or not (env / "0390-qwen3-env").is_file():
        raise ValueError(f"Run this repair with the campaign Python: {plan['python']}")
    # Import the campaign's own frozen verifier and lifecycle implementation.
    sys.path.insert(0, str(campaign / "code/src"))
    from experiments import qwen3_rich as q
    from conrep.night import campaign as scheduler

    receipt = None
    with q.locked(campaign / "start.lock", blocking=False), \
         q.locked(campaign / "supervisor.lock", blocking=False), \
         q.locked(campaign / "worker.lock", blocking=False):
        q.verify(campaign)
        if (campaign / "COMPLETE.json").exists():
            print("Campaign is complete; no repair or resubmission performed", flush=True)
            return
        if (campaign / "STOP").exists():
            raise ValueError("Explicit STOP preserved; resolve it separately before this repair")
        state = q.read(campaign / "state.json")
        final = require_finished(state, scheduler)
        before = installed()
        if before.get("numpy") != "1.26.4":
            raise ValueError("This repair expects the existing NumPy 1.26.4 pin")
        recorded_arrow = plan["environment"].get("pyarrow")
        if recorded_arrow not in {None, ARROW_VERSION}:
            raise ValueError("This plan records another Arrow version; explicit migration is required")
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
        audit = campaign / "maintenance" / ("pyarrow-" + stamp)
        audit.mkdir(parents=True)
        protected = [campaign / n for n in ("plan.json", "config.json", "code-files.json")]
        base = campaign / "validation/base/EVALUATED.json"
        if base.exists():
            protected.append(base)
        identity = {str(p): q.file_sha(p) for p in protected}
        receipt = dict(status="started", before=before, original_state=state, pbs_final=final,
                       protected_hashes=identity, helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        q.write(audit / "repair.json", receipt)
        with (audit / "before-freeze.txt").open("w") as output:
            subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=output, check=True)
        try:
            if before.get("pyarrow") != ARROW_VERSION:
                subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps",
                                "--only-binary=:all:", "pyarrow==" + ARROW_VERSION], check=True)
            subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
            subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--check", "--smoke"], check=True)
            after = installed()
            if ({k: v for k, v in before.items() if k != "pyarrow"} !=
                    {k: v for k, v in after.items() if k != "pyarrow"}):
                raise ValueError("A dependency other than Arrow changed; do not resume")
            q.verify(campaign)
            if identity != {str(p): q.file_sha(p) for p in protected}:
                raise ValueError("Campaign identity/evaluation marker changed; do not resume")
            with (audit / "after-freeze.txt").open("w") as output:
                subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=output, check=True)
            receipt.update(status="verified", after=after, cpu_smoke=True)
            q.write(audit / "repair.json", receipt)
        except Exception as exc:
            receipt.update(status="failed", error=str(exc))
            q.write(audit / "repair.json", receipt)
            raise
    print(f"REPAIR_VERIFIED={audit}; frozen campaign identity and cached evaluation preserved", flush=True)
    if resume:
        q.start(campaign, resume=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.check:
        if args.campaign or args.resume:
            parser.error("--check is separate from campaign repair")
        check_stack(args.smoke)
    else:
        if not args.campaign or args.smoke:
            parser.error("repair requires --campaign; smoke is included automatically")
        try:
            repair(args.campaign, args.resume)
        except BlockingIOError:
            raise SystemExit("Campaign supervisor/worker/start is still active. Wait 30 seconds and rerun; no environment change made.")


if __name__ == "__main__":
    main()
