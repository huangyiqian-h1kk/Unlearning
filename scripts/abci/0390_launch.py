#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--nproc", type=int, default=8)
p.add_argument("args", nargs=argparse.REMAINDER)
args = p.parse_args()
argv = args.args[1:] if args.args[:1] == ["--"] else args.args
if not argv:
    p.error("An experiment stage is required after --")
root = Path(__file__).resolve().parents[2]
os.chdir(root)
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < args.nproc:
    raise RuntimeError(
        f"Requested {args.nproc} GPU processes, visible GPUs={torch.cuda.device_count()}"
    )
if argv[0] == "sft":
    # Check the GPU-only DeepSpeed import path once before spawning workers or
    # loading model weights; CPU dependency tests cannot exercise this path.
    import deepspeed

    print(
        f"DeepSpeed import passed: {deepspeed.__version__}; PyTorch CUDA: {torch.version.cuda}",
        flush=True,
    )
command = [
    sys.executable,
    "-m",
    "torch.distributed.run",
    "--standalone",
    "--nnodes=1",
    f"--nproc_per_node={args.nproc}",
    "scripts/0390_experiment.py",
    *argv,
]
if args.nproc == 1:
    command = [sys.executable, "scripts/0390_experiment.py", *argv]
print("Launching:", command, flush=True)
subprocess.run(command, check=True)
