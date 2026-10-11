#!/usr/bin/env python3
"""Freeze the verified parent evaluator and overlay Qwen-only files from Git."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

OVERLAY = ["src/experiments/qwen3_rich.py", "scripts/abci/0390_qwen3_rich.py",
           "scripts/abci/0390_qwen3_worker.sh", "scripts/abci/0390_setup_qwen3_env.sh",
           "environments/0390/requirements-qwen3.txt", "configs/0390/qwen3_4b_rich_sft.json",
           "configs/0390/zero2.json", "scripts/abci/0390_repair_qwen3_arrow.py"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root, parent, campaign, ref):
    root, parent, campaign = (Path(p).resolve() for p in (root, parent, campaign))
    if campaign == parent or campaign.is_relative_to(parent) or parent.is_relative_to(campaign):
        raise ValueError("Use separate parent and Qwen campaign directories")
    plan = json.loads((parent / "plan.json").read_text())
    source = json.loads((parent / "source.json").read_text())
    if Path(plan["project_root"]).resolve() != root or source["source_hash"] != plan["source_hash"]:
        raise ValueError("Parent project/source identity mismatch")
    for name, expected in source["files"].items():
        path = (parent / "code" / name).resolve()
        if not path.is_relative_to(parent / "code") or digest(path) != expected:
            raise ValueError(f"Parent source changed: {name}")
    commit = subprocess.check_output(["git", "rev-parse", "--verify", ref + "^{commit}"], cwd=root, text=True).strip()
    payloads = {n: subprocess.check_output(["git", "show", f"{commit}:{n}"], cwd=root) for n in OVERLAY}
    metadata = dict(ref=commit, parent=str(parent), parent_source_hash=source["source_hash"],
                    project_root=str(root), overlay={n: hashlib.sha256(b).hexdigest() for n,b in payloads.items()})
    campaign.mkdir(parents=True, exist_ok=True)
    with (campaign / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = campaign / "PREPARING.json"
        if marker.exists():
            if json.loads(marker.read_text()) != metadata:
                raise ValueError("Different preparation already exists")
        else:
            if set(p.name for p in campaign.iterdir()) != {"prepare.lock"}:
                raise ValueError("Destination must be empty")
            marker.write_text(json.dumps(metadata, indent=2) + "\n")
        expected = {**source["files"], **metadata["overlay"]}
        code = campaign / "code"
        if not code.exists():
            staging = campaign / ".code-incomplete"
            if staging.exists(): shutil.rmtree(staging)
            for name in source["files"]:
                dest = staging / name; dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(parent / "code" / name, dest)
            for name, data in payloads.items():
                dest = staging / name; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(data)
            staging.rename(code)
        for name, value in expected.items():
            if digest(code / name) != value: raise ValueError(f"Frozen code changed: {name}")
        (campaign / "code-files.json").write_text(json.dumps(expected, indent=2) + "\n")
    print(f"FROZEN_CODE={code}\nNo jobs submitted. Configure the isolated environment, then run prepare.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True); p.add_argument("--parent", required=True)
    p.add_argument("--campaign", required=True); p.add_argument("--ref", required=True)
    a = p.parse_args(); prepare(a.root, a.parent, a.campaign, a.ref)
