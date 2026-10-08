#!/usr/bin/env python3
"""Prepare the approved 27-run follow-up without overwriting the server checkout.

Extract this file from a fetched ref and run it with python -I. The previous
campaign supplies the frozen model/data/evaluator helpers; only the additive
night package and its entry/worker receive the new revision. No PBS submission.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ENTRY = "scripts/abci/0390_conrep_night.py"
OVERLAY = [ENTRY, "scripts/abci/0390_conrep_night_worker.sh"] + [
    f"src/conrep/night/{name}.py" for name in
    ("__init__", "campaign", "trainer", "io", "losses", "positives", "followup", "diagnostics")]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def prepare(root, source, campaign, ref, *, hours=10, reserve_gb=100):
    root, source, campaign = (Path(p).resolve() for p in (root, source, campaign))
    if campaign == source or campaign.is_relative_to(source) or source.is_relative_to(campaign):
        raise ValueError("Source and follow-up campaigns must be separate, non-nested directories")
    if hours <= 0 or reserve_gb < 0:
        raise ValueError("hours must be positive and reserve-gb nonnegative")
    commit = subprocess.check_output(["git", "rev-parse", "--verify", ref + "^{commit}"], cwd=root, text=True).strip()
    parent = read(source / "plan.json")
    if Path(parent["project_root"]).resolve() != root:
        raise ValueError("Parent campaign belongs to another checkout")
    provenance = read(source / "source.json")
    if provenance["source_hash"] != parent["source_hash"]:
        raise ValueError("Parent source identity mismatch")
    for name, expected in provenance["files"].items():
        path = (source / "code" / name).resolve()
        if not path.is_relative_to(source / "code") or digest(path) != expected:
            raise ValueError(f"Parent frozen source changed: {name}")
    for name, expected in read(source / "inputs.json").items():
        if digest(name) != expected:
            raise ValueError(f"Parent frozen input changed: {name}")
    payloads = {name: subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root)
                for name in OVERLAY}
    metadata = {"ref": commit, "source_campaign": str(source),
                "parent_source_hash": parent["source_hash"],
                "overlay_files": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}}
    campaign.mkdir(parents=True, exist_ok=True)
    with (campaign / "prepare.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        preparing = campaign / "PREPARING.json"
        if preparing.exists():
            if read(preparing) != metadata:
                raise ValueError("Existing preparation has different provenance; no files changed")
        else:
            if set(p.name for p in campaign.iterdir()) != {"prepare.lock"}:
                raise FileExistsError("Destination is not an empty follow-up campaign")
            write(preparing, metadata)
        code = campaign / "code"
        if not code.exists():
            staging = campaign / ".code-bootstrap-incomplete"
            if staging.exists():
                shutil.rmtree(staging)  # Only this bootstrap's reconstructible staging directory.
            staging.mkdir()
            for name in provenance["files"]:
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / "code" / name, target)
            for name, data in payloads.items():
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            staging.rename(code)
        expected_files = dict(provenance["files"])
        expected_files.update(metadata["overlay_files"])
        for name, expected in expected_files.items():
            if digest(code / name) != expected:
                raise ValueError(f"Follow-up snapshot collision: {name}")
        command = [sys.executable, "-I", str(code / ENTRY), "prepare-followup",
                   "--project-root", str(root), "--source-campaign", str(source),
                   "--campaign", str(campaign), "--ref", commit,
                   "--hours", str(hours), "--reserve-gb", str(reserve_gb)]
        subprocess.run(command, cwd=root, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-campaign", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--hours", type=float, default=10)
    parser.add_argument("--reserve-gb", type=float, default=100)
    args = parser.parse_args()
    prepare(args.root, args.source_campaign, args.campaign, args.ref,
            hours=args.hours, reserve_gb=args.reserve_gb)


if __name__ == "__main__":
    main()
