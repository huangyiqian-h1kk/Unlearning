#!/usr/bin/env python3
"""Install only this additive package from a fetched ref; never overwrite server changes."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

PREFIXES = ("src/conrep/night/", "tests/0390/test_conrep_night.py", "docs/0390/conrep-night.md")
EXACT = {"scripts/abci/0390_conrep_night.py", "scripts/abci/0390_conrep_night_worker.sh",
         "scripts/abci/0390_install_conrep_night.py"}


def install(root, ref, upgrade_from=None):
    root = root.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "--verify", ref + "^{commit}"], cwd=root, text=True).strip()
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit], cwd=root, text=True).splitlines()
    selected = [n for n in names if n in EXACT or n.startswith(PREFIXES)]
    if not EXACT <= set(selected):
        raise ValueError("The fetched ref does not contain the complete overnight package")
    previous = None
    if upgrade_from:
        previous = subprocess.check_output(["git", "rev-parse", "--verify", upgrade_from + "^{commit}"],
                                           cwd=root, text=True).strip()
    payloads = {}
    replacements = []
    for name in selected:
        target = (root / name).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Destination escapes checkout: {name}")
        payload = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root)
        if target.exists() and target.read_bytes() != payload:
            old = subprocess.run(["git", "show", f"{previous}:{name}"], cwd=root,
                                 capture_output=True) if previous else None
            if old is None or old.returncode or old.stdout != target.read_bytes():
                raise FileExistsError(f"Existing different file preserved: {name}; no files installed")
            replacements.append(target)
        payloads[target] = payload
    # All collisions are checked before installing anything. A package upgrade
    # only replaces exact copies of the named previous revision, with backups.
    if replacements:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = root / "logs/0390/installations" / (stamp + "-" + uuid.uuid4().hex[:8])
        for target in replacements:
            saved = backup / target.relative_to(root)
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
        (backup / "upgrade.json").write_text(json.dumps({"from": previous, "to": commit,
            "replaced": [str(p.relative_to(root)) for p in replacements]}, indent=2) + "\n")
    for target, payload in payloads.items():
        if target in replacements:
            temporary = target.with_name(target.name + ".install-" + uuid.uuid4().hex)
            temporary.write_bytes(payload)
            os.replace(temporary, target)
        elif not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(payload)
    print(f"Installed {len(payloads)} additive files from {commit}. Existing trainers/evaluators untouched.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--ref", required=True)
    p.add_argument("--upgrade-from", help="Allow replacement only when the installed bytes match this old ref")
    a = p.parse_args()
    install(a.root, a.ref, a.upgrade_from)
