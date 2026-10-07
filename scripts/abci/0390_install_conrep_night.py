#!/usr/bin/env python3
"""Install only this additive package from a fetched ref; never overwrite server changes."""
import argparse
from pathlib import Path
import subprocess

PREFIXES = ("src/conrep/night/", "tests/0390/test_conrep_night.py", "docs/0390/conrep-night.md")
EXACT = {"scripts/abci/0390_conrep_night.py", "scripts/abci/0390_conrep_night_worker.sh",
         "scripts/abci/0390_install_conrep_night.py"}


def install(root, ref):
    root = root.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "--verify", ref + "^{commit}"], cwd=root, text=True).strip()
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit], cwd=root, text=True).splitlines()
    selected = [n for n in names if n in EXACT or n.startswith(PREFIXES)]
    if not EXACT <= set(selected):
        raise ValueError("The fetched ref does not contain the complete overnight package")
    payloads = {}
    for name in selected:
        target = (root / name).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Destination escapes checkout: {name}")
        payload = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root)
        if target.exists() and target.read_bytes() != payload:
            raise FileExistsError(f"Existing different file preserved: {name}; no files installed")
        payloads[target] = payload
    for target, payload in payloads.items():
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(payload)
    print(f"Installed {len(payloads)} additive files from {commit}. Existing trainers/evaluators untouched.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--ref", required=True)
    a = p.parse_args()
    install(a.root, a.ref)
