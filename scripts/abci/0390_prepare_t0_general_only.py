#!/usr/bin/env python3
"""Freeze a separate T0 general-only supplement from an existing selective campaign."""
import argparse
import ast
from pathlib import Path
import subprocess
import types


def prepare(root, parent, campaign, ref):
    root, parent = Path(root).resolve(), Path(parent).resolve()
    entry = 'scripts/abci/0390_prepare_selective.py'
    code = subprocess.check_output(['git', 'show', f'{ref}:{entry}'], cwd=root)
    module = types.ModuleType('selective_freezer')
    exec(compile(code, entry, 'exec'), module.__dict__)
    reader = ast.parse((parent/'code/src/experiments/selective_data.py').read_text())
    rows = next(n for n in reader.body if isinstance(n, ast.FunctionDef) and n.name == 'rows')
    if any(isinstance(n, ast.Attribute) and n.attr == 'splitlines' for n in ast.walk(rows)):
        raise ValueError('Apply the published MMLU reader repair to the parent first')
    module.OVERLAY += ['src/experiments/t0_general_only.py',
                       'scripts/abci/0390_t0_general_only.py',
                       'scripts/abci/0390_t0_general_only_worker.sh']
    module.prepare(root, parent, campaign, ref)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'parent', 'campaign', 'ref'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    prepare(args.root, args.parent, args.campaign, args.ref)
