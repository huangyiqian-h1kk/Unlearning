#!/usr/bin/env python3
"""Inspect or launch ClinicIA evaluation for a selected ConRep paper run."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinicia.entrypoints import (  # noqa: E402
    DEFAULT_ENTRYPOINT,
    ENTRYPOINTS,
    run_entrypoint,
)
from conrep.runs import get_run, load_runs, normalized_config  # noqa: E402


def evaluation_roles(run_id: str) -> tuple[str, ...]:
    run = get_run(run_id)
    return tuple(
        role
        for role in ("evaluate_retain", "evaluate_forget")
        if role in run.historical_files
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list selected runs and evaluation roles")
    show = subparsers.add_parser("show", help="show a selected evaluation capsule")
    show.add_argument("experiment_id")
    run = subparsers.add_parser("run", help="launch the preserved likelihood evaluator")
    run.add_argument("experiment_id")
    run.add_argument(
        "--role",
        choices=("evaluate_retain", "evaluate_forget"),
        default="evaluate_forget",
    )
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("passthrough", nargs=argparse.REMAINDER)
    subparsers.add_parser("legacy-list", help="list preserved evaluator programs")
    legacy = subparsers.add_parser("legacy-run", help="launch a preserved evaluator directly")
    legacy.add_argument(
        "--entrypoint",
        choices=sorted(ENTRYPOINTS),
        default=DEFAULT_ENTRYPOINT,
    )
    legacy.add_argument("passthrough", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        for run_id in sorted(load_runs()):
            print(f"{run_id}\t{','.join(evaluation_roles(run_id))}")
        return 0
    if args.command == "show":
        run = get_run(args.experiment_id)
        print(
            json.dumps(
                {
                    "experiment_id": run.experiment_id,
                    "evaluation_roles": evaluation_roles(run.experiment_id),
                    "historical_files": {
                        role: run.historical_files[role]
                        for role in evaluation_roles(run.experiment_id)
                    },
                    "probe_catalog": "src/clinicia/probes.py",
                    "dataset_catalog": "data/clinicia/catalog.json",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "legacy-list":
        for name in sorted(ENTRYPOINTS):
            marker = " (default)" if name == DEFAULT_ENTRYPOINT else ""
            print(f"{name}\t{ENTRYPOINTS[name]}{marker}")
        return 0
    if args.command == "legacy-run":
        run_entrypoint(args.entrypoint, args.passthrough)
        return 0

    roles = evaluation_roles(args.experiment_id)
    if args.role not in roles:
        raise ValueError(
            f"{args.experiment_id} has roles {', '.join(roles)}, not {args.role}"
        )
    config = normalized_config(
        args.experiment_id,
        args.role,
        output_root=args.output_root,
    )
    with tempfile.TemporaryDirectory(prefix="clinicia-config-") as temporary:
        path = Path(temporary) / f"{args.role}.json"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        run_entrypoint("likelihood", [str(path), *args.passthrough])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
