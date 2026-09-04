#!/usr/bin/env python3
"""Inspect or launch one of the five selected ConRep paper runs."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conrep.entrypoints import DEFAULT_VARIANT, VARIANTS, run_variant  # noqa: E402
from conrep.runs import get_run, load_runs, normalized_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list selected paper runs")
    show = subparsers.add_parser("show", help="describe one selected paper run")
    show.add_argument("experiment_id")
    config = subparsers.add_parser(
        "config",
        help="print the portable config that a selected paper run would use",
    )
    add_portable_arguments(config)
    run = subparsers.add_parser("run", help="launch one selected paper trainer")
    add_portable_arguments(run)
    subparsers.add_parser("legacy-list", help="list all preserved trainer variants")
    legacy = subparsers.add_parser("legacy-run", help="launch a preserved trainer directly")
    legacy.add_argument("--variant", choices=sorted(VARIANTS), default=DEFAULT_VARIANT)
    legacy.add_argument("passthrough", nargs=argparse.REMAINDER)
    return parser


def add_portable_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("experiment_id")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        help="override model_name_or_path; required for a portable Regime B rerun",
    )
    parser.add_argument(
        "--peft-model",
        help="override peft_model_name_or_path; use 'none' for no starting adapter",
    )
    parser.add_argument("--retain-data", type=Path)
    parser.add_argument("--forget-data", type=Path)


def portable_config(args: argparse.Namespace) -> dict:
    config = normalized_config(
        args.experiment_id,
        "train",
        output_root=args.output_root,
    )
    if args.model_path is not None:
        config["model_name_or_path"] = args.model_path
    if args.peft_model is not None:
        config["peft_model_name_or_path"] = (
            None if args.peft_model.lower() == "none" else args.peft_model
        )
    if args.retain_data is not None:
        config["retain_csv_path"] = str(args.retain_data.expanduser().resolve())
    if args.forget_data is not None:
        config["forget_csv_path"] = str(args.forget_data.expanduser().resolve())
    return config


def show_run(run_id: str) -> None:
    run = get_run(run_id)
    print(
        json.dumps(
            {
                "experiment_id": run.experiment_id,
                "regime": run.regime,
                "target": run.target,
                "model_id": run.model_id,
                "paper_tables": list(run.paper_tables),
                "historical_files": dict(run.historical_files),
                "portable_execution": (
                    "run with --output-root; recognized historical data paths "
                    "are mapped through data/clinicia/catalog.json"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        for run_id, run in sorted(load_runs().items()):
            print(f"{run_id}\tRegime {run.regime}\t{run.target}\t{run.model_id}")
        return 0
    if args.command == "show":
        show_run(args.experiment_id)
        return 0
    if args.command == "config":
        print(json.dumps(portable_config(args), indent=2, sort_keys=True))
        return 0
    if args.command == "legacy-list":
        for name in sorted(VARIANTS):
            marker = " (selected default)" if name == DEFAULT_VARIANT else ""
            print(f"{name}\t{VARIANTS[name]}{marker}")
        return 0
    if args.command == "legacy-run":
        run_variant(args.variant, args.passthrough)
        return 0

    get_run(args.experiment_id)
    config = portable_config(args)
    with tempfile.TemporaryDirectory(prefix="conrep-config-") as temporary:
        path = Path(temporary) / "train.json"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        run_variant(DEFAULT_VARIANT, [str(path)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
