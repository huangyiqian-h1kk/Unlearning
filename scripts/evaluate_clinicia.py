#!/usr/bin/env python3
"""Inspect or launch ClinicIA evaluation for a selected ConRep paper run."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_CONFIG_ROOT = ROOT / "configs" / "paper" / "historical"
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
    paper_config = subparsers.add_parser(
        "paper-config",
        help="print a portable ClinicIA config for any of the 25 paper cells",
    )
    add_model_arguments(paper_config)
    run_model = subparsers.add_parser(
        "run-model",
        help="evaluate any paper-cell model with its matching ClinicIA protocol",
    )
    add_model_arguments(run_model)
    subparsers.add_parser("legacy-list", help="list preserved evaluator programs")
    legacy = subparsers.add_parser("legacy-run", help="launch a preserved evaluator directly")
    legacy.add_argument(
        "--entrypoint",
        choices=sorted(ENTRYPOINTS),
        default=DEFAULT_ENTRYPOINT,
    )
    legacy.add_argument("passthrough", nargs=argparse.REMAINDER)
    return parser


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("experiment_id")
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--role",
        choices=("evaluate_retain", "evaluate_forget"),
        default="evaluate_forget",
    )
    parser.add_argument("--output-root", type=Path, required=True)


def _paper_record(experiment_id: str) -> dict:
    path = PAPER_CONFIG_ROOT / f"{experiment_id}.json"
    if not path.is_file():
        raise ValueError(f"unknown paper experiment: {experiment_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def _template_run_id(experiment_id: str) -> str:
    record = _paper_record(experiment_id)
    paper_model_id = (
        "meta-llama/Llama-2-7b-chat-hf"
        if "-llama2-" in experiment_id
        else "mistralai/Mistral-7B-Instruct-v0.2"
    )
    matches = [
        run.experiment_id
        for run in load_runs().values()
        if run.regime == record["regime"]
        and run.target == record["knowledge_target"]
        and run.model_id == paper_model_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one selected ConRep evaluation template for {experiment_id}, "
            f"found {matches}"
        )
    return matches[0]


def paper_model_config(args: argparse.Namespace) -> dict:
    template_id = _template_run_id(args.experiment_id)
    roles = evaluation_roles(template_id)
    if args.role not in roles:
        raise ValueError(
            f"{args.experiment_id} uses template {template_id} with roles "
            f"{', '.join(roles)}, not {args.role}"
        )
    output_root = args.output_root.expanduser().resolve()
    immutable = (ROOT / "results" / "paper").resolve()
    if output_root == immutable or immutable in output_root.parents:
        raise ValueError("new model evaluations may not write into results/paper")
    config = normalized_config(
        template_id,
        args.role,
        output_root=output_root,
    )
    config["model_path"] = args.model_path
    config["output_dir"] = str(
        output_root / args.experiment_id / "clinicia" / args.role.removeprefix("evaluate_")
    )
    return config


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
    if args.command == "paper-config":
        print(json.dumps(paper_model_config(args), indent=2, sort_keys=True))
        return 0
    if args.command == "legacy-list":
        for name in sorted(ENTRYPOINTS):
            marker = " (default)" if name == DEFAULT_ENTRYPOINT else ""
            print(f"{name}\t{ENTRYPOINTS[name]}{marker}")
        return 0
    if args.command == "legacy-run":
        run_entrypoint(args.entrypoint, args.passthrough)
        return 0

    if args.command == "run-model":
        config = paper_model_config(args)
        with tempfile.TemporaryDirectory(prefix="clinicia-config-") as temporary:
            path = Path(temporary) / f"{args.role}.json"
            path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            run_entrypoint("likelihood", [str(path)])
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
        run_entrypoint("likelihood", [str(path)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
