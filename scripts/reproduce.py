#!/usr/bin/env python3
"""Paper-facing reproduction entry points.

The lightweight commands in this module remain importable without installing
either model environment.  GPU commands are assembled here so the repository
can expose the exact current invocation without importing model dependencies.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPEN_UNLEARNING_ROOT = ROOT / "third_party" / "open-unlearning"
PAPER_CONFIG_ROOT = ROOT / "configs" / "paper" / "historical"
DATA_CATALOG = ROOT / "data" / "clinicia" / "catalog.json"
LFS_POINTER_HEADER = b"version https://git-lfs.github.com/spec/v1\n"
sys.path.insert(0, str(ROOT / "src"))

from clinicia.runs import TABLES, get_table  # noqa: E402


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _display_command(command: list[str], *, cwd: Path) -> None:
    print(f"working_directory={cwd.relative_to(ROOT)}")
    print(f"command={shlex.join(command)}")


def _is_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return stream.read(len(LFS_POINTER_HEADER)) == LFS_POINTER_HEADER


def _require_materialized(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if _is_lfs_pointer(path):
        raise RuntimeError(
            f"{path} is still a Git LFS pointer; materialize the authorized "
            "object or provide an equivalent local file before execution"
        )


def _require_new_result_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    paper_root = (ROOT / "results" / "paper").resolve()
    if resolved == paper_root or paper_root in resolved.parents:
        raise ValueError("model runs may not write into immutable results/paper")
    return resolved


def _model_reference(value: str) -> str:
    """Keep registry IDs intact and make repository-local model paths portable."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    if value.startswith(("./", "../", "results/")) or path.exists():
        return str((ROOT / path).resolve())
    return value


def data_status(*, require_materialized: bool) -> int:
    catalog = _load_json(DATA_CATALOG)
    states = {"materialized": [], "lfs_pointer": [], "missing": []}
    for row in catalog["datasets"]:
        relative = row["path"]
        path = ROOT / relative
        if not path.is_file():
            states["missing"].append(relative)
        elif _is_lfs_pointer(path):
            states["lfs_pointer"].append(relative)
        else:
            states["materialized"].append(relative)
    print(
        f"catalogued={sum(map(len, states.values()))} "
        f"materialized={len(states['materialized'])} "
        f"lfs_pointers={len(states['lfs_pointer'])} "
        f"missing={len(states['missing'])}"
    )
    for state in ("lfs_pointer", "missing"):
        for relative in states[state]:
            print(f"{state}\t{relative}")
    if require_materialized and (states["lfs_pointer"] or states["missing"]):
        return 2
    return 0


def _run_or_show(
    command: list[str],
    *,
    cwd: Path,
    dry_run: bool,
    required_inputs: tuple[Path, ...],
) -> int:
    _display_command(command, cwd=cwd)
    if dry_run:
        return 0
    for path in required_inputs:
        _require_materialized(path)
    runtime_command = [sys.executable, *command[1:]]
    subprocess.run(runtime_command, cwd=cwd, check=True)
    return 0


def sft_pmc(args: argparse.Namespace) -> int:
    train_data = args.train_data.expanduser().resolve()
    output_dir = _require_new_result_path(args.output_dir)
    command = [
        "python",
        "src/train.py",
        "--config-name=train.yaml",
        "experiment=finetune/pmc/default",
        f"model.model_args.pretrained_model_name_or_path={args.model}",
        f"model.tokenizer_args.pretrained_model_name_or_path={args.model}",
        f"data.train.pmc_hybrid.args.hf_args.data_files={train_data}",
        f"paths.output_dir={output_dir}",
        "task_name=validated_v2_pmc_sft",
    ]
    print("paper_experiment=Regime B PMC starting-model preparation")
    print(f"input={train_data}")
    print(f"output={output_dir}")
    print("verification=config-derived; command rendering tested; numerical run not verified")
    return _run_or_show(
        command,
        cwd=OPEN_UNLEARNING_ROOT,
        dry_run=args.dry_run,
        required_inputs=(train_data,),
    )


def _baseline_experiment_config(record: dict) -> str:
    method = record["method"].lower()
    method_dir = "grandiff" if method == "graddiff" else method
    target = record["knowledge_target"]
    if record["regime"] == "B":
        group = f"PMC_{method_dir}"
    else:
        config_target = "death" if target == "deaths" else target
        group = f"celebrity_{config_target}_{method_dir}"
    filename = "default_llama2" if "llama2" in record["experiment_id"] else "default"
    relative = Path("configs") / "experiment" / "unlearn" / group / f"{filename}.yaml"
    if not (OPEN_UNLEARNING_ROOT / relative).is_file():
        raise FileNotFoundError(OPEN_UNLEARNING_ROOT / relative)
    return f"unlearn/{group}/{filename}"


def baseline_unlearn(args: argparse.Namespace) -> int:
    record_path = PAPER_CONFIG_ROOT / f"{args.experiment_id}.json"
    if not record_path.is_file():
        raise ValueError(f"unknown paper experiment: {args.experiment_id}")
    record = _load_json(record_path)
    if record.get("backend") != "open_unlearning" or record.get("method") not in {
        "GradDiff",
        "NPO",
        "RMU",
    }:
        raise ValueError(
            "baseline-unlearn accepts paper GradDiff, NPO, or RMU experiment IDs"
        )

    model_path = _model_reference(args.model_path)
    tokenizer_path = _model_reference(args.tokenizer_path or args.model_path)
    forget_data = args.forget_data.expanduser().resolve()
    output_dir = _require_new_result_path(args.output_dir)
    experiment_config = _baseline_experiment_config(record)
    command = [
        "python",
        "src/train.py",
        "--config-name=unlearn.yaml",
        f"experiment={experiment_config}",
        f"model.model_args.pretrained_model_name_or_path={model_path}",
        f"model.tokenizer_args.pretrained_model_name_or_path={tokenizer_path}",
        (
            "data.forget.CELEBRITY_ALL_FORGET.args.hf_args.data_files="
            f"[{forget_data}]"
        ),
        f"paths.output_dir={output_dir}",
        f"task_name={args.experiment_id}",
    ]
    for name, value in sorted(record.get("resolved_hyperparameters", {}).items()):
        command.append(f"trainer.args.{name}={value}")
    required_inputs = [forget_data]
    if record["regime"] == "B":
        if args.retain_data is None:
            raise ValueError("Regime B requires --retain-data")
        retain_data = args.retain_data.expanduser().resolve()
        command.append(
            "data.retain.CELEBRITY_ALL_FORGET.args.hf_args.data_files="
            f"[{retain_data}]"
        )
        required_inputs.append(retain_data)

    print(f"paper_experiment={args.experiment_id}")
    print(f"input_model={model_path}")
    print(f"input_forget={forget_data}")
    if record["regime"] == "B":
        print(f"input_retain={required_inputs[-1]}")
    else:
        print("input_retain=OpenUnlearning WikiTextRetainDataset (network or cache required)")
    print(f"output={output_dir}")
    if record.get("reproduction_status") == "unresolved":
        print(
            "verification=evidence-only; historical model/target identity conflicts; "
            "command is a current canonical-cell proposal; numerical run not verified"
        )
    else:
        print(
            "verification=config-derived; archived metrics/config resolved; current "
            "command reconstructed; historical training entry point and numerical "
            "rerun not verified"
        )
    return _run_or_show(
        command,
        cwd=OPEN_UNLEARNING_ROOT,
        dry_run=args.dry_run,
        required_inputs=tuple(required_inputs),
    )


def rebuild_tables() -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/results/extract_archived_metrics.py",
            "--manifest",
            "results/paper/manifest.json",
            "--output-dir",
            "results/paper/raw_metrics",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/results/build_tables.py",
            "--manifest",
            "results/paper/manifest.json",
            "--output-dir",
            "results/paper/reconstructed",
        ],
        cwd=ROOT,
        check=True,
    )


def show_table(number: int, *, rebuild: bool) -> int:
    table = get_table(number)
    if rebuild:
        rebuild_tables()
    path = table.path()
    if not path.is_file():
        raise FileNotFoundError(path)
    print(f"Paper Table {number}: Regime {table.regime}, target={table.target}")
    print(path.relative_to(ROOT))
    if table.view == "normalized":
        print(path.read_text(encoding="utf-8"), end="")
    else:
        print("Raw values and their evidence/status fields are in reconciliation.csv.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    table_parser = subparsers.add_parser("table", help="show one of paper Tables 1-6")
    table_parser.add_argument("number", type=int, choices=TABLES)
    table_parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="read the committed reconstruction without regenerating it",
    )
    subparsers.add_parser("tables", help="list all paper table mappings")
    subparsers.add_parser(
        "rebuild-tables",
        help="rebuild every archived paper table from the committed evidence",
    )
    data_parser = subparsers.add_parser(
        "data-status",
        help="report which ClinicIA inputs are materialized files versus LFS pointers",
    )
    data_parser.add_argument(
        "--require-materialized",
        action="store_true",
        help="exit nonzero when any catalogued input is missing or still an LFS pointer",
    )
    sft = subparsers.add_parser(
        "sft-pmc",
        help="prepare the Regime B PMC starting model with the vendored OpenUnlearning backend",
    )
    sft.add_argument(
        "--train-data",
        type=Path,
        default=Path("data/clinicia/regime_b/pmc/training/easy_QA_PMC_full.csv"),
    )
    sft.add_argument(
        "--model",
        default="mistralai/Mistral-7B-Instruct-v0.2",
    )
    sft.add_argument("--output-dir", type=Path, required=True)
    sft.add_argument("--dry-run", action="store_true")

    baseline = subparsers.add_parser(
        "baseline-unlearn",
        help="run a paper GradDiff, NPO, or RMU configuration",
    )
    baseline.add_argument("experiment_id")
    baseline.add_argument("--model-path", required=True)
    baseline.add_argument("--tokenizer-path")
    baseline.add_argument("--forget-data", type=Path, required=True)
    baseline.add_argument("--retain-data", type=Path)
    baseline.add_argument("--output-dir", type=Path, required=True)
    baseline.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "table":
        return show_table(args.number, rebuild=not args.no_rebuild)
    if args.command == "rebuild-tables":
        rebuild_tables()
        print("rebuilt results/paper/raw_metrics and results/paper/reconstructed")
        return 0
    if args.command == "data-status":
        return data_status(require_materialized=args.require_materialized)
    if args.command == "sft-pmc":
        return sft_pmc(args)
    if args.command == "baseline-unlearn":
        return baseline_unlearn(args)
    for number, table in TABLES.items():
        print(f"{number}\tRegime {table.regime}\t{table.target}\t{table.view}\t{table.artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
