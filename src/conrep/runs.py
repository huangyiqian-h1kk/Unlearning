"""Paper-run discovery and portable views of historical ConRep configs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = REPOSITORY_ROOT / "experiments" / "paper_runs" / "index.json"
DEFAULT_DATA_CATALOG = REPOSITORY_ROOT / "data" / "clinicia" / "catalog.json"


@dataclass(frozen=True)
class PaperRun:
    experiment_id: str
    regime: str
    target: str
    model_id: str
    paper_tables: tuple[int, ...]
    historical_files: Mapping[str, str]

    def path_for(self, role: str, root: Path = REPOSITORY_ROOT) -> Path:
        try:
            relative = self.historical_files[role]
        except KeyError as exc:
            choices = ", ".join(sorted(self.historical_files))
            raise ValueError(
                f"{self.experiment_id} has no {role!r} file; choose one of: {choices}"
            ) from exc
        return Path(root) / relative


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_runs(index_path: Path = DEFAULT_INDEX) -> Mapping[str, PaperRun]:
    raw = _load_json(index_path)
    runs = {}
    for item in raw["selected_conrep_runs"]:
        run = PaperRun(
            experiment_id=item["experiment_id"],
            regime=item["regime"],
            target=item["target"],
            model_id=item["model_id"],
            paper_tables=tuple(item["paper_tables"]),
            historical_files=dict(item["historical_files"]),
        )
        runs[run.experiment_id] = run
    return runs


def get_run(experiment_id: str) -> PaperRun:
    runs = load_runs()
    try:
        return runs[experiment_id]
    except KeyError as exc:
        choices = ", ".join(sorted(runs))
        raise ValueError(
            f"unknown selected ConRep run {experiment_id!r}; choose one of: {choices}"
        ) from exc


def _data_paths_by_name() -> dict[str, Path]:
    catalog = _load_json(DEFAULT_DATA_CATALOG)
    result: dict[str, Path] = {}
    duplicates: set[str] = set()
    for row in catalog["datasets"]:
        name = Path(row["historical_path"]).name
        if name in result:
            duplicates.add(name)
        result[name] = REPOSITORY_ROOT / row["path"]
    for name in duplicates:
        result.pop(name, None)
    return result


def normalized_config(
    experiment_id: str,
    role: str,
    *,
    output_root: Path,
) -> dict:
    """Load a historical config and replace machine-specific data/output paths.

    The historical file itself is never edited. This produces a transient,
    researcher-reviewable view for a new machine.
    """

    run = get_run(experiment_id)
    raw = _load_json(run.path_for(role))
    by_name = _data_paths_by_name()
    run_output = Path(output_root) / experiment_id

    def visit(value, key: str | None = None):
        if isinstance(value, dict):
            return {name: visit(child, name) for name, child in value.items()}
        if isinstance(value, list):
            return [visit(child, key) for child in value]
        if not isinstance(value, str):
            return value
        if key == "output_dir":
            suffix = "model" if role == "train" else role
            return str(run_output / suffix)
        if key == "model_path":
            return str(run_output / "model")
        if key in {"retain_csv_path", "forget_csv_path"} or key in {
            "evaluation_sets",
            "mcq_sets",
        }:
            replacement = by_name.get(Path(value).name)
            if replacement is not None:
                return str(replacement)
        replacement = by_name.get(Path(value).name)
        if value.startswith("/") and replacement is not None:
            return str(replacement)
        return value

    return visit(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id", nargs="?")
    args = parser.parse_args(argv)
    runs = load_runs()
    if args.experiment_id:
        print(json.dumps(get_run(args.experiment_id).__dict__, indent=2, default=list))
    else:
        for run_id in sorted(runs):
            print(run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
