"""Paper-table locations and selected ClinicIA evaluation configs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PaperTable:
    number: int
    regime: str
    target: str
    view: str
    artifact: str

    def path(self, root: Path = REPOSITORY_ROOT) -> Path:
        return Path(root) / self.artifact


TABLES = {
    1: PaperTable(1, "A", "diagnosis", "normalized", "results/paper/reconstructed/regime_a_diagnosis/table.md"),
    2: PaperTable(2, "B", "pmc", "normalized", "results/paper/reconstructed/regime_b_pmc/table.md"),
    3: PaperTable(3, "A", "deaths", "normalized", "results/paper/reconstructed/regime_a_deaths/table.md"),
    4: PaperTable(4, "A", "diagnosis", "raw-evidence", "results/paper/reconstructed/reconciliation.csv"),
    5: PaperTable(5, "A", "deaths", "raw-evidence", "results/paper/reconstructed/reconciliation.csv"),
    6: PaperTable(6, "B", "pmc", "raw-evidence", "results/paper/reconstructed/reconciliation.csv"),
}


def get_table(number: int) -> PaperTable:
    try:
        return TABLES[number]
    except KeyError as exc:
        raise ValueError("paper table must be an integer from 1 through 6") from exc


def normalized_evaluation_config(
    experiment_id: str,
    role: str,
    *,
    output_root: Path,
) -> dict:
    from conrep.runs import normalized_config

    if role not in {"evaluate_forget", "evaluate_retain"}:
        raise ValueError("evaluation role must be evaluate_forget or evaluate_retain")
    return normalized_config(experiment_id, role, output_root=output_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", nargs="?", type=int)
    args = parser.parse_args(argv)
    if args.table is None:
        for number in TABLES:
            table = TABLES[number]
            print(f"{number}\tRegime {table.regime}\t{table.target}\t{table.view}")
    else:
        table = get_table(args.table)
        print(table.path())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
