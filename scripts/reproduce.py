#!/usr/bin/env python3
"""Inspect or rebuild the archived paper tables without loading a model."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinicia.runs import TABLES, get_table  # noqa: E402


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
    args = parser.parse_args(argv)
    if args.command == "table":
        return show_table(args.number, rebuild=not args.no_rebuild)
    for number, table in TABLES.items():
        print(f"{number}\tRegime {table.regime}\t{table.target}\t{table.view}\t{table.artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
