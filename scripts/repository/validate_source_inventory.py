#!/usr/bin/env python3
"""Validate the paper-facing Phase 3D-R source and data layout offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]
PHASE_BASE = "623e305655a10a87685e49d83404fe7cd5f2ed81"
EXPECTED_TRACKED = 718
REQUIRED_PREFIXES = (
    "src/conrep/",
    "src/clinicia/",
    "third_party/llm2vec/",
    "third_party/open-unlearning/",
    "legacy/",
    "experiments/paper_runs/",
    "data/clinicia/",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def git(*args: str, text: bool = False):
    env = {"GIT_NO_LAZY_FETCH": "1"}
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        text=text,
    ).stdout


def load_json(path: pathlib.Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load JSON {path}: {exc}") from exc


def index_entries() -> dict[str, str]:
    entries: dict[str, str] = {}
    for row in git("ls-files", "-s", "-z").split(b"\0"):
        if not row:
            continue
        metadata, raw_path = row.split(b"\t", 1)
        _mode, blob, stage = metadata.split()
        require(stage == b"0", f"non-zero index stage: {raw_path!r}")
        entries[raw_path.decode("utf-8")] = blob.decode("ascii")
    return entries


def tree_entries(ref: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for row in git("ls-tree", "-rz", ref).split(b"\0"):
        if not row:
            continue
        metadata, raw_path = row.split(b"\t", 1)
        _mode, kind, blob = metadata.split()
        if kind == b"blob":
            entries[raw_path.decode("utf-8")] = blob.decode("ascii")
    return entries


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_catalog(entries: dict[str, str], base: dict[str, str], path: pathlib.Path) -> int:
    catalog = load_json(path)
    require(catalog.get("record_kind") == "clinicia_dataset_catalog", "wrong catalog kind")
    rows = catalog.get("datasets")
    require(isinstance(rows, list) and len(rows) == 28, "expected 28 ClinicIA datasets")
    current = [row.get("path") for row in rows]
    historical = [row.get("historical_path") for row in rows]
    require(len(set(current)) == len(rows), "duplicate current ClinicIA path")
    require(len(set(historical)) == len(rows), "duplicate historical ClinicIA path")
    for row in rows:
        old_path = row["historical_path"]
        new_path = row["path"]
        blob = row["git_blob_oid"]
        require(new_path.startswith("data/clinicia/"), f"non-semantic ClinicIA path: {new_path}")
        require(entries.get(new_path) == blob, f"current ClinicIA blob mismatch: {new_path}")
        require(base.get(old_path) == blob, f"historical ClinicIA blob mismatch: {old_path}")
        require((ROOT / new_path).is_file(), f"missing ClinicIA file: {new_path}")
    return len(rows)


def validate_selected_sources(entries: dict[str, str], path: pathlib.Path) -> tuple[int, int]:
    inventory = load_json(path)
    rows = inventory.get("sources")
    require(isinstance(rows, list) and len(rows) == 16, "expected 16 selected historical files")
    require(inventory.get("materialized_file_count") == 16, "selected materialized count mismatch")
    require(inventory.get("selected_experiment_count") == 5, "selected experiment count mismatch")
    materialized = [row.get("materialized_path") for row in rows]
    require(len(set(materialized)) == len(rows), "duplicate selected materialized path")
    experiment_ids = {row.get("experiment_id") for row in rows}
    require(len(experiment_ids) == 5, "expected five selected ConRep runs")
    trees: dict[str, dict[str, str]] = {}
    for row in rows:
        old_commit = row["starting_git_commit"]
        old_path = row["path"]
        new_path = row["materialized_path"]
        trees.setdefault(old_commit, tree_entries(old_commit))
        require(
            trees[old_commit].get(old_path) == row["git_blob_object_id"],
            f"historical selected source blob mismatch: {old_path}",
        )
        require(new_path in entries, f"selected source is not tracked: {new_path}")
        require((ROOT / new_path).is_file(), f"selected source is not materialized: {new_path}")
        require(sha256(ROOT / new_path) == row["sha256"], f"selected source SHA-256 mismatch: {new_path}")
    return len(rows), len(experiment_ids)


def validate(catalog_path: pathlib.Path, selected_path: pathlib.Path) -> tuple[int, int, int, int]:
    entries = index_entries()
    require(len(entries) == EXPECTED_TRACKED, f"expected {EXPECTED_TRACKED} tracked paths, found {len(entries)}")
    require(not any(path.startswith("llm2vec/") for path in entries), "obsolete top-level llm2vec path remains")
    for prefix in REQUIRED_PREFIXES:
        require(any(path.startswith(prefix) for path in entries), f"required layout prefix is empty: {prefix}")
    base = tree_entries(PHASE_BASE)
    dataset_count = validate_catalog(entries, base, catalog_path)
    source_count, experiment_count = validate_selected_sources(entries, selected_path)
    return len(entries), dataset_count, source_count, experiment_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=pathlib.Path, default=ROOT / "data/clinicia/catalog.json")
    parser.add_argument(
        "--selected-sources",
        type=pathlib.Path,
        default=ROOT / "results/repository_selected_legacy_sources.json",
    )
    args = parser.parse_args(argv)
    try:
        tracked, datasets, sources, experiments = validate(
            args.catalog.resolve(), args.selected_sources.resolve()
        )
    except (ValueError, subprocess.CalledProcessError) as exc:
        print(f"source inventory validation failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"validated paper-facing layout: {tracked} tracked paths, {datasets} ClinicIA datasets, "
        f"{sources} selected historical files across {experiments} ConRep runs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
