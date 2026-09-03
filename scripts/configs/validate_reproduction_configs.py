#!/usr/bin/env python3
"""Validate portable future-run configuration records without executing them."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]
ENV_PATH = re.compile(r"^\$\{[A-Z][A-Z0-9_]*\}(?:/[A-Za-z0-9_.-]+)*$")
PRIVATE_MARKERS = ("/home/", "/gs/", "/workspace/", "\\Users\\", "C:\\")


def load_json(path: pathlib.Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load JSON {path}: {exc}") from exc


def repository_path(raw: str) -> pathlib.Path:
    pure = pathlib.PurePosixPath(raw)
    if not raw or pure.is_absolute() or ".." in pure.parts or "\\" in raw:
        raise ValueError(f"path is not repository-relative POSIX: {raw!r}")
    return ROOT.joinpath(*pure.parts)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_index(index_path: pathlib.Path) -> tuple[int, int, int]:
    index = load_json(index_path)
    require(index.get("schema_version") == "1.0", "unsupported index schema")
    require(index.get("record_kind") == "reproduction_config_index", "wrong index kind")

    declared_registries = index.get("component_registries")
    require(isinstance(declared_registries, dict), "component_registries must be an object")
    require(
        set(declared_registries) == {"datasets", "methods", "models", "protocols"},
        "exactly four component registries are required",
    )
    registries = {}
    for kind, raw_path in declared_registries.items():
        path = repository_path(raw_path)
        record = load_json(path)
        require(record.get("schema_version") == "1.0", f"unsupported registry schema: {raw_path}")
        require(record.get("record_kind") == "reproduction_component_registry", f"wrong registry kind: {raw_path}")
        require(record.get("component_kind") == kind, f"registry kind mismatch: {raw_path}")
        values = record.get(kind)
        require(isinstance(values, dict) and values, f"empty component registry: {raw_path}")
        registries[kind] = values

    dataset_record = load_json(repository_path(declared_registries["datasets"]))
    inventory = load_json(repository_path(dataset_record["inventory"]))
    inventory_paths = {row["path"] for row in inventory.get("datasets", [])}
    require(set(registries["datasets"].values()) == inventory_paths, "dataset registry and paper inventory differ")
    require(dataset_record.get("redistribution_status") == "unresolved_do_not_redistribute", "dataset rights must remain explicit")

    for method_id, method in registries["methods"].items():
        require(isinstance(method, dict), f"method must be an object: {method_id}")
        entrypoint = method.get("entrypoint")
        if entrypoint is not None:
            require(repository_path(entrypoint).is_file(), f"missing method entry point: {entrypoint}")

    candidate_paths = [repository_path(path) for path in index.get("candidates", [])]
    expected_candidates = {
        path
        for path in (ROOT / "configs" / "reproduction").glob("*.json")
        if path.name != "index.json"
    }
    require(set(candidate_paths) == expected_candidates, "candidate index does not exactly cover reproduction JSON files")

    runnable_count = 0
    for path in candidate_paths:
        record = load_json(path)
        label = path.relative_to(ROOT).as_posix()
        require(record.get("schema_version") == "1.0", f"unsupported candidate schema: {label}")
        require(record.get("record_kind") == "reproduction_candidate", f"wrong candidate kind: {label}")
        require(record.get("model") in registries["models"], f"unknown model in {label}")
        require(record.get("method") in registries["methods"], f"unknown method in {label}")
        require(record.get("protocol") in registries["protocols"], f"unknown protocol in {label}")
        dataset_ids = record.get("dataset_ids")
        require(isinstance(dataset_ids, list) and dataset_ids, f"candidate datasets missing: {label}")
        require(len(dataset_ids) == len(set(dataset_ids)), f"duplicate dataset ID: {label}")
        require(set(dataset_ids) <= set(registries["datasets"]), f"unknown dataset ID: {label}")
        entrypoint = record.get("entrypoint")
        require(repository_path(entrypoint).is_file(), f"missing candidate entry point: {entrypoint}")
        require(entrypoint == registries["methods"][record["method"]]["entrypoint"], f"method entry point mismatch: {label}")
        environment = record.get("environment")
        require(isinstance(environment, dict) and environment, f"environment variables missing: {label}")
        for key, value in environment.items():
            require(isinstance(value, str) and ENV_PATH.fullmatch(value), f"non-portable environment path {key}: {value!r}")
            require(not any(marker in value for marker in PRIVATE_MARKERS), f"private path marker in {label}")
        gates = record.get("gates")
        require(isinstance(gates, dict) and gates, f"candidate gates missing: {label}")
        require(all(isinstance(value, bool) for value in gates.values()), f"candidate gates must be boolean: {label}")
        runnable = record.get("runnable")
        require(isinstance(runnable, bool), f"runnable must be boolean: {label}")
        if runnable:
            runnable_count += 1
            require(all(gates.values()), f"runnable candidate has an open gate: {label}")
        require(record.get("historical_equivalence_claimed") is False, f"historical equivalence overclaimed: {label}")
        require(record.get("may_write_archived_paper_results") is False, f"paper results are not writable: {label}")
        protocol = registries["protocols"][record["protocol"]]
        require(protocol.get("may_modify_archived_paper_results") is False, f"protocol may modify paper results: {label}")

    sweep_paths = [repository_path(path) for path in index.get("sweeps", [])]
    expected_sweeps = set((ROOT / "configs" / "sweeps").glob("*.json"))
    require(set(sweep_paths) == expected_sweeps, "sweep index does not exactly cover sweep JSON files")
    candidate_raw_paths = set(index.get("candidates", []))
    for path in sweep_paths:
        record = load_json(path)
        label = path.relative_to(ROOT).as_posix()
        require(record.get("schema_version") == "1.0", f"unsupported sweep schema: {label}")
        require(record.get("record_kind") == "compact_reproduction_sweep", f"wrong sweep kind: {label}")
        require(record.get("status") == "review_only_not_runnable", f"sweep status overclaims readiness: {label}")
        require(record.get("base_candidate") in candidate_raw_paths, f"unknown sweep base candidate: {label}")
        axes = record.get("axes")
        require(isinstance(axes, dict) and axes, f"sweep axes missing: {label}")
        require(set(axes.get("model", [])) <= set(registries["models"]), f"unknown sweep model: {label}")
        require(set(axes.get("method", [])) <= set(registries["methods"]), f"unknown sweep method: {label}")
        require(all(isinstance(values, list) and values for values in axes.values()), f"empty sweep axis: {label}")
        expected = math.prod(len(values) for values in axes.values())
        require(record.get("expected_combinations") == expected, f"sweep product mismatch: {label}")
        require(record.get("expanded_products_tracked") is False, f"expanded sweep products must stay untracked: {label}")
        require(record.get("historical_equivalence_claimed") is False, f"sweep equivalence overclaimed: {label}")

    return len(registries), len(candidate_paths), len(sweep_paths), runnable_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    try:
        counts = validate_index(args.index.resolve())
    except ValueError as exc:
        print(f"reproduction configuration validation failed: {exc}", file=sys.stderr)
        return 1
    registries, candidates, sweeps, runnable = counts
    print(
        f"validated {registries} component registries, {candidates} reproduction candidates, "
        f"and {sweeps} compact sweeps; runnable={runnable}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
