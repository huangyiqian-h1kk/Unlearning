#!/usr/bin/env python3
"""Validate the immutable Phase 3D-0 source-migration baseline.

The validator is intentionally offline and reads Git tree/blob metadata.  It does
not import model code, initialize models, download data, or execute experiments.
"""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = ROOT / "docs" / "repository_source_inventory.json"
MAIN_GUARD = re.compile(r"if __name__\s*==\s*['\"]__main__['\"]")


class InventoryError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InventoryError(message)


def git(*args, input_bytes=None):
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def path_hash(paths):
    payload = "".join(path + "\n" for path in sorted(paths)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def identity_hash(entries, paths):
    payload = "".join(
        "{}\0{}\0{}\0{}\n".format(
            path,
            entries[path]["mode"],
            entries[path]["blob_sha"],
            entries[path]["size"],
        )
        for path in sorted(paths)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def set_summary(paths):
    paths = set(paths)
    return {
        "path_count": len(paths),
        "sorted_path_list_sha256": path_hash(paths),
    }


def full_summary(entries, paths):
    paths = set(paths)
    return {
        "path_count": len(paths),
        "ordinary_blob_bytes": sum(entries[path]["size"] for path in paths),
        "sorted_path_list_sha256": path_hash(paths),
        "sorted_identity_list_sha256": identity_hash(entries, paths),
    }


def tree_entries(commit):
    result = {}
    for record in git("ls-tree", "-rlz", commit).split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, object_type, blob_sha, raw_size = metadata.split()
        require(object_type == b"blob", "non-blob recursive tree entry")
        path = raw_path.decode("utf-8")
        require(path not in result, "duplicate Git tree path: {}".format(path))
        result[path] = {
            "mode": mode.decode("ascii"),
            "blob_sha": blob_sha.decode("ascii"),
            "size": int(raw_size),
        }
    return result


def blob_bytes(entry):
    return git("cat-file", "blob", entry["blob_sha"])


def validate_summary(actual, expected, label):
    require(actual == expected, "{} mismatch: actual={!r}, expected={!r}".format(label, actual, expected))


def validate_snapshot(path, expected_component):
    snapshot = load_json(path)
    require(snapshot.get("schema_version") == "1.0", "snapshot schema version")
    require(snapshot.get("record_kind") == "upstream_git_tree_snapshot", "snapshot record kind")
    require(snapshot.get("repository") == expected_component["upstream_repository"], "snapshot repository")
    require(snapshot.get("commit") == expected_component["upstream_commit"], "snapshot commit")
    require(snapshot.get("tree") == expected_component["upstream_tree"], "snapshot tree")
    raw_entries = snapshot.get("entries")
    require(isinstance(raw_entries, list), "snapshot entries must be a list")
    paths = [entry.get("path") for entry in raw_entries]
    require(paths == sorted(paths), "snapshot entries are not Unicode-path sorted")
    require(len(paths) == len(set(paths)), "snapshot contains duplicate paths")
    entries = {}
    for entry in raw_entries:
        path_value = entry.get("path")
        require(isinstance(path_value, str) and path_value, "invalid snapshot path")
        require(re.fullmatch(r"[0-7]{6}", str(entry.get("mode"))), "invalid snapshot mode")
        require(re.fullmatch(r"[0-9a-f]{40}", str(entry.get("blob_sha"))), "invalid snapshot blob SHA")
        require(isinstance(entry.get("size"), int) and entry["size"] >= 0, "invalid snapshot blob size")
        entries[path_value] = {
            "mode": entry["mode"],
            "blob_sha": entry["blob_sha"],
            "size": entry["size"],
        }
    expected = {
        "path_count": snapshot["entry_count"],
        "ordinary_blob_bytes": snapshot["ordinary_blob_bytes"],
        "sorted_path_list_sha256": snapshot["sorted_path_list_sha256"],
        "sorted_identity_list_sha256": snapshot["sorted_identity_list_sha256"],
    }
    validate_summary(full_summary(entries, entries), expected, "{} snapshot census".format(expected_component["component"]))
    return entries


def compare_to_upstream(base_entries, component, upstream):
    prefix = component["local_prefix"]
    excluded = tuple(component.get("excluded_prefixes", []))
    local = {
        path[len(prefix) :]: entry
        for path, entry in base_entries.items()
        if path.startswith(prefix) and not path.startswith(excluded)
    }
    common = set(local) & set(upstream)
    identical = {
        path
        for path in common
        if (local[path]["mode"], local[path]["blob_sha"])
        == (upstream[path]["mode"], upstream[path]["blob_sha"])
    }
    sets = {
        "local": set(local),
        "upstream": set(upstream),
        "common": common,
        "identical": identical,
        "modified": common - identical,
        "local_only": set(local) - set(upstream),
        "upstream_only": set(upstream) - set(local),
    }
    for name, paths in sets.items():
        validate_summary(set_summary(paths), component[name], "{} {} paths".format(component["component"], name))
    modified = [
        {
            "path": path,
            "upstream_blob_sha": upstream[path]["blob_sha"],
            "local_blob_sha": local[path]["blob_sha"],
        }
        for path in sorted(sets["modified"])
    ]
    require(modified == component["modified_paths"], "{} modified path identities".format(component["component"]))
    require(sorted(sets["upstream_only"]) == component["upstream_only_paths"], "{} upstream-only paths".format(component["component"]))
    return local


def validate_inventory(inventory_path):
    inventory = load_json(inventory_path)
    require(inventory.get("schema_version") == "1.0", "inventory schema version")
    require(inventory.get("record_kind") == "phase3d0_source_migration_baseline", "inventory record kind")
    baseline = inventory["baseline"]
    commit = baseline["commit"]
    actual_tree = git("rev-parse", "{}^{{tree}}".format(commit)).decode().strip()
    require(actual_tree == baseline["tree"], "baseline tree SHA")
    base_entries = tree_entries(commit)
    validate_summary(
        full_summary(base_entries, base_entries),
        {key: baseline[key] for key in ("path_count", "ordinary_blob_bytes", "sorted_path_list_sha256", "sorted_identity_list_sha256")},
        "baseline census",
    )

    top_level = {}
    for path in base_entries:
        top_level.setdefault(path.split("/", 1)[0], set()).add(path)
    actual_top = [
        {"name": name, **full_summary(base_entries, paths)}
        for name, paths in sorted(top_level.items())
    ]
    require(actual_top == inventory["top_level_census"], "top-level census")

    surface = inventory["file_surface_census"]
    actual_suffixes = []
    for expected in surface["suffixes"]:
        suffix = expected["suffix"]
        actual_suffixes.append({"suffix": suffix, **set_summary(path for path in base_entries if path.endswith(suffix))})
    require(actual_suffixes == surface["suffixes"], "file suffix census")

    main_guards = set()
    hydra_main = set()
    lfs_pointers = set()
    for path, entry in base_entries.items():
        content = blob_bytes(entry)
        if content.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
            lfs_pointers.add(path)
        if path.endswith(".py"):
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if MAIN_GUARD.search(text):
                main_guards.add(path)
            if "@hydra.main" in text:
                hydra_main.add(path)
    validate_summary(set_summary(main_guards), surface["python_main_guard_paths"], "Python main guards")
    validate_summary(set_summary(hydra_main), surface["hydra_main_paths"], "Hydra main paths")
    validate_summary(set_summary(lfs_pointers), surface["git_lfs_pointer_paths"], "Git LFS pointers")
    executable = {path for path, entry in base_entries.items() if entry["mode"] == "100755"}
    validate_summary(set_summary(executable), surface["executable_mode_paths"], "executable modes")

    comparison_locals = {}
    for component in inventory["upstream_comparisons"]:
        snapshot = validate_snapshot(ROOT / component["snapshot_path"], component)
        comparison_locals[component["component"]] = compare_to_upstream(base_entries, component, snapshot)

    selectors = {
        "ConRep": {path for path in base_entries if path.startswith("llm2vec/ContrastiveUnlearning") and path.endswith(".py")},
        "ClinicIA": {path for path in base_entries if path.startswith("llm2vec/unlearn_eval/")},
        "LLM2Vec": {"llm2vec/" + path for path in comparison_locals["LLM2Vec"]},
        "OpenUnlearning": {"llm2vec/open_unlearning/" + path for path in comparison_locals["OpenUnlearning"]},
        "legacy_datasets": {path for path in base_entries if path.startswith("llm2vec/UnlearnData/")},
    }
    for boundary in inventory["ownership_boundaries"]:
        if "path_count" in boundary:
            validate_summary(
                set_summary(selectors[boundary["component"]]),
                {key: boundary[key] for key in ("path_count", "sorted_path_list_sha256")},
                "{} ownership scope".format(boundary["component"]),
            )

    entrypoint_paths = set()
    entrypoint_ids = set()
    for entrypoint in inventory["critical_entrypoints"]:
        path = entrypoint["path"]
        require(path in base_entries, "missing critical entrypoint: {}".format(path))
        require(path not in entrypoint_paths, "duplicate critical entrypoint path")
        require(entrypoint["id"] not in entrypoint_ids, "duplicate critical entrypoint id")
        entrypoint_paths.add(path)
        entrypoint_ids.add(entrypoint["id"])
        require(base_entries[path]["blob_sha"] == entrypoint["git_blob_sha"], "critical entrypoint blob: {}".format(path))
        require(base_entries[path]["mode"] == entrypoint["mode"], "critical entrypoint mode: {}".format(path))

    historical_entrypoints = set()
    historical_index = json.loads(blob_bytes(base_entries["configs/historical/paper/index.json"]).decode("utf-8"))
    for item in historical_index["experiments"]:
        record_path = "configs/historical/paper/" + item["path"]
        record = json.loads(blob_bytes(base_entries[record_path]).decode("utf-8"))
        if record.get("historical_training_entry_point"):
            historical_entrypoints.add(record["historical_training_entry_point"])
    require(
        historical_entrypoints == {"llm2vec/ContrastiveUnlearning_Adaptive_RandomToken_LMloss_margin.py"},
        "resolved historical training entrypoints",
    )
    require(historical_entrypoints <= entrypoint_paths, "historical entrypoint is not protected as critical")

    duplicate = inventory["duplicate_packages"]
    canonical_prefix = duplicate["canonical_prefix"]
    derivative_prefix = duplicate["derivative_prefix"]
    canonical = {path[len(canonical_prefix) :]: entry for path, entry in base_entries.items() if path.startswith(canonical_prefix)}
    derivative = {path[len(derivative_prefix) :]: entry for path, entry in base_entries.items() if path.startswith(derivative_prefix)}
    common = set(canonical) & set(derivative)
    identical = {
        path
        for path in common
        if (canonical[path]["mode"], canonical[path]["blob_sha"])
        == (derivative[path]["mode"], derivative[path]["blob_sha"])
    }
    validate_summary(set_summary(canonical), duplicate["canonical"], "canonical LLM2Vec package")
    validate_summary(set_summary(derivative), duplicate["derivative"], "derivative LLM2Vec package")
    validate_summary(set_summary(common), duplicate["same_named_common"], "same-named package files")
    validate_summary(set_summary(identical), duplicate["same_named_identical"], "identical same-named package files")
    require(sorted(common - identical) == duplicate["same_named_modified_paths"], "modified same-named package files")
    require(sorted(set(canonical) - set(derivative)) == duplicate["canonical_only_paths"], "canonical-only package files")
    require(sorted(set(derivative) - set(canonical)) == duplicate["derivative_only_paths"], "derivative-only package files")
    require(duplicate["equivalence_status"] == "not_validated" and duplicate["removal_allowed"] is False, "duplicate-package safety gate")
    for implementation in duplicate["implementation_pair"]:
        require(base_entries[implementation["path"]]["blob_sha"] == implementation["git_blob_sha"], "implementation-pair blob")

    dependencies = inventory["dependency_surfaces"]
    for absent in dependencies["absent_root_paths"]:
        require(absent not in base_entries, "unexpected root environment path: {}".format(absent))
    for surface_item in dependencies["surfaces"]:
        require(base_entries[surface_item["path"]]["blob_sha"] == surface_item["git_blob_sha"], "dependency surface blob")
    require(dependencies["validated_reproduction_environment_available"] is False, "reproduction environment must not be claimed validated")

    licenses = inventory["license_status"]
    require("LICENSE" not in base_entries, "unexpected root project license")
    for license_item in licenses["third_party_licenses"]:
        entry = base_entries[license_item["path"]]
        require(entry["blob_sha"] == license_item["git_blob_sha"], "third-party license blob")
        require(hashlib.sha256(blob_bytes(entry)).hexdigest() == license_item["sha256"], "third-party license content")

    privacy = inventory["portability_and_privacy_scan"]
    absolute_pattern = re.compile(privacy["absolute_path_pattern"].encode("utf-8"), re.I)
    identity_pattern = re.compile(privacy["cluster_or_account_marker_pattern"].encode("utf-8"), re.I)
    credential_pattern = re.compile(privacy["credential_pattern"].encode("utf-8"))
    absolute_matches = {path for path, entry in base_entries.items() if absolute_pattern.search(blob_bytes(entry))}
    identity_matches = {path for path, entry in base_entries.items() if identity_pattern.search(blob_bytes(entry))}
    credential_matches = {path for path, entry in base_entries.items() if credential_pattern.search(blob_bytes(entry))}
    validate_summary(set_summary(absolute_matches), privacy["absolute_path_matches"], "absolute-path scan")
    safe_matches = set(privacy["validator_or_test_match_paths"])
    require(safe_matches <= absolute_matches, "validator/test scan exemptions")
    legacy_absolute = absolute_matches - safe_matches
    validate_summary(set_summary(legacy_absolute), privacy["legacy_absolute_path_matches"], "legacy absolute-path scan")
    validate_summary(set_summary(identity_matches), privacy["cluster_or_account_marker_matches"], "cluster/account marker scan")
    validate_summary(set_summary(legacy_absolute | identity_matches), privacy["legacy_absolute_or_identity_matches"], "combined portability scan")
    validate_summary(set_summary(credential_matches), privacy["credential_pattern_matches"], "credential scan")

    for anomaly in inventory["recorded_anomalies"]:
        require(anomaly["path"] in base_entries, "missing recorded anomaly: {}".format(anomaly["path"]))
    gaps = inventory["release_surface_gaps"]
    require(gaps["root_readme"] is ("README.md" in base_entries), "root README status")
    require(gaps["root_license"] is ("LICENSE" in base_entries), "root license status")
    root_packaging = any(path in base_entries for path in ("pyproject.toml", "requirements.txt", "environment.yml", "setup.py"))
    require(gaps["root_packaging_or_environment"] is root_packaging, "root packaging status")

    phase = inventory["phase_scope"]
    require(phase["source_movement_performed"] is False, "Phase 3D-0 must not claim source movement")
    require(phase["scientific_or_provenance_content_changed"] is False, "Phase 3D-0 scientific-change claim")
    require(len(phase["allowed_additive_paths"]) == len(set(phase["allowed_additive_paths"])), "duplicate allowed additive path")
    require(len(phase["allowed_modified_paths"]) == len(set(phase["allowed_modified_paths"])), "duplicate allowed modified path")
    require(
        not set(phase["allowed_additive_paths"]) & set(phase["allowed_modified_paths"]),
        "additive and modified path scopes overlap",
    )

    return {
        "baseline_paths": len(base_entries),
        "llm2vec_identical": inventory["upstream_comparisons"][0]["identical"]["path_count"],
        "llm2vec_modified": inventory["upstream_comparisons"][0]["modified"]["path_count"],
        "open_unlearning_identical": inventory["upstream_comparisons"][1]["identical"]["path_count"],
        "open_unlearning_modified": inventory["upstream_comparisons"][1]["modified"]["path_count"],
        "portability_files": privacy["legacy_absolute_or_identity_matches"]["path_count"],
        "credential_matches": privacy["credential_pattern_matches"]["path_count"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    args = parser.parse_args(argv)
    result = validate_inventory(args.inventory.resolve())
    print(
        "validated Phase 3D-0 source inventory: "
        "baseline_paths={baseline_paths}, "
        "LLM2Vec identical/modified={llm2vec_identical}/{llm2vec_modified}, "
        "OpenUnlearning identical/modified={open_unlearning_identical}/{open_unlearning_modified}, "
        "portability_files={portability_files}, credential_matches={credential_matches}".format(**result)
    )


if __name__ == "__main__":
    main()
