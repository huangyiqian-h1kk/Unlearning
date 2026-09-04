#!/usr/bin/env python3
"""Validate path-migrated LFS, dependency, upstream, and license records offline."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import re
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]
LFS_HEADER = "version https://git-lfs.github.com/spec/v1"
LFS_OID = re.compile(r"oid sha256:([0-9a-f]{64})")
LFS_SIZE = re.compile(r"size ([0-9]+)")


def git(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout


def load_json(path: pathlib.Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load JSON {path}: {exc}") from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def index_entries() -> dict[str, str]:
    entries = {}
    for record in git("ls-files", "-s", "-z").split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        _mode, blob, stage = metadata.split()
        require(stage == b"0", f"non-zero index stage: {raw_path!r}")
        entries[raw_path.decode("utf-8")] = blob.decode("ascii")
    return entries


def tree_entries(ref: str) -> dict[str, str]:
    entries = {}
    for record in git("ls-tree", "-rz", ref).split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        _mode, kind, blob = metadata.split()
        if kind == b"blob":
            entries[raw_path.decode("utf-8")] = blob.decode("ascii")
    return entries


def blob_bytes(blob: str) -> bytes:
    return git("cat-file", "blob", blob)


def parse_pointer(content: bytes, path: str) -> tuple[str, int]:
    try:
        text = content.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"LFS pointer is not ASCII: {path}") from exc
    lines = text.splitlines()
    require(len(lines) == 3 and lines[0] == LFS_HEADER, f"invalid LFS pointer structure: {path}")
    oid_match = LFS_OID.fullmatch(lines[1])
    size_match = LFS_SIZE.fullmatch(lines[2])
    require(oid_match is not None and size_match is not None, f"invalid LFS pointer fields: {path}")
    return oid_match.group(1), int(size_match.group(1))


def setup_call(path: str, blob: str) -> dict[str, ast.AST]:
    text = blob_bytes(blob).decode("utf-8")
    tree = ast.parse(text, filename=path)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setup"
    ]
    require(len(calls) == 1, f"expected one setup() call: {path}")
    return {keyword.arg: keyword.value for keyword in calls[0].keywords if keyword.arg}


def literal_keyword(keywords: dict[str, ast.AST], name: str, path: str):
    require(name in keywords, f"setup keyword {name!r} missing: {path}")
    try:
        return ast.literal_eval(keywords[name])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"setup keyword {name!r} is not literal: {path}") from exc


def pointer_entries(entries: dict[str, str], scopes: tuple[str, ...]) -> dict[str, str]:
    pointers = {}
    for path, blob in entries.items():
        if path.startswith(scopes):
            content = blob_bytes(blob)
            if content.startswith((LFS_HEADER + "\n").encode("ascii")):
                parse_pointer(content, path)
                pointers[path] = blob
    return pointers


def validate_lfs(manifest: dict, entries: dict[str, str]) -> int:
    base = manifest.get("phase_base", {})
    actual_tree = git("rev-parse", base.get("commit", "") + "^{tree}").decode().strip()
    require(actual_tree == base.get("tree"), "LFS manifest base tree mismatch")
    migrations = manifest.get("path_migrations")
    require(isinstance(migrations, list), "LFS path_migrations must be a list")
    require(len(migrations) == manifest.get("pointer_count"), "LFS pointer count mismatch")
    historical_paths = [row.get("historical_path") for row in migrations]
    current_paths = [row.get("path") for row in migrations]
    require(len(set(historical_paths)) == len(migrations), "duplicate historical LFS path")
    require(len(set(current_paths)) == len(migrations), "duplicate current LFS path")

    base_entries = tree_entries(base["commit"])
    for row in migrations:
        historical = row["historical_path"]
        current = row["path"]
        blob = row["pointer_blob_oid"]
        require(base_entries.get(historical) == blob, f"base LFS blob mismatch: {historical}")
        require(entries.get(current) == blob, f"current LFS blob mismatch: {current}")
        oid, size = parse_pointer(blob_bytes(blob), current)
        require(oid == row["lfs_content_oid"], f"LFS content OID mismatch: {current}")
        require(size == row["lfs_content_bytes"], f"LFS content size mismatch: {current}")

    current_pointers = pointer_entries(entries, ("data/",))
    require(set(current_pointers) == set(current_paths), "unrecorded or missing current data pointer")
    require(
        manifest.get("identity_policy")
        == "prove_each_current_pointer_is_byte_identical_to_its_base-tree_historical_path",
        "LFS identity policy mismatch",
    )
    require(manifest.get("object_availability_status") == "not_verified_by_phase3dr", "LFS availability is overclaimed")
    require(manifest.get("default_redistribution_status") == "unresolved_do_not_redistribute", "LFS rights are overclaimed")
    paper_inventory = manifest.get("paper_mcq_inventory")
    require(paper_inventory == "results/paper/mcq_dataset_inventory.json", "paper MCQ inventory reference mismatch")
    require((ROOT / paper_inventory).is_file(), "paper MCQ inventory is missing")
    return len(current_pointers)


def validate_dependencies(matrix: dict, entries: dict[str, str]) -> int:
    base = matrix.get("phase_base", {})
    actual_tree = git("rev-parse", base.get("commit", "") + "^{tree}").decode().strip()
    require(actual_tree == base.get("tree"), "dependency matrix base tree mismatch")
    components = matrix.get("components", {})
    require(set(components) == {"llm2vec", "open_unlearning"}, "unexpected dependency components")

    for name, record in components.items():
        for path_key, blob_key in (("packaging_path", "packaging_blob"), ("license_path", "license_blob")):
            path = record[path_key]
            require(entries.get(path) == record[blob_key], f"{name} {path_key} blob mismatch")
        license_content = blob_bytes(record["license_blob"])
        require(hashlib.sha256(license_content).hexdigest() == record["license_sha256"], f"{name} license hash mismatch")
        snapshot_path = ROOT / "docs" / "upstream_snapshots" / (
            "llm2vec-312adcf.json" if name == "llm2vec" else "open-unlearning-d33c476.json"
        )
        snapshot = load_json(snapshot_path)
        require(snapshot.get("commit") == record["inferred_upstream_commit"], f"{name} upstream commit mismatch")
        require(snapshot.get("tree") == record["inferred_upstream_tree"], f"{name} upstream tree mismatch")

    llm = components["llm2vec"]
    llm_setup = setup_call(llm["packaging_path"], llm["packaging_blob"])
    require(literal_keyword(llm_setup, "python_requires", llm["packaging_path"]) == llm["python_requires"], "LLM2Vec Python constraint mismatch")
    require(literal_keyword(llm_setup, "install_requires", llm["packaging_path"]) == llm["requirements"], "LLM2Vec requirements mismatch")
    require(literal_keyword(llm_setup, "extras_require", llm["packaging_path"]) == llm["optional_requirements"], "LLM2Vec extras mismatch")

    opened = components["open_unlearning"]
    require(entries.get(opened["requirements_path"]) == opened["requirements_blob"], "OpenUnlearning requirements blob mismatch")
    requirement_lines = [
        line.strip()
        for line in blob_bytes(opened["requirements_blob"]).decode("utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    require(requirement_lines == opened["requirements"], "OpenUnlearning requirements mismatch")
    open_setup = setup_call(opened["packaging_path"], opened["packaging_blob"])
    require(literal_keyword(open_setup, "python_requires", opened["packaging_path"]) == opened["python_requires"], "OpenUnlearning Python constraint mismatch")
    require(literal_keyword(open_setup, "extras_require", opened["packaging_path"]) == opened["optional_requirements"], "OpenUnlearning extras mismatch")

    conflicts = matrix.get("conflicts")
    require(isinstance(conflicts, list) and len(conflicts) == 1, "exactly one dependency conflict is expected")
    conflict = conflicts[0]
    require(
        conflict == {
            "dependency": "transformers",
            "llm2vec_constraint": ">=4.43.1,<=4.44.2",
            "open_unlearning_constraint": "==4.45.1",
            "intersection": "empty",
            "resolution": "keep_model_environments_separate",
        },
        "Transformers conflict record mismatch",
    )
    policy = matrix.get("release_policy", {})
    require(policy and all(value is False for value in policy.values()), "dependency release policy overclaims validation")
    require("LICENSE" not in entries, "root project license must remain an explicit researcher decision")
    return len(components)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--dependencies", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    try:
        entries = index_entries()
        pointer_count = validate_lfs(load_json(args.manifest.resolve()), entries)
        component_count = validate_dependencies(load_json(args.dependencies.resolve()), entries)
    except (ValueError, subprocess.CalledProcessError) as exc:
        print(f"release inventory validation failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"validated {pointer_count} path-migrated, base-tree-anchored LFS pointers, "
        f"{component_count} dependency components, 2 upstream pins, and 2 license identities"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
