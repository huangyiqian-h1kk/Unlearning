#!/usr/bin/env python3
"""Validate evidence-backed historical experiment records using only stdlib."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
from collections.abc import Callable


ROOT = pathlib.Path(__file__).resolve().parents[2]
STATUS = {
    "verified",
    "label_corrected",
    "non_comparable",
    "unresolved",
    "missing",
    "manuscript_transcription_error",
}
REPRODUCTION_STATUS = {
    "historical_resolved",
    "historical_partial",
    "evaluation_only",
    "not_runnable",
    "unresolved",
}
PRIVATE_PATH = re.compile(
    r"(/home/|/scratch/|/gpfs/|/lustre/|/tmp/|/var/tmp/|mktemp)",
    re.IGNORECASE,
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def fail(message: str) -> None:
    raise ValueError(message)


def load_json(path: pathlib.Path) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def repository_relative_path(raw: str, *, label: str) -> pathlib.PurePosixPath:
    path = pathlib.PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts or str(path) != raw:
        fail(f"unsafe {label} path: {raw}")
    return path


def validate_legacy_source(
    root: pathlib.Path,
    source: dict,
    runner: Callable = subprocess.run,
) -> str:
    """Verify a selected source against history and its current materialization."""

    historical = source["path"]
    repository_relative_path(historical, label="inventory")
    digest = source["sha256"]
    if not SHA256.fullmatch(digest):
        fail("invalid inventory SHA-256")

    environment = os.environ.copy()
    environment["GIT_NO_LAZY_FETCH"] = "1"
    result = runner(
        [
            "git",
            "ls-tree",
            "--full-tree",
            source["starting_git_commit"],
            "--",
            historical,
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    entries = [line for line in result.stdout.splitlines() if line]
    if len(entries) != 1:
        fail(f"expected one tree entry for {historical}, found {len(entries)}")
    metadata, separator, entry_path = entries[0].partition("\t")
    fields = metadata.split()
    if not separator or len(fields) != 3 or entry_path != historical:
        fail("malformed tree entry: " + historical)
    _mode, object_type, object_id = fields
    if object_type != "blob":
        fail("tree entry is not a blob: " + historical)
    if object_id != source["git_blob_object_id"]:
        fail("blob mismatch: " + historical)

    materialized = source.get("materialized_path", historical)
    relative = repository_relative_path(materialized, label="materialized")
    worktree_path = pathlib.Path(root).joinpath(*relative.parts)
    if worktree_path.is_symlink():
        fail("materialized path is not a regular file: " + materialized)
    if not worktree_path.exists():
        return "not_materialized"
    if not worktree_path.is_file():
        fail("materialized path is not a regular file: " + materialized)
    if hashlib.sha256(worktree_path.read_bytes()).hexdigest() != digest:
        fail("content hash mismatch: " + materialized)
    return "verified"


def validate(
    index_path: pathlib.Path,
    manifest_path: pathlib.Path,
) -> tuple[list[dict], list[str]]:
    index_path = pathlib.Path(index_path)
    manifest_path = pathlib.Path(manifest_path)
    index = load_json(index_path)
    manifest = load_json(manifest_path)
    entries = index["experiments"]
    if len(entries) != 25:
        fail("exactly 25 records required")

    experiment_ids = [item["experiment_id"] for item in entries]
    manifest_ids = [item["id"] for item in manifest["experiments"]]
    if experiment_ids != sorted(experiment_ids):
        fail("index ordering is not deterministic")
    if set(experiment_ids) != set(manifest_ids):
        fail("record IDs do not exactly match manifest")
    manifest_by_id = {item["id"]: item for item in manifest["experiments"]}
    required = set(load_json(ROOT / "configs" / "historical" / "schema.json")["required"])

    records = []
    for entry in entries:
        path = index_path.parent / entry["path"]
        raw = path.read_bytes()
        record = json.loads(raw)
        expected = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()
        if raw != expected:
            fail(f"nondeterministic serialization: {path}")
        missing = required - set(record)
        if missing:
            fail(f"{record.get('experiment_id')}: missing {sorted(missing)}")

        manifest_record = manifest_by_id[record["experiment_id"]]
        for record_key, manifest_key in (
            ("regime", "regime"),
            ("knowledge_target", "knowledge_target"),
            ("method", "method"),
            ("resolved_model_id", "resolved_model_id"),
            ("status_flags", "status_flags"),
        ):
            if record[record_key] != manifest_record.get(manifest_key):
                fail(
                    f"{record['experiment_id']}: manifest disagreement: {record_key}"
                )
        if not set(record["status_flags"]) <= STATUS:
            fail("invalid status vocabulary")
        if record["reproduction_status"] not in REPRODUCTION_STATUS:
            fail("invalid reproduction vocabulary")
        if PRIVATE_PATH.search(json.dumps(record)):
            fail(f"{record['experiment_id']}: private or temporary path")
        if record["identity_evidence_basis"].lower().startswith("filename"):
            fail("filename-only identity")
        for evidence in record["archive_evidence_references"]:
            for key in ("sha256", "archive_sha256", "member_sha256"):
                if key in evidence and not SHA256.fullmatch(evidence[key]):
                    fail("invalid evidence SHA-256")
        records.append(record)

    inventory = load_json(ROOT / "results" / "repository_selected_legacy_sources.json")
    if PRIVATE_PATH.search(json.dumps(inventory)):
        fail("private path in inventory")
    sources = inventory["sources"]
    if len(sources) != 16:
        fail("exactly 16 selected historical files are required")
    content_checks = [validate_legacy_source(ROOT, source) for source in sources]

    conrep = [record for record in records if record["method"] == "ConRep"]
    if len(conrep) != 5:
        fail("exactly five selected ConRep records are required")
    if any(record["resolved_hyperparameters"].get("lm_weight") != 0 for record in conrep):
        fail("ConRep lm_weight invariant")
    expected_entrypoint = (
        "llm2vec/ContrastiveUnlearning_Adaptive_RandomToken_LMloss_margin.py"
    )
    if any(
        record["historical_training_entry_point"] != expected_entrypoint
        for record in conrep
    ):
        fail("ConRep entry point invariant")

    by_id = {record["experiment_id"]: record for record in records}
    if "non_comparable" not in by_id["b-pmc-mistral-graddiff"]["status_flags"]:
        fail("PMC GradDiff comparability")
    for experiment_id in ("a-deaths-llama2-rmu", "a-deaths-mistral-rmu"):
        if "unresolved" not in by_id[experiment_id]["status_flags"]:
            fail("Deaths RMU must be unresolved")
    pmc = by_id["b-pmc-mistral-conrep"]
    mmlu_text = json.dumps(pmc["mmlu_evidence"])
    if (
        "0.2659" in mmlu_text
        or pmc["mmlu_evidence"]["settings_status"] not in {"missing", "unresolved"}
    ):
        fail("PMC ConRep MMLU contamination")
    for record in records:
        if (
            record["unresolved_fields"]
            and record["reproduction_status"] == "historical_resolved"
        ):
            fail("unresolved fact rendered runnable")
    return records, content_checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=pathlib.Path)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        records, checks = validate(args.index, args.manifest)
    except (
        ValueError,
        KeyError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"validated {len(records)} historical experiment records; "
        f"legacy content checks: verified={checks.count('verified')}, "
        f"not_materialized={checks.count('not_materialized')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
