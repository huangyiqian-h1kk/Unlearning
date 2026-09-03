"""Integrity-checking adapters for versioned ClinicIA JSONL datasets."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Mapping

from .registry import REPOSITORY_ROOT, DatasetSpec, get_dataset


_LFS_POINTER = re.compile(
    rb"\Aversion https://git-lfs\.github\.com/spec/v1\n"
    rb"oid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n?\Z"
)


class DatasetIntegrityError(ValueError):
    """Raised when materialized bytes do not match the protected inventory."""


class DatasetNotMaterializedError(FileNotFoundError):
    """Raised when only a verified Git LFS pointer is available locally."""


def _read_records(spec: DatasetSpec, path: Path) -> tuple[Mapping[str, object], ...]:
    data = path.read_bytes()
    pointer = _LFS_POINTER.fullmatch(data)
    if pointer:
        oid = pointer.group(1).decode("ascii")
        if not spec.lfs_backed or oid != spec.content_sha256:
            raise DatasetIntegrityError(
                f"unexpected Git LFS pointer for {spec.dataset_id}: sha256:{oid}"
            )
        raise DatasetNotMaterializedError(
            f"{spec.dataset_id} is represented by verified Git LFS object sha256:{oid}; "
            "materialize it explicitly before evaluation"
        )

    digest = hashlib.sha256(data).hexdigest()
    if digest != spec.content_sha256:
        raise DatasetIntegrityError(
            f"content hash mismatch for {spec.dataset_id}: expected "
            f"{spec.content_sha256}, got {digest}"
        )

    records = []
    for line_number, raw_line in enumerate(data.splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise DatasetIntegrityError(
                f"invalid JSONL for {spec.dataset_id} at line {line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise DatasetIntegrityError(
                f"ClinicIA record for {spec.dataset_id} at line {line_number} is not an object"
            )
        records.append(record)

    if len(records) != spec.record_count:
        raise DatasetIntegrityError(
            f"record-count mismatch for {spec.dataset_id}: expected "
            f"{spec.record_count}, got {len(records)}"
        )
    return tuple(records)


def load_records(
    dataset_id: str,
    *,
    root: Path = REPOSITORY_ROOT,
) -> tuple[Mapping[str, object], ...]:
    """Load one registered dataset after verifying its exact bytes and count.

    The adapter never downloads an LFS object. A caller must deliberately
    materialize verified data before invoking an evaluation environment.
    """

    spec = get_dataset(dataset_id)
    return _read_records(spec, Path(root) / spec.repository_path)
