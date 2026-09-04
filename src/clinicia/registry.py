"""Versioned registry for the evidence-backed ClinicIA datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = REPOSITORY_ROOT / "results" / "paper" / "mcq_dataset_inventory.json"
DEFAULT_CATALOG = REPOSITORY_ROOT / "data" / "clinicia" / "catalog.json"

_ID_BY_HISTORICAL_PATH = {
    "llm2vec/unlearn_eval/data/mcqs_deaths_att.jsonl": "clinicia/a/deaths/att@historical_v1",
    "llm2vec/unlearn_eval/data/mcqs_deaths_id_eq.jsonl": "clinicia/a/deaths/id-equal@historical_v1",
    "llm2vec/unlearn_eval/data/mcqs_deaths_id_sim.jsonl": "clinicia/a/deaths/id-similar@historical_v1",
    "llm2vec/unlearn_eval/data/mcqs_diagnosis_att.jsonl": "clinicia/a/diagnosis/att@historical_v1",
    "llm2vec/unlearn_eval/data/mcqs_diagnosis_id.jsonl": "clinicia/a/diagnosis/id@historical_v1",
    "llm2vec/UnlearnData/mcqs_PMC_forget_att.jsonl": "clinicia/b/pmc/forget/att@historical_v1",
    "llm2vec/UnlearnData/mcqs_PMC_forget_id_equal.jsonl": "clinicia/b/pmc/forget/id-equal@historical_v1",
    "llm2vec/UnlearnData/mcqs_PMC_forget_id_identical.jsonl": "clinicia/b/pmc/forget/id@historical_v1",
    "llm2vec/UnlearnData/mcqs_PMC_retain_att.jsonl": "clinicia/b/pmc/retain/att@historical_v1",
    "llm2vec/UnlearnData/mcqs_PMC_retain_id_equal.jsonl": "clinicia/b/pmc/retain/id-equal@historical_v1",
    "llm2vec/UnlearnData/mcqs_PMC_retain_id_identical.jsonl": "clinicia/b/pmc/retain/id@historical_v1",
}


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    repository_path: str
    historical_path: str
    split: str
    probe: str
    record_count: int
    verification_method: str
    content_sha256: str
    lfs_backed: bool

    def path(self, root: Path = REPOSITORY_ROOT) -> Path:
        return Path(root) / self.repository_path


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_registry(
    inventory_path: Path = DEFAULT_INVENTORY,
    catalog_path: Path = DEFAULT_CATALOG,
) -> Mapping[str, DatasetSpec]:
    """Join immutable paper metadata to the current semantic data paths."""

    inventory = _load(inventory_path)
    by_historical = {row["path"]: row for row in inventory.get("datasets", [])}
    if set(by_historical) != set(_ID_BY_HISTORICAL_PATH):
        missing = sorted(set(_ID_BY_HISTORICAL_PATH) - set(by_historical))
        extra = sorted(set(by_historical) - set(_ID_BY_HISTORICAL_PATH))
        raise ValueError(
            f"ClinicIA inventory path mismatch; missing={missing}, extra={extra}"
        )

    catalog = _load(catalog_path)
    current_by_historical = {
        row["historical_path"]: row["path"] for row in catalog.get("datasets", [])
    }
    missing_catalog = sorted(set(_ID_BY_HISTORICAL_PATH) - set(current_by_historical))
    if missing_catalog:
        raise ValueError(f"ClinicIA catalog is missing historical paths: {missing_catalog}")

    registry = {}
    for historical_path, dataset_id in _ID_BY_HISTORICAL_PATH.items():
        row = by_historical[historical_path]
        digest = row.get("expected_content_sha256", row.get("file_sha256"))
        if not digest:
            raise ValueError(
                f"ClinicIA dataset lacks a verified content digest: {historical_path}"
            )
        registry[dataset_id] = DatasetSpec(
            dataset_id=dataset_id,
            repository_path=current_by_historical[historical_path],
            historical_path=historical_path,
            split=row["split"],
            probe=row["probe"],
            record_count=int(row["record_count"]),
            verification_method=row["verification_method"],
            content_sha256=digest,
            lfs_backed="lfs_content_oid" in row,
        )
    if len(registry) != len(_ID_BY_HISTORICAL_PATH):
        raise ValueError("ClinicIA dataset IDs must be unique")
    return MappingProxyType(registry)


def dataset_ids() -> tuple[str, ...]:
    return tuple(sorted(load_registry()))


def get_dataset(dataset_id: str) -> DatasetSpec:
    registry = load_registry()
    try:
        return registry[dataset_id]
    except KeyError as exc:
        choices = ", ".join(registry)
        raise ValueError(
            f"unknown ClinicIA dataset {dataset_id!r}; choose one of: {choices}"
        ) from exc
