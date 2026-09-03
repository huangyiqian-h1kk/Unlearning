"""Project-owned ClinicIA registry, protocols, and compatibility entry points."""

from .adapters import DatasetIntegrityError, DatasetNotMaterializedError, load_records
from .protocols import HISTORICAL_V1, PROTOCOLS, VALIDATED_V2, get_protocol
from .registry import DatasetSpec, dataset_ids, get_dataset, load_registry

__all__ = [
    "DatasetSpec",
    "DatasetIntegrityError",
    "DatasetNotMaterializedError",
    "HISTORICAL_V1",
    "PROTOCOLS",
    "VALIDATED_V2",
    "dataset_ids",
    "get_dataset",
    "get_protocol",
    "load_records",
    "load_registry",
]
