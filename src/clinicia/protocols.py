"""Versioned ClinicIA protocol boundaries.

These records distinguish archived paper behavior from future corrected
evaluation.  They intentionally make no claim that a validated-v2 model
environment is available yet.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationProtocol:
    protocol_id: str
    status: str
    result_namespace: str
    runnable: bool
    may_modify_archived_paper_results: bool
    description: str


HISTORICAL_V1 = EvaluationProtocol(
    protocol_id="historical_v1",
    status="archived_compatibility",
    result_namespace="results/paper",
    runnable=False,
    may_modify_archived_paper_results=False,
    description="Preserved paper-era behavior and evidence; incomplete environments stay explicit.",
)

VALIDATED_V2 = EvaluationProtocol(
    protocol_id="validated_v2",
    status="contract_defined_not_yet_runnable",
    result_namespace="results/validated_v2",
    runnable=False,
    may_modify_archived_paper_results=False,
    description="Future shared ConRep/baseline protocol; results must remain separate from paper evidence.",
)

PROTOCOLS = {
    HISTORICAL_V1.protocol_id: HISTORICAL_V1,
    VALIDATED_V2.protocol_id: VALIDATED_V2,
}


def get_protocol(protocol_id: str) -> EvaluationProtocol:
    try:
        return PROTOCOLS[protocol_id]
    except KeyError as exc:
        choices = ", ".join(sorted(PROTOCOLS))
        raise ValueError(f"unknown ClinicIA protocol {protocol_id!r}; choose one of: {choices}") from exc
