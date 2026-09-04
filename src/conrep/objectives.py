"""Names and paper anchors for the ConRep objective family.

This module is intentionally descriptive. The executable historical
implementations remain in :mod:`conrep.legacy`; recording the equation groups
here makes the paper-to-code relationship inspectable without importing
PyTorch or a model stack.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveSpec:
    """One objective family described in the paper."""

    objective_id: str
    paper_equations: tuple[int, ...]
    purpose: str
    historical_variant: str | None


OBJECTIVES = {
    "general": ObjectiveSpec(
        "general",
        (5, 6),
        "General representation-space forget/retain objective.",
        None,
    ),
    "token-swap": ObjectiveSpec(
        "token-swap",
        (7,),
        "Construct the corrupted representation target used by ConRep.",
        "adaptive-random-token-lmloss",
    ),
    "combined": ObjectiveSpec(
        "combined",
        (8, 9, 10),
        "Combine forgetting, retention, and optional language-model terms.",
        "adaptive-random-token-lmloss-margin",
    ),
    "symmetric": ObjectiveSpec(
        "symmetric",
        (11, 12, 13),
        "Symmetric specialization used by the selected paper runs.",
        "adaptive-random-token-lmloss-margin",
    ),
}

SELECTED_PAPER_OBJECTIVE = "symmetric"


def get_objective(objective_id: str) -> ObjectiveSpec:
    """Return an objective description without loading model dependencies."""

    try:
        return OBJECTIVES[objective_id]
    except KeyError as exc:
        choices = ", ".join(sorted(OBJECTIVES))
        raise ValueError(
            f"unknown ConRep objective {objective_id!r}; choose one of: {choices}"
        ) from exc
