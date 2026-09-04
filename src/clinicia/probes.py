"""Canonical names for the diversified ClinicIA probe families."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Probe:
    probe_id: str
    family: str
    paper_label: str
    question: str


PROBES = {
    "qa": Probe("qa", "generation", "QA", "Can the model answer a direct question?"),
    "cloze": Probe("cloze", "generation", "Cloze", "Can it complete the attribute in context?"),
    "background": Probe(
        "background",
        "generation",
        "BG",
        "Can it reveal the attribute through background knowledge?",
    ),
    "attribute": Probe(
        "attribute",
        "multiple_choice",
        "ATT",
        "Can it select the attribute associated with an identifier?",
    ),
    "identifier-equal": Probe(
        "identifier-equal",
        "multiple_choice",
        "IDeq",
        "Can it recover the exact identifier from the attribute?",
    ),
    "identifier-related": Probe(
        "identifier-related",
        "multiple_choice",
        "ID",
        "Can it distinguish the intended identifier from related alternatives?",
    ),
}

PAPER_ORDER = ("qa", "cloze", "background", "attribute", "identifier-equal", "identifier-related")


def get_probe(probe_id: str) -> Probe:
    try:
        return PROBES[probe_id]
    except KeyError as exc:
        choices = ", ".join(PAPER_ORDER)
        raise ValueError(f"unknown ClinicIA probe {probe_id!r}; choose one of: {choices}") from exc
