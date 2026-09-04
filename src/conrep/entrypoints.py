"""Stable dispatch for the preserved ConRep training implementations."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LEGACY_SOURCE_ROOT = REPOSITORY_ROOT / "src" / "conrep" / "legacy"
PROJECT_BACKEND_ROOT = REPOSITORY_ROOT / "src" / "conrep" / "backends"
LEGACY_LLM2VEC_ROOT = REPOSITORY_ROOT / "third_party" / "llm2vec"

VARIANTS = {
    "base": "ContrastiveUnlearning.py",
    "adaptive-random-token-lmloss": "ContrastiveUnlearning_Adaptive_RandomToken_LMloss.py",
    "adaptive-random-token-lmloss-epoch-eval": "ContrastiveUnlearning_Adaptive_RandomToken_LMloss_EpochEval.py",
    "adaptive-random-token-lmloss-check": "ContrastiveUnlearning_Adaptive_RandomToken_LMloss_check.py",
    "adaptive-random-token-lmloss-eval": "ContrastiveUnlearning_Adaptive_RandomToken_LMloss_eval.py",
    "adaptive-random-token-lmloss-margin": "ContrastiveUnlearning_Adaptive_RandomToken_LMloss_margin.py",
    "adaptive-random-token-lmloss-margin-progressive": "ContrastiveUnlearning_Adaptive_RandomToken_LMloss_margin_progressive.py",
    "adaptive-random-token-lmloss-only": "ContrastiveUnlearning_Adaptive_RandomToken_LMloss_only.py",
    "adaptive-random-token-lmloss-zero": "ContrastiveUnlearning_Adaptive_RandomToken_LMloss_zero.py",
    "l2-distance-random-token-lmloss": "ContrastiveUnlearning_L2Distance_RandomToken_LMloss.py",
    "random-token-lmloss": "ContrastiveUnlearning_RandomToken_LMloss.py",
    "random-token-zero-lmloss": "ContrastiveUnlearning_RandomToken_zeroLMloss.py",
}

DEFAULT_VARIANT = "adaptive-random-token-lmloss-margin"


def source_path(variant: str = DEFAULT_VARIANT) -> Path:
    """Return the canonical source file for a named historical variant."""

    try:
        filename = VARIANTS[variant]
    except KeyError as exc:
        choices = ", ".join(sorted(VARIANTS))
        raise ValueError(f"unknown ConRep variant {variant!r}; choose one of: {choices}") from exc
    path = LEGACY_SOURCE_ROOT / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def run_variant(variant: str = DEFAULT_VARIANT, argv: Iterable[str] | None = None) -> None:
    """Execute one preserved implementation with its historical CLI contract."""

    source = source_path(variant)
    old_argv = sys.argv
    inserted = [str(PROJECT_BACKEND_ROOT), str(LEGACY_LLM2VEC_ROOT)]
    sys.argv = [str(source), *(list(argv) if argv is not None else old_argv[1:])]
    sys.path[0:0] = inserted
    try:
        runpy.run_path(str(source), run_name="__main__")
    finally:
        sys.argv = old_argv
        for path in reversed(inserted):
            if path in sys.path:
                sys.path.remove(path)
