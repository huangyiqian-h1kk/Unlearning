"""Stable dispatch for preserved ClinicIA evaluation programs."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LEGACY_SOURCE_ROOT = REPOSITORY_ROOT / "src" / "clinicia" / "legacy"

ENTRYPOINTS = {
    "main": "main_eval.py",
    "generation": "batchedEval.py",
    "likelihood": "batchedEval_loglikelihood.py",
    "pmc": "EvalPMC.py",
}

DEFAULT_ENTRYPOINT = "main"


def source_path(entrypoint: str = DEFAULT_ENTRYPOINT) -> Path:
    """Return the canonical source file for a legacy evaluator."""

    try:
        filename = ENTRYPOINTS[entrypoint]
    except KeyError as exc:
        choices = ", ".join(sorted(ENTRYPOINTS))
        raise ValueError(f"unknown ClinicIA entry point {entrypoint!r}; choose one of: {choices}") from exc
    path = LEGACY_SOURCE_ROOT / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def run_entrypoint(entrypoint: str = DEFAULT_ENTRYPOINT, argv: Iterable[str] | None = None) -> None:
    """Execute a preserved evaluator without importing model dependencies early."""

    source = source_path(entrypoint)
    old_argv = sys.argv
    inserted = str(LEGACY_SOURCE_ROOT)
    sys.argv = [str(source), *(list(argv) if argv is not None else old_argv[1:])]
    sys.path.insert(0, inserted)
    try:
        runpy.run_path(str(source), run_name="__main__")
    finally:
        sys.argv = old_argv
        if sys.path and sys.path[0] == inserted:
            sys.path.pop(0)
        else:
            sys.path.remove(inserted)
