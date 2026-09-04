"""Project-owned ConRep concepts and stable entry points.

All public metadata helpers are dependency-free. Model-facing imports remain
lazy so repository inspection never initializes a model.
"""

from .corruption import (
    TOKEN_SWAP_SEPARATOR,
    TokenSwapText,
    annotate_token_swap,
    parse_token_swap,
)
from .entrypoints import DEFAULT_VARIANT, VARIANTS, run_variant, source_path
from .objectives import OBJECTIVES, SELECTED_PAPER_OBJECTIVE, ObjectiveSpec, get_objective
from .runs import PaperRun, get_run, load_runs, normalized_config

__all__ = [
    "DEFAULT_VARIANT",
    "OBJECTIVES",
    "SELECTED_PAPER_OBJECTIVE",
    "TOKEN_SWAP_SEPARATOR",
    "VARIANTS",
    "ObjectiveSpec",
    "PaperRun",
    "TokenSwapText",
    "annotate_token_swap",
    "get_objective",
    "get_run",
    "load_runs",
    "normalized_config",
    "parse_token_swap",
    "run_variant",
    "source_path",
]
