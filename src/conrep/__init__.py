"""Project-owned ConRep entry points.

Model-facing imports remain lazy so repository validation never initializes a
model or requires the historical training environment.
"""

from .entrypoints import DEFAULT_VARIANT, VARIANTS, run_variant, source_path

__all__ = ["DEFAULT_VARIANT", "VARIANTS", "run_variant", "source_path"]
