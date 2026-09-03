"""Causal LLM2Vec adapter with shared canonical support modules.

Only the causal core and OpenUnlearning wrapper are project-specific.  The
package search path falls back to the sibling upstream ``llm2vec`` package for
support modules that were byte-identical before Phase 3D-3.
"""

from pathlib import Path as _Path


_CANONICAL_PACKAGE = str(_Path(__file__).resolve().parents[1] / "llm2vec")
if _CANONICAL_PACKAGE not in __path__:
    __path__.append(_CANONICAL_PACKAGE)

from .llm22vec import LLM2Vec

__all__ = ["LLM2Vec"]
