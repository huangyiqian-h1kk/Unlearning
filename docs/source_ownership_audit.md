# Source ownership

## Current boundary

- **Project-owned:** `src/conrep/`, `src/clinicia/`, `scripts/`, paper-run
  indexes/wrappers, and project documentation.
- **Vendored, upstream-derived dependencies:** `third_party/llm2vec/` and
  `third_party/open-unlearning/`, retaining upstream licenses and documented
  historical local deltas; they are not pristine upstream checkouts.
- **Historical project work:** `legacy/`, retained for traceability but not
  presented as the reproduction API.
- **Scientific evidence:** `configs/paper/historical/` and `results/paper/`;
  these preserve historical identities even when paths are obsolete.

The causal `src/conrep/backends/llm22vec/` adapter is project-specific. Its
support-module fallback points to the pinned LLM2Vec package because the shared
modules were previously verified byte-identical.

This document supersedes earlier directory-based ownership assumptions. The
machine-readable Phase 3D inventories remain historical audit records rather
than the current navigation map.
