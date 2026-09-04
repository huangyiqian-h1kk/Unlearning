# Project-owned Python code

This directory contains the code that the paper contributes.

- `conrep/` names the representation-space unlearning method, its historical
  implementations, the token-swap convention, and the five selected paper runs.
- `clinicia/` names the evaluation benchmark: probe families, normalization
  formulas, dataset integrity checks, and compatibility evaluators.

The model-facing implementations under each `legacy/` directory are preserved
for provenance. New callers should start with `scripts/train_conrep.py`,
`scripts/evaluate_clinicia.py`, or `scripts/reproduce.py`.

Pinned upstream snapshots are deliberately outside this package under
`third_party/`; superseded exploratory code is under `legacy/`.
