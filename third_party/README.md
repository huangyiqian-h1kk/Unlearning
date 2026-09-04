# Vendored dependency snapshots

These directories are upstream-derived historical working snapshots retained
so ConRep and baseline environments remain inspectable:

- `llm2vec/`: compared with McGill-NLP/llm2vec at the inferred revision.
- `open-unlearning/`: compared with locuslab/open-unlearning at the inferred
  revision; it includes documented project-era integrations.

They keep their upstream licenses and dependency constraints, but neither
directory should be described as a pristine upstream checkout. Project-facing
ConRep/ClinicIA code lives in `src/`; do not add new project logic here. See
[`THIRD_PARTY.md`](../THIRD_PARTY.md) for revisions and exact local-delta notes.
