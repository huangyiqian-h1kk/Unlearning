# ConRep and ClinicIA

Representation-space unlearning and diversified evaluation for
identifier–attribute knowledge in language models.

## Artifact status

This repository preserves the scientific evidence used to audit the ConRep and
ClinicIA paper artifact. The archived tables can be rebuilt without loading a
model. No experiment was rerun during repository reorganization, and the
repository does **not** yet claim complete end-to-end reproduction.

Historical records may be incomplete or non-runnable by design. Future
measurements use the separate `validated_v2` namespace and must not overwrite
the archived `historical_v1` evidence.

## Research overview

Identifier–attribute knowledge associates an identifier, an attribute type,
and a value. Fixed textual probes can underestimate extraction risk when the
same meaning has diverse expressions. **ConRep** is the project-owned
representation-space unlearning method. **ClinicIA** is the project-owned
evaluation layer for QA, cloze, background, and identifier/attribute MCQ
expressions.

The archived study compares no-unlearning baselines, GradDiff, NPO, RMU, and
ConRep across celebrity diagnosis, celebrity death, and injected PMC settings.
Known provenance gaps, non-comparable checkpoints, missing measurements, and
manuscript transcription discrepancies remain explicit in the evidence layer.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/conrep/` | Stable dispatch and blob-preserved project-owned ConRep implementations. |
| `src/clinicia/` | ClinicIA registry, integrity adapters, protocol boundary, and preserved evaluators. |
| `llm2vec/llm2vec/` | Pinned LLM2Vec-derived source with recorded local modifications. |
| `llm2vec/llm22vec/` | Project-specific causal adapter; byte-identical support modules are shared with `llm2vec`. |
| `llm2vec/open_unlearning/` | Pinned OpenUnlearning-derived source with recorded local modifications. |
| `configs/historical/` | Immutable descriptions of evidence-backed past runs. |
| `configs/components/` | Portable model, method, dataset, and protocol references. |
| `configs/reproduction/` | Structurally validated future-run candidates, explicitly marked non-runnable until their gates close. |
| `configs/sweeps/` | Compact review matrices; expanded jobs and results are not tracked. |
| `results/paper/` | Protected manifest, compact metrics, and reconstructed paper tables. |
| `data/` | Dataset/LFS release inventory and unresolved redistribution decisions. |
| `docs/` | Provenance, architecture, ownership, migration, dependency, and release records. |

## Rebuild the archived tables

Only Python's standard library is required for these two commands:

```bash
python scripts/results/extract_archived_metrics.py \
  --manifest results/paper/manifest.json \
  --output-dir results/paper/raw_metrics

python scripts/results/build_tables.py \
  --manifest results/paper/manifest.json \
  --output-dir results/paper/reconstructed
```

These commands reconstruct tables from archived evidence. They do not train or
evaluate a model. See [`results/paper/README.md`](results/paper/README.md) and
[`docs/experiment_provenance.md`](docs/experiment_provenance.md).

## Validate the repository contracts

```bash
python scripts/configs/validate_historical_experiments.py \
  --index configs/historical/paper/index.json \
  --manifest results/paper/manifest.json

python scripts/configs/validate_reproduction_configs.py \
  --index configs/reproduction/index.json

python scripts/repository/validate_release_inventory.py \
  --manifest data/lfs_manifest.json \
  --dependencies docs/dependency_matrix.json

python -m unittest discover -s tests -p 'test_*.py'
```

All are lightweight, offline checks. They do not download dependencies or LFS
objects, initialize a model, train, infer, or submit scheduler work.

## Model-facing entry points

The stable launchers can enumerate preserved programs without importing model
dependencies:

```bash
python scripts/train_conrep.py --list
python scripts/evaluate_clinicia.py --list
```

Actual execution remains gated on a separately prepared model environment,
available datasets/checkpoints, and researcher validation. LLM2Vec and
OpenUnlearning currently require incompatible Transformers versions, so their
model stacks are intentionally not combined. See
[`docs/dependencies.md`](docs/dependencies.md) and
[`environments/README.md`](environments/README.md).

## Data availability and rights

Git LFS pointer identities are anchored to an immutable base tree by
[`data/lfs_manifest.json`](data/lfs_manifest.json). A pointer proves object
identity, not local availability or redistribution permission. Dataset and
model terms must be reviewed before public redistribution; unresolved entries
must not be treated as licensed for redistribution. See
[`data/README.md`](data/README.md).

## Third-party code and project license

LLM2Vec and OpenUnlearning are attributed at pinned inferred revisions and
retain their upstream MIT license files. Local modifications are documented in
[`THIRD_PARTY.md`](THIRD_PARTY.md) and
[`docs/source_ownership_audit.md`](docs/source_ownership_audit.md).

Those upstream licenses do not select a license for project-owned ConRep and
ClinicIA code. The repository has no root project license; that remains a
researcher release decision. Do not infer permission beyond the licenses and
terms attached to each component or dataset.

## Citation

The manuscript identity and public citation have not been supplied. No author
names, venue, DOI, arXiv identifier, or `CITATION.cff` is fabricated here. See
[`docs/manuscript/README.md`](docs/manuscript/README.md) for the release gate.
