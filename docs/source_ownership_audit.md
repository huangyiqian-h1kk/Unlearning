# Phase 3D-0 source ownership and migration audit

Phase 3D-0 establishes a reviewable baseline for later repository moves. It
does not move source, consolidate dependencies, delete either LLM2Vec package,
rewrite historical configurations, or change scientific evidence. Exact
machine-readable invariants live in
[`repository_source_inventory.json`](repository_source_inventory.json); the
offline validator is `scripts/repository/validate_source_inventory.py`.

The batch adds six audit/validation files and updates one existing cleanup test
so its Phase 3C-1 assertion reads the immutable Phase 3C-1 completion tree
rather than treating every future tracked addition as a cleanup regression.

## Baseline and interpretation

The audit is anchored to `main` commit
`9e843af06e9f5dcf6e69c14e6500ca0c812c84fc` and tree
`db501bce7d205a3de9f3ef75a1fe6855aadf0d08`, immediately after Phase 3C-1.
That tree contains 632 tracked paths and 5,998,289 ordinary Git blob bytes.
All comparisons in this document use Git tree and blob identities rather than
the materialized working tree.

The upstream comparisons are content-tree inferences, not proof of Git
ancestry. The checked-in snapshots make the comparison repeatable offline and
do not replace the upstream licenses. Counts in older architecture notes may
describe the pre-cleanup tree; this document and its machine-readable inventory
describe the post-Phase-3C-1 baseline.

## Ownership boundaries

The scopes overlap intentionally: for example, project-owned ConRep files are
currently located inside the broad legacy `llm2vec/` tree. A path is not
third-party merely because it sits below that directory.

| Component | Intended ownership | Current boundary | Phase 3D-0 conclusion |
| --- | --- | --- | --- |
| ConRep | Project | Root `llm2vec/ContrastiveUnlearning*.py` files (12 paths) | Mixed legacy layout; preserve the historical entry point before any move. |
| ClinicIA | Project | `llm2vec/unlearn_eval/` (49 tracked paths) | Evaluation code, configurations, data, and launchers remain mixed. |
| LLM2Vec | Third party plus project modifications | `llm2vec/`, excluding nested `open_unlearning/` (251 paths) | Pin the inferred upstream tree and classify local changes before isolation. |
| OpenUnlearning | Third party plus project adapters/configurations | `llm2vec/open_unlearning/` (289 paths) | Keep GradDiff, NPO, and RMU attributable to upstream; do not copy them into ConRep ownership. |
| Legacy datasets | Dataset-specific; redistribution unresolved | `llm2vec/UnlearnData/` (26 paths) | Contains 24 LFS pointers, one JSON file, and one misplaced Python source file. |
| Historical and paper evidence | Project, protected | `configs/historical/`, `results/paper/`, root archives, release tooling | Must remain unchanged during source migration. |

## Pinned upstream comparisons

| Comparison | Local | Upstream | Common | Identical | Modified | Local-only | Upstream-only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LLM2Vec at `312adcfb` | 251 | 105 | 104 | 99 | 5 | 147 | 1 |
| OpenUnlearning at `d33c476` | 289 | 204 | 204 | 199 | 5 | 85 | 0 |

The five LLM2Vec same-path modifications are:

- `experiments/run_simcse.py`
- `llm2vec/loss/HardNegativeNLLLoss.py`
- `llm2vec/loss/__init__.py`
- `llm2vec/loss/utils.py`
- `train_configs/simcse/Mistral.json`

The only upstream-only LLM2Vec path is `.gitignore`. The five OpenUnlearning
same-path modifications are:

- `configs/eval/tofu.yaml`
- `configs/experiment/finetune/tofu/default.yaml`
- `configs/trainer/finetune.yaml`
- `src/data/__init__.py`
- `src/trainer/unlearn/npo.py`

“Local-only” is a comparison result, not an ownership verdict. Those paths
still require classification as project source, adapter, configuration, data,
launcher, historical artifact, or another category.

## Entrypoints and migration contracts

The inventory blob-anchors eleven critical entry points. The most important
historical contract is
`llm2vec/ContrastiveUnlearning_Adaptive_RandomToken_LMloss_margin.py`: it is the
single non-null training entry point resolved by all five selected historical
ConRep records. It must remain traceable if a later batch introduces a new
canonical entry point.

Model-facing training and evaluation entry points are recorded but deliberately
not executed by this audit. The historical-configuration validator, archived
metric extractor, and paper-table builder remain the only entry points approved
for lightweight Phase 3D-0 validation.

## Duplicate LLM2Vec packages

`llm2vec/llm2vec/` has 20 tracked paths and `llm2vec/llm22vec/` has 21. They
share 19 relative names; 18 are blob-identical, while `__init__.py` differs.
The canonical-only path is `llm2vec.py`; the derivative-only paths are
`llm22vec.py` and `openunlearn_wrapper.py`.

The derivative changes model selection, bidirectional defaults, token handling,
hidden-state pooling, and OpenUnlearning adaptation. Import, construction, and
forward/pooling equivalence have not been demonstrated. Therefore deleting or
folding `llm22vec` is explicitly blocked until focused adapter contract tests
exist.

## Dependencies and licensing

There is no root `pyproject.toml`, `requirements.txt`, `environment.yml`, or
`setup.py`, so the repository has no validated top-level reproduction
environment. LLM2Vec declares Python `>=3.8` and Transformers
`>=4.43.1,<=4.44.2`; nested OpenUnlearning declares Python `>=3.11`,
Transformers `==4.45.1`, Torch `==2.4.1`, and optional `lm-eval==0.4.8`.
The Transformers constraints conflict and must be resolved deliberately rather
than combined into an untested environment.

Both inferred upstream components retain tracked MIT license files, anchored by
Git blob and SHA-256 identities in the inventory. The repository has no root
project license. Choosing the license for project-owned ConRep/ClinicIA code is
a researcher decision and is a release blocker; Phase 3D-0 does not infer one.

## Portability, privacy, and data release

The baseline-only scan finds 107 legacy files containing either private-style
absolute paths or known cluster/account markers. These are primarily historical
configuration and launcher records. They were not rewritten because doing so
would alter historical evidence. New validated reproduction configurations
must instead use dataset IDs, environment variables, and portable paths.

The conservative credential-pattern scan found zero matching tracked blobs.
This result is a narrow regression check, not a complete security audit.
Dataset redistribution remains blocked pending license and availability review
for every dataset and LFS object.

## Recorded anomalies

The baseline preserves several paths that need later classification: the stray
`llm2vec/]` file, `llm2vec/train_configs/simcse/Contrasn`, the temporary-looking
`llm2vec/unlearn_eval/tryEval_Mistral_PMC.sh2RR0`, Python source under
`llm2vec/UnlearnData/`, and an uppercase `llm2vec.LOSS.utils` import for which
only the lowercase package is tracked. They are findings, not authorization to
repair or remove content in this audit batch.

## Gates and recommended sequence

The upstream snapshots and historical baseline anchors are complete. The
following remain blocked and should be handled as separately reviewed batches:

1. Add import, adapter, construction, and forward/pooling contract tests,
   without moving source.
2. Isolate project-owned ConRep behind a stable entry point while preserving a
   compatibility path for `historical_v1` records.
3. Establish the ClinicIA registry, dataset adapters, and shared evaluation
   protocol, keeping future `validated_v2` behavior distinct from historical
   measurements.
4. Consolidate or vendor upstream code only after equivalence, attribution,
   dependency, and license gates pass.
5. Add a portable reproduction environment and `configs/reproduction/`, then
   review datasets, public release surfaces, and the root README.

Run the offline baseline check with:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python scripts/repository/validate_source_inventory.py \
  --inventory docs/repository_source_inventory.json
```

This validator inspects Git metadata and blob content only. It performs no model
initialization, download, training, inference, or scientific evaluation.
