# ConRep and ClinicIA

Representation-space unlearning and diversified evaluation for identifier–attribute knowledge in language models.

## Artifact status

This release reconstructs tables from archived experiment outputs; no experiment was rerun. Incomplete provenance, missing metrics, and different starting checkpoints are explicitly marked. This is not currently a claim of complete end-to-end reproduction.

## Research overview

Identifier–attribute knowledge associates an identifier, attribute type, and value. Fixed textual probes can underestimate extraction risk when equivalent meanings have diverse expressions. **ConRep** performs representation-space unlearning, while **ClinicIA** evaluates QA, cloze, background, and identifier/attribute MCQ expressions. Regime A studies asymmetric celebrity diagnosis/death retention and forgetting; Regime B studies injected PMC knowledge with symmetric retain/forget evaluation. Future detail will live in `docs/method.md` and `docs/benchmark.md`.

## Repository layout

Current research implementations and evidence remain under `llm2vec/` and the two root archives. The additive provenance layer is under `results/paper/`, `scripts/results/`, `tests/results/`, and `docs/`. A future cleanup may introduce `src/`, `configs/paper/`, `data/clinicia/`, and organized training/evaluation scripts; those moves have not occurred.

## Installation

A pinned training environment is still to be published. Archived table construction is CPU-only and uses the Python standard library.

## Data availability

| Dataset/regime | Purpose | Included | Download/generation | License/status | Restrictions |
|---|---|---|---|---|---|
| Celebrity IA / A | Forget and complementary-retain probes | Archived/partly LFS | Generator documentation pending | Redistribution unverified | Do not infer redistribution rights |
| Wikipedia retain / A | Retain textual views | LFS pointer evidence | Preparation pending | Status unresolved | Upstream terms apply |
| PMC IA / B | Injected retain/forget knowledge and probes | Archived/partly LFS | Generation documentation pending | Redistribution unverified | Clinical-data review required |

## Reproduction matrix

The manifest records experiment ID, regime, model, method, training/evaluation evidence, archived status, and comparability notes. See `results/paper/manifest.json` and `docs/experiment_provenance.md`.

## Rebuild archived tables

```bash
python scripts/results/build_tables.py \
  --manifest results/paper/manifest.json \
  --output-dir results/paper/reconstructed
```

This command only rebuilds tables from compact snapshots. It does not train or evaluate a model.

## Results

See `results/paper/reconstructed/`. `†` means the exact baseline MCQ test did not exceed chance; `‡` means a different/non-comparable starting checkpoint; `N/A` means no valid archived value; `–` means a significance-masked probe.

The exact one-sided binomial decision is made per model and probe from archived full-precision accuracy and verified dataset size. Regime B row averages include unmasked retain and forget percentages but not their derived Delta columns, and MMLU is never averaged.

## Provenance and known limitations

- [`docs/experiment_provenance.md`](experiment_provenance.md)
- [`docs/result_reconciliation.md`](result_reconciliation.md)
- [`docs/archived_results_policy.md`](archived_results_policy.md)

## Third-party code

The final README will explicitly attribute LLM2Vec and OpenUnlearning with upstream repositories, licenses, pinned revisions, and descriptions of local modifications. Exact upstream revisions remain to be established.

## Citation

Anonymous manuscript under review. An arXiv/OpenReview link and `CITATION.cff` will be added after public release; no author identities or fabricated citation are provided here.
