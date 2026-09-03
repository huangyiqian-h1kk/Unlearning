# Third-party attribution and local modifications

This file distinguishes attributed upstream components from project-owned
ConRep/ClinicIA code. The pinned comparisons are reproducible from the checked
in snapshots under `docs/upstream_snapshots/`; they are content-tree
inferences, not claims of Git ancestry.

## LLM2Vec

- Upstream: <https://github.com/McGill-NLP/llm2vec>
- Pinned inferred revision: `312adcfb578b4023bfc9d3bcc83b6b87448ab059`
- Recorded tree: `0bf0b8f4960e4c65f6995b16756ac147a30fa000`
- License: MIT, preserved at [`llm2vec/LICENSE`](llm2vec/LICENSE)
- Local boundary: `llm2vec/`, excluding nested `open_unlearning/` and the
  separately classified project-owned ConRep/ClinicIA paths

The pinned comparison records five same-path modifications:

- `experiments/run_simcse.py`
- `llm2vec/loss/HardNegativeNLLLoss.py`
- `llm2vec/loss/__init__.py`
- `llm2vec/loss/utils.py`
- `train_configs/simcse/Mistral.json`

The project-specific `llm22vec` causal implementation and OpenUnlearning
wrapper are not represented as unmodified upstream LLM2Vec. Phase 3D-3 removes
only eighteen support modules that were exactly blob-identical to their
canonical `llm2vec` peers; the adapter's behaviorally distinct source remains.

## OpenUnlearning

- Upstream: <https://github.com/locuslab/open-unlearning>
- Pinned inferred revision: `d33c4762941731cb397494f605f71528d8ce9dce`
- Recorded tree: `22cf1a2efd0dd194dfbdd0be3d4d3723a4936888`
- License: MIT, preserved at
  [`llm2vec/open_unlearning/LICENSE`](llm2vec/open_unlearning/LICENSE)
- Local boundary: `llm2vec/open_unlearning/`

The pinned comparison records five same-path modifications:

- `configs/eval/tofu.yaml`
- `configs/experiment/finetune/tofu/default.yaml`
- `configs/trainer/finetune.yaml`
- `src/data/__init__.py`
- `src/trainer/unlearn/npo.py`

Project-added datasets, configurations, adapters, launchers, and historical
artifacts inside the nested tree are not automatically upstream-owned merely
because of their path.

## Separate rights surfaces

Model weights, tokenizers, datasets, papers, and generated outputs can have
terms different from the source-code licenses above. Their inclusion or LFS
identity does not establish redistribution permission. Dataset decisions are
recorded in [`data/lfs_manifest.json`](data/lfs_manifest.json).

## Project-owned source

`src/conrep/` and `src/clinicia/` are classified as project-owned. No root
license has been selected for them. This document records the unresolved state;
it does not grant a license or infer one from either upstream dependency.
