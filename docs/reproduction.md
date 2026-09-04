# Reproduction guide

The repository supports three different tasks. They should not be conflated.

## 1. Rebuild the published evidence tables

This path is lightweight, offline, and available now:

```bash
python scripts/reproduce.py tables
python scripts/reproduce.py table 1
```

It verifies the two evidence archives, extracts 25 experiment snapshots, and
reconstructs Tables 1–3 plus the raw-value reconciliation used for Tables 4–6.
It does not load or evaluate a language model.

## 2. Inspect or relaunch a selected historical ConRep run

```bash
python scripts/train_conrep.py list
python scripts/train_conrep.py show a-diagnosis-mistral-conrep
python scripts/evaluate_clinicia.py show a-diagnosis-mistral-conrep
```

To launch after preparing the correct model environment and LFS data:

```bash
python scripts/train_conrep.py run a-diagnosis-mistral-conrep \
  --output-root /path/to/new/artifacts
```

The launcher writes a temporary normalized config: old cluster dataset paths
become repository paths and outputs go under the requested root. Original
historical files remain untouched. Model weights, PEFT checkpoints, compatible
CUDA/PyTorch packages, and restricted datasets are not bundled.

## 3. Run a new validated-v2 experiment

`configs/reproduction/` describes the intended portable contract, but its
gates remain explicit and currently false. Do not write new measurements into
`results/paper/`; use the separate `results/validated_v2/` namespace after
dependency, dataset-rights, checkpoint-lineage, smoke-test, and numerical
protocol review.

## Environment boundary

Pinned LLM2Vec and OpenUnlearning snapshots require incompatible Transformers
versions. Prepare them as separate environments; see
[`environments/README.md`](../environments/README.md). No setup command in
this repository downloads a model or LFS object automatically.
