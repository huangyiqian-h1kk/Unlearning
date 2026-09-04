# ConRep + ClinicIA

This is the paper repository for *Towards Unlearning Beyond Textual Expressions for LLMs*.
It contains the project-owned **ConRep** unlearning method, the
**ClinicIA** evaluation, paper experiment records, archived evidence, and the
LLM2Vec/OpenUnlearning code used by the implementations.

Paper: [OpenReview submission S9JBN7LmH0](https://openreview.net/forum?id=S9JBN7LmH0).

ConRep removes an identifier-attribute relation in representation space.
ClinicIA then asks whether the same relation remains recoverable through six
linguistic views:

- generated **QA**, **Cloze**, and **Background (BG)** probes;
- multiple-choice **ATT**, **IDeq**, and **ID** probes.

The paper studies two settings:

- **Regime A:** unlearn pre-existing celebrity diagnosis or death attributes
  from Llama-2 7B Chat and Mistral 7B Instruct;
- **Regime B:** first inject PMC clinical relations into Mistral, then unlearn
  100 forget relations while retaining the other 900.

## Reproducibility contract

Every runnable path below states its command, required inputs, output location,
paper experiment, and verification level. The labels mean:

| Label | What has actually been verified |
| --- | --- |
| **Offline-verified** | The command is run by the repository test suite and reproduces committed artifacts byte-for-byte. |
| **Capsule-verified** | The exact historical config/job identity is preserved and the portable config is tested; a real model rerun has not been performed in this release. |
| **Config-derived** | The command is reconstructed from archived resolved configuration; the historical training entry point or numerical rerun is not verified. |
| **Evidence-only** | Metrics/provenance exist, but the repository cannot honestly claim an executable numerical reproduction. |

No model command writes into `results/paper/`. New runs use
`results/validated_v2/` so archived evidence remains immutable.

## Quick start

| Reproduction path | Required inputs | Output | Paper experiment | Verification | Commands |
| --- | --- | --- | --- | --- | --- |
| Archived tables | Two root archives and paper manifest | Archived metrics and tables | Tables 1-6, all 25 cells | **Offline-verified** | [Section 6](#6-rebuild-the-archived-paper-tables) |
| PMC SFT | Mistral and full PMC CSV | Regime B starting model | Tables 2 and 6 baseline/start | **Config-derived** | [Section 2](#2-prepare-the-regime-b-pmc-starting-model-with-sft) |
| ConRep | Base model, LLM2Vec adapter, forget/retain CSVs | ConRep adapter | Five ConRep cells | **Capsule-verified** | [Section 3.1](#31-conrep) |
| GradDiff/NPO/RMU | Starting model and forget/retain data | Unlearned model | 15 baseline cells | 13 **Config-derived**; two deaths-RMU **Evidence-only** | [Section 3.2](#32-graddiff-npo-and-rmu) |
| ClinicIA | Model plus generation/MCQ probes | Six-view metrics and logs | All 25 cells | Protocol **Capsule-verified**; numbers pending | [Section 4](#4-evaluate-all-six-clinicia-views) |
| MMLU | Model/adapter and lm-eval | General-utility result | Utility column in Tables 1-6 | Mixed by cell | [Section 5](#5-evaluate-general-utility-with-mmlu) |

Each linked section contains complete commands rather than placeholders.

## 0. Install the two model environments

The evidence-only table path needs Python 3.10+ and no ML packages. Model work
uses two environments because the pinned Transformers ranges do not overlap.

### ConRep and ClinicIA environment

```bash
conda create -n conrep-clinicia python=3.11 -y
conda activate conrep-clinicia
pip install -e third_party/llm2vec
pip install -e .
pip install lm-eval==0.4.8
pip install --no-build-isolation flash-attn
```

- **Inputs:** a CUDA/PyTorch installation compatible with the host, and access
  to the paper's Hugging Face base models/adapters.
- **Output:** the `conrep-clinicia` environment; no result files.
- **Paper experiments:** all ConRep and ClinicIA/MMLU runs.
- **Verification:** dependency pins are statically validated; installation and
  a real GPU smoke test are **not yet verified** by this release.

### OpenUnlearning environment for SFT, GradDiff, NPO, and RMU

```bash
conda create -n conrep-openunlearning python=3.11 -y
conda activate conrep-openunlearning
pip install -e './third_party/open-unlearning[lm-eval]'
pip install --no-build-isolation flash-attn==2.6.3
```

- **Inputs:** a CUDA stack compatible with pinned `torch==2.4.1`.
- **Output:** the `conrep-openunlearning` environment; no result files.
- **Paper experiments:** PMC SFT and the 15 GradDiff/NPO/RMU cells.
- **Verification:** the vendored requirements and incompatible environment
  boundary are validated offline; installation and a real GPU smoke test are
  **not yet verified**.

The historical jobs requested one GPU and used bf16/FlashAttention 2. They did
not preserve the GPU model or a portable CUDA module specification, so this
repository does not invent an exact VRAM claim. See
[`environments/README.md`](environments/README.md) for the full boundary.

## 1. Materialize and check the data

First inspect what is already present:

```bash
python scripts/reproduce.py data-status
```

If you are authorized to access the repository's LFS objects, materialize the
paper data and require a complete catalog:

```bash
git lfs install
git lfs pull --include='data/clinicia/**'
python scripts/reproduce.py data-status --require-materialized
```

- **Inputs:** the 28 paths in `data/clinicia/catalog.json`; 21 are identity-
  preserved LFS pointers in Git and seven small non-LFS probes are committed.
- **Output:** materialized files at their catalogued paths; the check itself
  writes nothing.
- **Paper experiments:** all Regime A/B training and ClinicIA evaluation.
- **Verification:** path identity, LFS OIDs, and historical-to-current mapping
  are **offline-verified**. Data redistribution permission and availability of
  the LFS payloads remain external requirements.

## 2. Prepare the Regime B PMC starting model with SFT

Activate `conrep-openunlearning`, then run:

```bash
python scripts/reproduce.py sft-pmc \
  --train-data data/clinicia/regime_b/pmc/training/easy_QA_PMC_full.csv \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --output-dir results/validated_v2/b-pmc-mistral-baseline/model
```

Use `--dry-run` to print the exact underlying Hydra command without importing a
model:

```bash
python scripts/reproduce.py sft-pmc \
  --train-data data/clinicia/regime_b/pmc/training/easy_QA_PMC_full.csv \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --output-dir results/validated_v2/b-pmc-mistral-baseline/model \
  --dry-run
```

- **Inputs:** Mistral-7B-Instruct-v0.2 and the materialized full PMC training
  CSV. The vendored config uses learning rate `1.5e-5`, weight decay `0.01`, one
  warmup epoch, and ten training epochs.
- **Output:** model/checkpoint and Hydra run files under
  `results/validated_v2/b-pmc-mistral-baseline/model/`.
- **Paper experiment:** Regime B starting model/baseline for Tables 2 and 6.
- **Verification:** **Config-derived.** The archived records name both
  `pmc_full_universal_Mistral-7B_continual` and
  `pmc_full_csv_Mistral-7B-Instruct_epoch10/checkpoint-7500`. That lineage
  difference is unresolved, so this command is a current validated-v2
  candidate, not a claim that it recreates every historical PMC starting
  checkpoint byte-for-byte.

Regime A starts directly from the public Llama-2 or Mistral instruction models
and therefore has no paper-specific SFT stage.

## 3. Perform unlearning

### 3.1 ConRep

Preview the complete portable configuration before spending GPU time:

```bash
python scripts/train_conrep.py config a-diagnosis-mistral-conrep \
  --output-root results/validated_v2
```

Run that paper cell:

```bash
python scripts/train_conrep.py run a-diagnosis-mistral-conrep \
  --output-root results/validated_v2
```

- **Inputs:** `mistralai/Mistral-7B-Instruct-v0.2`,
  `McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp`,
  `data/clinicia/regime_a/diagnosis/training/easy_qa.csv`, and
  `data/clinicia/regime_a/shared/retain/wikitext_dup_1_trunc_1.csv`.
- **Output:**
  `results/validated_v2/a-diagnosis-mistral-conrep/model/`.
- **Paper experiment:** Regime A diagnosis, Mistral ConRep, Tables 1 and 4.
- **Verification:** **Capsule-verified.** The exact historical config and job
  are preserved; config normalization is tested. Model download, GPU execution,
  and numerical equivalence have not been verified in the public release.

The five exact ConRep capsule IDs are:

| Experiment ID | Starting model | Paper tables | Extra required override | Verification |
| --- | --- | --- | --- | --- |
| `a-diagnosis-llama2-conrep` | Llama-2 7B Chat | 1, 4 | none | **Capsule-verified** |
| `a-diagnosis-mistral-conrep` | Mistral 7B Instruct | 1, 4 | none | **Capsule-verified** |
| `a-deaths-llama2-conrep` | Llama-2 7B Chat | 3, 5 | none | **Capsule-verified** |
| `a-deaths-mistral-conrep` | Mistral 7B Instruct | 3, 5 | none | **Capsule-verified** |
| `b-pmc-mistral-conrep` | PMC-injected Mistral | 2, 6 | `--model-path <PMC_SFT_MODEL> --peft-model none` | **Capsule-verified config; starting checkpoint external** |

For example, after producing a reviewed PMC starting model:

```bash
python scripts/train_conrep.py run b-pmc-mistral-conrep \
  --model-path results/validated_v2/b-pmc-mistral-baseline/model \
  --peft-model none \
  --output-root results/validated_v2
```

### 3.2 GradDiff, NPO, and RMU

Regime A Mistral-diagnosis commands for all three baselines:

```bash
conda activate conrep-openunlearning
python scripts/reproduce.py baseline-unlearn a-diagnosis-mistral-graddiff \
  --model-path mistralai/Mistral-7B-Instruct-v0.2 \
  --forget-data data/clinicia/regime_a/diagnosis/training/easy_qa.csv \
  --output-dir results/validated_v2/a-diagnosis-mistral-graddiff/model

python scripts/reproduce.py baseline-unlearn a-diagnosis-mistral-npo \
  --model-path mistralai/Mistral-7B-Instruct-v0.2 \
  --forget-data data/clinicia/regime_a/diagnosis/training/easy_qa.csv \
  --output-dir results/validated_v2/a-diagnosis-mistral-npo/model

python scripts/reproduce.py baseline-unlearn a-diagnosis-mistral-rmu \
  --model-path mistralai/Mistral-7B-Instruct-v0.2 \
  --forget-data data/clinicia/regime_a/diagnosis/training/easy_qa.csv \
  --output-dir results/validated_v2/a-diagnosis-mistral-rmu/model
```

Regime B example (PMC NPO):

```bash
python scripts/reproduce.py baseline-unlearn b-pmc-mistral-npo \
  --model-path results/validated_v2/b-pmc-mistral-baseline/model \
  --forget-data data/clinicia/regime_b/pmc/training/easy_QA_PMC_forget100_state.csv \
  --retain-data data/clinicia/regime_b/pmc/training/easy_QA_PMC_retain900_full.csv \
  --output-dir results/validated_v2/b-pmc-mistral-npo/model
```

Append `--dry-run` to either command to inspect the exact Hydra invocation.

- **Inputs:** the cell's starting model and forget data. Regime A's vendored
  configs use `WikiTextRetainDataset`, which additionally needs dataset network
  access or a populated Hugging Face cache. Regime B requires the explicit
  retain CSV shown above.
- **Output:** `results/validated_v2/<experiment-id>/model/` with trainer state,
  Hydra config, and saved model/checkpoint files.
- **Paper experiments:** use IDs of the form
  `a-{diagnosis|deaths}-{llama2|mistral}-{graddiff|npo|rmu}` or
  `b-pmc-mistral-{graddiff|npo|rmu}`. These are the 15 non-ConRep unlearning
  cells in Tables 1-6.
- **Verification:** 13 cells are **Config-derived**. Archived metrics, resolved Hydra config,
  and checkpoint lineage exist, but the records explicitly say that the
  historical training entry point is unresolved. These commands expose the
  closest auditable current execution path, explicitly override the template
  with each record's resolved learning rate and epoch count, and do not claim
  historical command identity or reproduced numbers. The two deaths-RMU cells
  (`a-deaths-llama2-rmu` and `a-deaths-mistral-rmu`) have conflicting historical
  model/target identity and are **Evidence-only**. The CLI can render a current
  canonical-cell proposal for them, but that proposal is not historical
  reproduction evidence.

Baseline rows do not perform unlearning: Regime A evaluates the public base
model; Regime B evaluates the reviewed PMC SFT model.

## 4. Evaluate all six ClinicIA views

The same entry point evaluates ConRep, baselines, and unlearned OpenUnlearning
models. Preview the exact normalized evaluation config:

```bash
python scripts/evaluate_clinicia.py paper-config \
  a-diagnosis-mistral-conrep \
  --model-path results/validated_v2/a-diagnosis-mistral-conrep/model \
  --output-root results/validated_v2
```

Then execute it in the ConRep/ClinicIA environment:

```bash
python scripts/evaluate_clinicia.py run-model \
  a-diagnosis-mistral-conrep \
  --model-path results/validated_v2/a-diagnosis-mistral-conrep/model \
  --output-root results/validated_v2
```

For Regime B, run both forget and retain roles:

```bash
python scripts/evaluate_clinicia.py run-model b-pmc-mistral-npo \
  --model-path results/validated_v2/b-pmc-mistral-npo/model \
  --role evaluate_forget \
  --output-root results/validated_v2

python scripts/evaluate_clinicia.py run-model b-pmc-mistral-npo \
  --model-path results/validated_v2/b-pmc-mistral-npo/model \
  --role evaluate_retain \
  --output-root results/validated_v2
```

- **Inputs:** a model or PEFT adapter; the matching generation JSONL; ATT,
  IDeq, and ID MCQ JSONL files. The command selects the matching Regime/target/
  model protocol from the five exact ConRep evaluation capsules.
- **Output:**
  `results/validated_v2/<experiment-id>/clinicia/<forget|retain>/evaluation_results.json`
  and `detailed_outputs.jsonl`.
- **Paper experiments:** all 25 cells. Generation outputs implement QA, Cloze,
  and BG; MCQs are scored by option log likelihood for ATT, IDeq, and ID.
- **Verification:** the evaluation source, prompts, dataset mapping, and selected
  ConRep configs are **capsule-verified**. Real-model numerical reproduction is
  not yet verified; for the 20 non-ConRep cells the model checkpoint remains an
  explicit external input.

## 5. Evaluate general utility with MMLU

ConRep saves a PEFT adapter, so supply both the starting model and adapter:

```bash
lm-eval --model hf \
  --model_args pretrained=mistralai/Mistral-7B-Instruct-v0.2,peft=results/validated_v2/a-diagnosis-mistral-conrep/model,tokenizer=mistralai/Mistral-7B-Instruct-v0.2,parallelize=True \
  --tasks mmlu \
  --batch_size 32 \
  --output_path results/validated_v2/a-diagnosis-mistral-conrep/mmlu
```

For a full baseline/unlearned checkpoint:

```bash
lm-eval --model hf \
  --model_args pretrained=results/validated_v2/a-diagnosis-mistral-npo/model,tokenizer=results/validated_v2/a-diagnosis-mistral-npo/model,parallelize=True \
  --tasks mmlu \
  --batch_size 32 \
  --output_path results/validated_v2/a-diagnosis-mistral-npo/mmlu
```

- **Inputs:** `lm-eval==0.4.8`, the starting model, and either the ConRep PEFT
  adapter or a complete baseline/unlearned checkpoint.
- **Output:** lm-eval results under
  `results/validated_v2/<experiment-id>/mmlu/`.
- **Paper experiment:** the MMLU/general-utility column associated with the
  chosen experiment ID in Tables 1-6.
- **Verification:** the command form is directly preserved for the four
  selected Regime A ConRep jobs. MMLU settings/evidence are missing for some
  cells, including the selected PMC ConRep capsule and several diagnosis
  baselines. Check `mmlu_evidence.settings_status` in
  `configs/paper/historical/<experiment-id>.json`; a `missing` value means this
  command is a current protocol proposal, not an exact historical reproduction.

## 6. Rebuild the archived paper tables

```bash
python scripts/reproduce.py rebuild-tables
python scripts/reproduce.py table 1 --no-rebuild
```

- **Inputs:** `clinicia_provenance_bundle.tar.gz`,
  `clinicia_configs_mmlu_bundle.tar.gz`, and `results/paper/manifest.json`.
- **Output:** 26 extracted files in `results/paper/raw_metrics/` and ten table/
  reconciliation files in `results/paper/reconstructed/`.
- **Paper experiment:** all 25 method/model/target cells in Tables 1-6.
- **Verification:** **Offline-verified** and byte-deterministic across repeated
  extraction/table builds.

This path rebuilds the archived paper evidence. A converter that promotes new
`results/validated_v2/` measurements into a reviewed manifest does not yet
exist, so the repository does **not** claim that new GPU runs automatically
recreate the paper tables.

## Paper matrix and experiment IDs

| Paper tables | Regime and target | Models | Methods |
| --- | --- | --- | --- |
| 1 and 4 | A: celebrity diagnosis | Llama-2 7B Chat, Mistral 7B Instruct | Baseline, GradDiff, NPO, RMU, ConRep |
| 2 and 6 | B: injected PMC | Mistral 7B Instruct | Baseline, GradDiff, NPO, RMU, ConRep |
| 3 and 5 | A: celebrity deaths | Llama-2 7B Chat, Mistral 7B Instruct | Baseline, GradDiff, NPO, RMU, ConRep |

All 25 records are indexed in
[`configs/paper/historical/index.json`](configs/paper/historical/index.json).
Only the five ConRep cells have exact selected training/evaluation/job capsules
under [`experiments/paper_runs/`](experiments/paper_runs/). This distinction is
intentional and visible in every command above.

## Repository map

| Path | Purpose |
| --- | --- |
| [`src/conrep/`](src/conrep/) | ConRep objectives, token corruption, stable dispatch, and preserved trainers. |
| [`src/clinicia/`](src/clinicia/) | ClinicIA probe definitions, metrics, registry, and evaluators. |
| [`scripts/`](scripts/) | Paper-facing SFT, unlearning, evaluation, and table commands. |
| [`experiments/paper_runs/`](experiments/paper_runs/) | Five exact selected ConRep run capsules. |
| [`configs/paper/historical/`](configs/paper/historical/) | Evidence-backed records for all 25 paper cells. |
| [`data/clinicia/`](data/clinicia/) | Regime A/B datasets and the historical-to-current path catalog. |
| [`results/paper/`](results/paper/) | Immutable archived evidence and reconstructed paper tables. |
| `results/validated_v2/` | Reserved output namespace for new model runs; not committed as paper evidence. |
| [`third_party/`](third_party/) | Audited LLM2Vec and OpenUnlearning snapshots. |
| [`legacy/`](legacy/) | Exploratory or superseded code/jobs not claimed as selected paper runs. |

See the [paper-to-code map](docs/paper-to-code.md) for equations and modules and
the [expanded reproduction guide](docs/reproduction.md) for provenance details.

## Offline validation

```bash
python scripts/configs/validate_historical_experiments.py \
  --index configs/paper/historical/index.json \
  --manifest results/paper/manifest.json
python scripts/configs/validate_reproduction_configs.py \
  --index configs/reproduction/index.json
python scripts/repository/validate_release_inventory.py \
  --manifest data/lfs_manifest.json \
  --dependencies docs/dependency_matrix.json
python scripts/train_conrep.py config a-diagnosis-mistral-conrep \
  --output-root results/validated_v2
python scripts/evaluate_clinicia.py paper-config \
  a-diagnosis-mistral-conrep \
  --model-path results/validated_v2/a-diagnosis-mistral-conrep/model \
  --output-root results/validated_v2
python -m unittest discover -s tests -p 'test_*.py'
```

These checks do not download a model or LFS object, initialize a model, submit a
scheduler job, train, or evaluate. Historical machine paths remain visible only
as provenance; public commands require explicit current-machine inputs.
