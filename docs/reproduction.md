# Reproduction guide and evidence boundary

The [top-level README](../README.md) is the executable guide. It gives complete
commands, required inputs, outputs, paper-table correspondence, and verification
status for environment setup, data checks, PMC SFT, ConRep, GradDiff/NPO/RMU,
ClinicIA, MMLU, and archived table reconstruction.

This document explains why those commands have different confidence labels.

## Two reproducibility targets

| Target | Command | Inputs | Outputs | Paper scope | Status |
| --- | --- | --- | --- | --- | --- |
| Rebuild archived evidence | `python scripts/reproduce.py rebuild-tables` | Root archives plus `results/paper/manifest.json` | `results/paper/raw_metrics/` and `results/paper/reconstructed/` | Tables 1-6, 25 cells | Offline-verified and byte-deterministic |
| Rerun models | README stages 0-5 | Models, materialized data, two GPU environments | `results/validated_v2/<experiment-id>/` | SFT, unlearning, ClinicIA, MMLU | Mixed; capsule/config status is recorded per stage |

The first target reproduces what is publicly archived. The second attempts to
recreate models and measurements. Success at the first target is not evidence
that the second target has been numerically validated.

## Historical identity versus a current command

### Selected ConRep runs

Five ConRep cells have exact repository files selected by evidence:

- a training JSON;
- a ClinicIA evaluation JSON (two for Regime B);
- the original scheduler job.

They live under `experiments/paper_runs/<experiment-id>/historical/`. The public
launcher reads those files, maps known dataset names through
`data/clinicia/catalog.json`, and redirects output to a caller-supplied root.
The original capsule is never edited.

```bash
python scripts/train_conrep.py config a-diagnosis-mistral-conrep \
  --output-root results/validated_v2
python scripts/train_conrep.py run a-diagnosis-mistral-conrep \
  --output-root results/validated_v2
```

This is **capsule-verified**, but still requires model access, materialized LFS
data, a compatible environment, and a real GPU validation before numerical
equivalence can be claimed.

### GradDiff, NPO, and RMU

For these 15 cells, the evidence archives preserve metrics, resolved Hydra
configuration, trainer state, and (for some cells) MMLU logs. The historical
records also explicitly state that the training entry point is unresolved.

The command below therefore reconstructs a current invocation from the
resolved config and the matching vendored experiment template:

```bash
python scripts/reproduce.py baseline-unlearn a-diagnosis-mistral-npo \
  --model-path mistralai/Mistral-7B-Instruct-v0.2 \
  --forget-data data/clinicia/regime_a/diagnosis/training/easy_qa.csv \
  --output-dir results/validated_v2/a-diagnosis-mistral-npo/model \
  --dry-run
```

This is **config-derived**, not an assertion that the displayed command was the
historical job command. Thirteen baseline cells have this status. The two
deaths-RMU cells have conflicting historical model/target identity and remain
**evidence-only**; their rendered commands are current canonical-cell proposals.

### PMC SFT lineage

The Regime B records contain two starting-model labels:

- `pmc_full_universal_Mistral-7B_continual` for ConRep/NPO/RMU and the baseline
  path label;
- `pmc_full_csv_Mistral-7B-Instruct_epoch10/checkpoint-7500` for the baseline's
  explicit starting checkpoint and GradDiff.

The current SFT command uses the committed PMC finetuning template and full PMC
CSV. It is a validated-v2 candidate, not proof that both labels refer to the
same checkpoint:

```bash
python scripts/reproduce.py sft-pmc \
  --train-data data/clinicia/regime_b/pmc/training/easy_QA_PMC_full.csv \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --output-dir results/validated_v2/b-pmc-mistral-baseline/model \
  --dry-run
```

## ClinicIA protocol

`scripts/evaluate_clinicia.py` can build an evaluation config for any of the 25
paper experiment IDs. It selects the exact ConRep capsule with the same regime,
target, and base model as the protocol template; only the evaluated checkpoint
and new output location change.

```bash
python scripts/evaluate_clinicia.py paper-config \
  a-diagnosis-mistral-npo \
  --model-path results/validated_v2/a-diagnosis-mistral-npo/model \
  --output-root results/validated_v2
```

The resulting evaluator implements QA, Cloze, BG, ATT, IDeq, and ID and writes
`evaluation_results.json` plus `detailed_outputs.jsonl`. Regime B requires
separate `evaluate_forget` and `evaluate_retain` invocations, as shown in the
README.

## MMLU evidence

The four selected Regime A ConRep jobs preserve their exact `lm-eval` command
form. Other cells have mixed status. Read the authoritative field before making
an exact historical claim:

```bash
python -c "import json; print(json.load(open('configs/paper/historical/a-diagnosis-mistral-npo.json'))['mmlu_evidence'])"
```

- `settings_status: resolved` means the archive identifies the settings source;
- `settings_status: missing` means the README command is a current protocol
  proposal only.

## Output policy

- `results/paper/` is immutable archived evidence.
- New model runs and measurements go to
  `results/validated_v2/<experiment-id>/`.
- A reviewed converter from new measurement files to a new table manifest is
  not implemented. Do not copy new values into the archived manifest by hand
  and call them reproduced paper results.

The structural candidate records in `configs/reproduction/` remain conservative
until dependency, data-rights, checkpoint-lineage, real-model smoke-test, and
numerical-protocol gates have all been reviewed.
