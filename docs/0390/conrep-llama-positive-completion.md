# Llama completion of the Gemma positive/noise experiments

This campaign adds 18 Llama-3.1-8B-Instruct seed42 runs, matching the positive
and insertion grids already run on Gemma. It reuses the existing Llama
G256B32W2P1/P4 seed42 controls. The rank/batch/weight grid and P1/P4 seeds43/44
are already covered; historical A/B/C/D/J/K seed44 repeats are outside this
positive/noise completion campaign.

| Group | New configurations | Runs |
|---|---|---:|
| Forget corruption / positive count | P1: C50V4, C50V8, C70V8, C90V4, C90V8; retain N00 | 5 |
| Protected retain replacement | P1/P4 × N10/N20; forget C70V4 | 4 |
| Retain insertion | P2/P3/P4 × I1/I2/IB2N20; forget C70V4 | 9 |

Every variant begins with `G256B32W2`; IDs end in `-s42`. C is the forget
token corruption percentage, V its positive count, and N the replacement
probability for eligible retain tokens. I1/I2 insert one/two tokens per
positive at legal random gaps. IB2N20 draws Binomial(2,.2) independently per
positive, giving probabilities .64/.32/.04 for 0/1/2 inserted tokens and mean
.4. It is not 20% per-token insertion and is not intensity-matched to N20.

Priority 0 covers the five forget runs and P2-I1, P4-I1, P4-IB2N20 (eight
runs); priority 1 covers the other ten. Three workers share one queue. A
worker can take the next priority after every priority-0 task is claimed;
there is no barrier waiting for all earlier tasks to finish.

## Fixed protocol and preparation

The base is the verified historical Llama M configuration, transformed with
the same config builders as the Gemma grids. The actual SFT is the existing
`rich-all-ia-seed42/llama8b/injection/sft/final`. Full parameters include:

- LoRA rank256, alpha512, dropout.05; all seven projection modules.
- Global forget/retain/general batch sizes 8/32/32; accumulation1.
- LR1e-5, warmup.05, weight decay0, max gradient norm1, length512, BF16.
- Forget CL weight2; retain/general CL and LM weights1; temperatures .08/.1/.1.
- Forget negative views4 and retain negative views1. No stop-gradient branches.
- 125 updates; save/evaluate 10,20,...,120,125: 13 checkpoints per run and
  **234 new checkpoint validations**. Primary reporting at125, common120 and
  full curves. Ten forget-equivalent sampled epochs: 125×8/100.
- Seed42 unlearning with the previous seed42 SFT and fixed 100/900 training
  split. Validation partition, complete MMLU, and the frozen parent v5 evaluator.

Protected replacement/insertion and clean dropout negatives use the existing
implementations. This campaign does not install parser-review changes or
retrospectively overwrite metrics. Cross-model comparisons must name the same
scoring protocol; any later parser rescore should be applied across methods.

Before a plan can be published, preparation verifies six historical controls
(the four original provenance controls plus Llama P1/P4), their training
completion and all13 committed validation records. Their config, evidence,
data and frozen helper hashes are checked. Missing controls fail preparation
instead of scheduling duplicates. Existing matching run IDs in parent ancestry
also fail preparation and direct the operator to resume that campaign.

The Llama tokenizer audits all900 retain rows separately for replacement and
insertion, including legal vocabulary IDs, protected spans, offsets, gaps and
length headroom. Audits are immutable inputs. No model weights are loaded for
this CPU preflight. Each allocation then performs the existing actual eight-GPU
save/resume smoke; worker0 exercises C90V8, worker1 P4N20, worker2 P4I2.

## Start, or wait for the existing insertion campaign

The bootstrap reads the parent's frozen source and overlays a pinned commit
into the new campaign. It leaves the server checkout and existing jobs intact.
Run from an ABCI login node:

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"

git fetch origin codex/0390-conrep-llama-positive-completion-20261010
lc_ref="$(git rev-parse FETCH_HEAD)"
lc_boot="$(mktemp -d /tmp/0390_llama_completion.XXXXXX)"
lc_base="$PWD/results/validated_v2/0390"
lc_campaign="$lc_base/conrep-llama-positive-completion-20261010"
git show "${lc_ref}:scripts/abci/0390_prepare_conrep_followup.py" > "$lc_boot/prepare.py"

python -I "$lc_boot/prepare.py" \
  --root "$PWD" --ref "$lc_ref" \
  --source-campaign "$lc_base/conrep-insertion-grid-20261010" \
  --campaign "$lc_campaign" --profile llama-positive-completion --hours 10

lc_entry="$lc_campaign/code/scripts/abci/0390_conrep_night.py"
python -I "$lc_entry" chain --campaign "$lc_campaign"
python -I "$lc_entry" chain-status --campaign "$lc_campaign"
)
```

Expected preparation: `tasks=18, gemma=0, llama=18`. The detached handoff waits
for the insertion campaign to finish all training/validation and for its PBS
allocations to exit and reconcile. Waiting does not start the new10-hour budget.
If the parent is already ready, it starts automatically when account capacity
allows. Existing failed/stopped/storage-paused/budget-paused parents block the
handoff; they are not silently resumed. See `handoff.log` and `chain-status`.

Queue R9920261000, account gcg51557, rt_HF8GPU nodes, 12-hour PBS walltime,
maximum three account jobs/nodes and100GB storage reserve are unchanged.
Administrator interruption recovery waits20minutes; intentional cancellation
is not automatically reversed. Queue time after campaign start counts toward
the budget. SSH disconnect is supported; loss of the login host/watcher itself
requires re-arming before start or supervisor recovery after start.

## Status and continuation

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"
lc_campaign="$PWD/results/validated_v2/0390/conrep-llama-positive-completion-20261010"
lc_entry="$lc_campaign/code/scripts/abci/0390_conrep_night.py"
python -I "$lc_entry" chain-status --campaign "$lc_campaign"
python -I "$lc_entry" status --campaign "$lc_campaign"
)
```

Before the successor first starts, repeat `chain` to re-arm a lost watcher.
After start, recover a lost supervisor with the original deadline:

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"
lc_campaign="$PWD/results/validated_v2/0390/conrep-llama-positive-completion-20261010"
python -I "$lc_campaign/code/scripts/abci/0390_conrep_night.py" \
  recover --campaign "$lc_campaign"
)
```

To explicitly give a started campaign another10 hours, use the following
self-contained block. Completed tasks are retained and partial tasks resume:

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"
lc_campaign="$PWD/results/validated_v2/0390/conrep-llama-positive-completion-20261010"
python -I "$lc_campaign/code/scripts/abci/0390_conrep_night.py" \
  recover --campaign "$lc_campaign" --hours 10
)
```

`recover` does not override a deliberate STOP or revive an intentionally
cancelled PBS job without explicit recovery instructions. Inspect such a state
before choosing `resume` or `--requeue-jobs`.

## Export all six campaigns, without prediction payloads

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"
lc_base="$PWD/results/validated_v2/0390"
lc_campaign="$lc_base/conrep-llama-positive-completion-20261010"
lc_ref="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["ref"])' "$lc_campaign/PREPARING.json")"
lc_export="$(mktemp -d /tmp/0390_llama_export.XXXXXX)"
git show "${lc_ref}:scripts/abci/0390_collect_conrep_results.py" > "$lc_export/collect.py"
python -I "$lc_export/collect.py" --root "$PWD" \
  --campaign "$lc_base/conrep-night-20261008" \
  --campaign "$lc_base/conrep-followup-20261008" \
  --campaign "$lc_base/conrep-mixed-grid-20261009" \
  --campaign "$lc_base/conrep-positive-grid-20261010" \
  --campaign "$lc_base/conrep-insertion-grid-20261010" \
  --campaign "$lc_campaign" --no-include-predictions
)
```

This includes all configuration/design records, audits, committed metrics,
augmentation/gradient/sampling observations and handoff state. Weights and
optimizer files are excluded. Use `--include-predictions` when individual
answers are needed; that substantially increases ZIP size. The printed BUNDLE
path is under the persistent project results directory, not node-local /tmp.

## Read-only Qwen SFT checkpoint inventory

The historical candidate is
`rich-all-ia-seed42/qwen7b/injection/sft/checkpoint-700`. Its current availability
must be checked on ABCI. Validation metadata surviving does not establish that
the weights or optimizer/RNG states survived cleanup. This command reads small
JSON files, lists files and checks nonzero shard sizes; it does not load weights,
download models, change a checkpoint, or submit a Qwen job.

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
git fetch origin codex/0390-conrep-llama-positive-completion-20261010
qi_ref="$(git rev-parse FETCH_HEAD)"
qi_tmp="$(mktemp -d /tmp/0390_qwen_inventory.XXXXXX)"
git show "${qi_ref}:scripts/abci/0390_inspect_qwen_sft.py" > "$qi_tmp/inspect.py"
python3 -I "$qi_tmp/inspect.py" \
  --root "$PWD/results/validated_v2/0390/rich-all-ia-seed42/qwen7b"
)
```

`hf_weight_files_complete=true` means config and all index-listed weight files
exist and are nonempty. It is a candidate for model-load/re-evaluation, not proof
of tensor integrity. `resume_files` lists observed Trainer/DeepSpeed state;
`exact_resume_verified` remains false because inventory cannot establish exact
optimizer/scheduler/all-rank RNG restoration. For a usable retained checkpoint,
the next Qwen diagnostic should distinguish training-fact fit from validation
question generalization and compare native chat versus evaluation prompts.

## Verification

The focused ConRep regression suite and Qwen inventory tests passed **191 tests**;
three CPU distributed tests skipped because this executor denies Gloo interface
access. All six shell examples pass `bash -n`. A separate read-only comparison
against the latest complete result export confirmed that all18 algorithm,
optimizer and LoRA configurations match their Gemma counterparts and none of
the new Llama IDs have already run.

CPU tests cover the exact18-cell parity, no overlap with prior Llama controls,
historical completion checks, full900-row token audits, unchanged frozen
evaluator, idempotent preparation, queue priorities, tiny Llama insertion
training and exact RNG/weight/sampler resume, lightweight exports and Qwen shard
inventory. Actual ABCI eight-GPU smoke is enforced at launch and has not been
executed in this local environment.
