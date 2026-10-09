# Mixed Llama completion and Gemma retain grid, 2026-10-09

Use `--profile mixed-grid` for one campaign with three persistent single-node
workers. It combines the unstarted Llama M-S completion with the approved
first-stage Gemma grid. Start this campaign instead of the standalone
`llama-ms` campaign; there is no second three-job pool.

## Matrix and admission

There are normally **28 new runs**:

- Llama M-S at seeds 42 and 43: 14 runs, with exactly the factor definitions in
  [conrep-llama-ms.md](conrep-llama-ms.md).
- Gemma rank 64 and 256, seed 42, each with seven new cells below: 14 runs.
  Alpha/rank remains 2. The eighth cell B16/W5/P1 is historical J/M.

| Admission wave | Llama, seed 42 | Gemma retain batch | Forget weight | Retain positive views |
|---|---|---:|---:|---:|
| 0 | M | 32 | 2 | 4 |
| 1 | N | 16 | 2 | 1 |
| 2 | O | 32 | 5 | 1 |
| 3 | P | 16 | 5 | 4 |
| 4 | Q | 32 | 2 | 1 |
| 5 | R | 16 | 2 | 4 |
| 6 | S | 32 | 5 | 4 |

Every wave includes both Gemma ranks. Within a wave, worker 0 prefers Llama,
worker 1 prefers rank 64 and worker 2 prefers rank 256. These preferences only
break ties: tasks are claimed under the same file lock, and the first free
worker may claim any next task without waiting for a wave barrier. Once these
21 tasks have been admitted, Llama M-S/43 are next. Paused recoverable work
keeps the existing priority over fresh work.

Normally the first three tasks are Llama M/42, Gemma G64B32W2P4/42 and Gemma
G256B32W2P4/42. Their two-step GPU save/resume smokes use these configurations,
so both Gemma smokes exercise B32/P4 before the full runs. The name
`G256B32W2P4` means rank256, retain batch32, forget weight2, four retain views.

Global forget/general batches remain 8/32. Gemma learning rate is 1e-5,
forget views/negative-view budget are 4/4, corruption is .7, LoRA dropout is
.05, and all other CL/LM weights are 1. All runs start from the approved
injected SFT model, use 125 optimizer steps and validate checkpoints
10,20,...,120,125. No test evaluation or automatic checkpoint selection runs.

Preparation checks the historical J and M configurations, frozen core helper
hashes (including sampler/model/evaluator) and all 13 validation markers.
`grid-design.json` records reuse evidence. An absent/incompatible control is
scheduled afresh as G64B16W5P1 or G256B16W5P1, so task count can be 29 or 30.
Historical metrics are referenced, never copied into a new completed task.

The original 59 completed runs plus 28 new runs give 87 total and 1,131
validated checkpoints if all finish. Each fallback control adds one run and
13 validations. The later B64/P8/W1 extensions and winner-seed replications
depend on reviewing this grid and are not preselected/submitted here.

## Positive counts and diagnostics

`conrep.specified_views` counts extra views per original retain anchor.
The grid uses independent LoRA-dropout forwards of the same text. P4 means
four stochastic views, not four distinct semantic paraphrases. General retain
still has one extra view. Specified retain LM is evaluated once on the original
batch; extra views contribute only to retain CL.

Per-positive InfoNCE is averaged over K and then anchors. All own views are
positives. Other retain anchors supply their original representation and exactly
`specified_negative_views=1` extra view, independent of K; forget anchors remain
negatives. K=1 executes the legacy paired-loss path and preserves forward/RNG
order. Tests verify value/gradient equality and that repeating identical views
does not multiply the loss or gradient weight.

Changing retain batch size changes both exposure and the negative pool. With
F8/G32, B16/B32 give 38/70 negatives per retain anchor and 83/99 per forget
anchor. This grid estimates the combined batch effect; it does not identify
coverage separately from negative-bank size. Loss components remain separately
averaged, so doubling retain batch does not double its scalar loss weight.

The bootstrap preserves the actual original server sampler, including any
server-only source differences. The new audit gathers row indices actually
returned to all ranks and records each group's draws, unique rows, coverage
and frequency range. Counts are included in each atomic checkpoint's
`sampling_state.json` and restored with optimizer/RNG state. Retraining lost
steps does not add the abandoned attempt to committed coverage. Raw observation
logs may include uncommitted attempts and are labelled accordingly.

Existing training-only answer-probability, representation and sampled LoRA-B
gradient diagnostics remain enabled at steps 0/25/50/75/100/125 (gradients start
at step 1). Per-step logs also record positive-view cosine/diversity, unique
positive texts per anchor, negative count, step duration and maximum CUDA
allocated/reserved peak across all ranks. Gradients cover sampled tensors,
not the full model. Different batch/view counts have different compute costs.

## Start on ABCI

The account-wide limit is three active jobs/nodes, counting queued/held and
other account jobs. Each allocation is one eight-GPU HF node, reserved queue
R9920261000, project gcg51557, a `0390_` job name and 12-hour walltime.
The campaign's default ten-hour budget includes queue wait. Administrator and
queued-job termination retry after 1,200 seconds without a count limit while
the budget is open. Checkpoint/validation resume is retained. Manual qdel is
intentional cancellation; deterministic failures retain the bounded policy.

The bootstrap writes a new frozen campaign and preserves previous campaigns
and the active server checkout. Preparation is idempotent for the same revision
and profile. Only `start` submits PBS jobs.

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"

git fetch origin codex/0390-conrep-night-20261008
mixed_ref="$(git rev-parse FETCH_HEAD)"
mixed_dir="$(mktemp -d /tmp/0390_conrep_mixed.XXXXXX)"
git show "${mixed_ref}:scripts/abci/0390_prepare_conrep_followup.py" > "$mixed_dir/prepare.py"

campaign="$PWD/results/validated_v2/0390/conrep-mixed-grid-20261009"
python -I "$mixed_dir/prepare.py" \
  --root "$PWD" --ref "$mixed_ref" \
  --source-campaign "$PWD/results/validated_v2/0390/conrep-night-20261008" \
  --campaign "$campaign" --profile mixed-grid --hours 10

entry="$campaign/code/scripts/abci/0390_conrep_night.py"
python -I "$entry" start --campaign "$campaign"
python -I "$entry" status --campaign "$campaign"
qstat -u "$USER"
)
```

Preparation normally prints `tasks=28`, `llama_runs=14`, `gemma_explorations=14`,
`historical_controls_reused=2`, `workers=3`, `pbs_walltime=12:00:00`.
If a control cannot be reused, inspect `grid-design.json` for the reason and
expect one/two extra tasks. Preparation does not load models or establish GPU
success; each allocation performs its own real eight-GPU save/resume smoke.

To inspect current work:

```bash
campaign="$PWD/results/validated_v2/0390/conrep-mixed-grid-20261009"
entry="$campaign/code/scripts/abci/0390_conrep_night.py"
python -I "$entry" status --campaign "$campaign"
tail -n 30 "$campaign/events.jsonl"
```

If the budget ends with work remaining, extend it explicitly; this resumes the
same tasks and checkpoints instead of preparing another campaign:

```bash
python -I "$entry" resume --campaign "$campaign" --hours 10
```

## Collect together with previous results

```bash
git fetch origin codex/0390-conrep-night-20261008
collect_dir="$(mktemp -d /tmp/0390_conrep_collect.XXXXXX)"
git show FETCH_HEAD:scripts/abci/0390_collect_conrep_results.py > "$collect_dir/collect.py"
python -I "$collect_dir/collect.py" --root "$PWD"
```

The collector includes the mixed campaign when present, alongside the original
two campaigns and any standalone Llama campaign. Explicit repeated `--campaign`
arguments override the defaults. Export adds `experiment-configurations.csv`
and `sampling-coverage.csv`; the latter uses checkpoint-committed counts only.
The raw bundle keeps grid-design evidence, configuration/provenance, observations
and actual sampled row indices. It does not average seeds or select checkpoints.

## Verification

CPU tests cover the mixed matrix, dynamic claims without duplicate tasks,
three-node submission/recovery, preserved frozen helpers, historical-control
verification/fallback, K1 equivalence, explicit K4/K8 loss and gradient formulas,
actual Llama/Gemma tiny-model multi-view training/resume, sampling neutrality and
committed result export. Distributed CPU tests skip when the executor denies
Gloo interfaces. Real eight-GPU save/resume smoke remains required on ABCI.
