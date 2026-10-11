# T0 general-only control

This supplement adds 14 trainings: Gemma and Llama, one frozen ConRep recipe per
model and all six baselines (NPO, RMU, FALCON, LUNAR, SAGO, ReLearn), seed 42.
ConRep inherits Gemma `G256B32W2P1C70V4N10` and Llama
`G256B32W2P1C50V8N00`. Specified-positive settings become inactive; these labels
identify the historical parent recipe, not a positive branch still being trained.

Original T0 forget bytes/order (100 facts) and general Wiki bytes are preserved.
The F/G sampler retains its original group seed offsets, global F batch 8, 125
steps and accumulation 1. General batch, loss weights, LR, optimizer and schedule
are inherited per method. All specified weights and the specified batch are zero.
There is no specified training file or request file. In ConRep, specified anchors,
positives **and forget negatives** disappear. FALCON calibrates layers using only
F/G. LUNAR fits F/G only; ReLearn augmentation joins the original `forget:i` IDs
through an audited mapping from selective relation IDs.

The original T0-C0 probes are copied unchanged: F100, R-same200, R-other900 and
the same MMLU partitions. R-same/R-other remain evaluation-only in this supplement.
Completed shared SFT references are verified and copied; they are never retrained
or regenerated concurrently with the parent. The 13 validation checkpoints and
validation-selected/terminal test evaluation follow the existing selective protocol.
No grid or additional seeds are implicitly added.

This is an ablation matched for forget exposure, **not matched compute**.
Removing specified data also removes computation and changes the objective/negative
pool. Logs report F/specified draws, canonical input token counts, per-step elapsed
time and peak GPU memory; calibration time is separate. Canonical tokens exclude
repeated contrastive views and are not FLOPs. Do not attribute differences solely
to data locality or claim equal-compute superiority from this control. The export
includes `training-cost.json` with logged training-attempt costs, separate from
evaluation and model-loading time.

The parent is not modified scientifically: source, task configs, identities,
checkpoints and running children remain intact. Only its operational `peers`
list gains this supplement. While the parent is unfinished, the supplement runs
one worker and reserves the next free account slot; the parent can use the other
two. Existing three allocations are never interrupted to create that slot, so
admission may wait until an existing allocation ends. Once the parent is complete,
the supplement may use up to three workers. Shared submission locks, queued jobs,
unseen/ambiguous qsub records and node counts enforce the total cap of three.
Attachment waits for the existing supervisor to reload the peer reservation
(normally two polling cycles); it cannot race a dispatch cycle holding the old plan.
The existing 20-minute administrator-kill recovery policy is inherited.
On completion, STOP or a blocked supplement, the reservation is released.

Launch (safe to repeat):

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"
go_base="$PWD/results/validated_v2/0390"
go_parent="$go_base/selective-extensive-20261011"
go_campaign="$go_base/selective-t0-general-only-20261012"
git fetch origin codex/0390-selective-training-20261011
go_ref="$(git rev-parse FETCH_HEAD)"
if [ -f "$go_campaign/PREPARING.json" ]; then
  go_ref="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["ref"])' "$go_campaign/PREPARING.json")"
fi
go_boot="$(mktemp -d /tmp/0390_general_only.XXXXXX)"
git show "${go_ref}:scripts/abci/0390_prepare_t0_general_only.py" > "$go_boot/prepare.py"
python -I "$go_boot/prepare.py" --root "$PWD" --parent "$go_parent" \
  --campaign "$go_campaign" --ref "$go_ref"
python -I "$go_campaign/code/scripts/abci/0390_t0_general_only.py" prepare --campaign "$go_campaign"
python -I "$go_campaign/code/scripts/abci/0390_t0_general_only.py" arm --campaign "$go_campaign"
)
```

Status:

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"
go_campaign="$PWD/results/validated_v2/0390/selective-t0-general-only-20261012"
python -I "$go_campaign/code/scripts/abci/0390_t0_general_only.py" status --campaign "$go_campaign"
qstat
)
```

Resume after a cooperative stop or blocked supervisor, retrying failed tasks:

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"
go_campaign="$PWD/results/validated_v2/0390/selective-t0-general-only-20261012"
python -I "$go_campaign/code/scripts/abci/0390_t0_general_only.py" resume \
  --campaign "$go_campaign" --retry-failed
)
```

Normal administrator-kill continuation is automatic; a user-cancelled allocation
still needs the existing explicit `--requeue-job JOB_ID` recovery mechanism.
Do not run `--retry-failed` against an already-running supervisor.

Export, including in-progress results, with large predictions in a separate ZIP:

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"
go_campaign="$PWD/results/validated_v2/0390/selective-t0-general-only-20261012"
python -I "$go_campaign/code/scripts/abci/0390_t0_general_only.py" export \
  --campaign "$go_campaign" --include-predictions
)
```

Export the parent campaign separately for its T0-C100 results. Historical T0-C0
v5 scores require same-protocol reevaluation before combining them with this table;
the existing parent's optional two C0 bridges are not all 14 matched controls.

Validation: tests include real tiny Llama/Gemma training and exact interrupted
resume for all seven methods with no specified file, FALCON calibration without
specified data, ConRep general-only negatives, stable F/G sampling, legacy ReLearn
IDs, idempotent peer attachment and one-to-three admission. These CPU checks do
not constitute an ABCI production run. Every model/method also runs the established
two-step 8-GPU interrupted/resumed smoke in its actual PBS allocation.
