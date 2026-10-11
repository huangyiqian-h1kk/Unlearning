# Selective unlearning: runnable campaign

The training implementation is available on `codex/0390-selective-training-20261011`.
It freezes the server's existing Llama positive-completion campaign and overlays the new
controller, selective adapters, and checkpoint support. It does not check out a branch,
change the environment, or modify the running Qwen/SFT campaign.

## Admission and ordering

- Shared PBS submission lock; account-wide limit of three active jobs and three nodes.
  Queued, held, running, transitioning and exiting jobs all consume capacity.
- With one SFT allocation, at most two selective allocations are submitted. After it
  finishes, another selective worker is admitted automatically when work is ready.
- The existing Qwen campaign reserves one slot while armed or waiting for a retry.
  The supervisor polls every 30 seconds and survives a normal shell logout.
  A login-node reboot requires rerunning `resume`; no external service is assumed.
- Every selective worker requests one `rt_HF` node, eight GPUs, 12 hours,
  project `gcg51557`, queue `R9920261000`. Workers serially claim experiments and
  use the same allocation for training, calibration, and per-GPU evaluation.
- Fixed historical recipes come first: 180 main trainings, then 116 seed/request/
  mechanism trainings. Only when their training **and evaluation** are complete
  are grids appended: 32 learning-rate, six forget-weight, four retain-LM runs.
  Total: 338 trainings, plus shared SFT evaluation and up to two historical-weight
  C0 bridges. These are not 338 simultaneous PBS jobs.
- A failed baseline does not block other fixed experiments. It remains visibly
  failed and prevents opening the grid gate. Plain `G256B32W2P4` is excluded;
  the previously agreed `P4C70V4IB2N20` insertion candidate remains.

## Training and evaluation

Historical resolved configurations are loaded from the server, including each
baseline's native parameterization and optimizer. ConRep's objective AST must match
the frozen parent's objective before the bootstrap permits an overlay.
No historical checkpoint is silently retrained. Missing historical C0 bridge weights
are reported in `reuse-ledger.json`; the new selective tasks can still run.

Tasks cover T0-C100, T1–T8, E2, the agreed seed/request repeats, C50 and retain-CL-off.
Global forget batch is eight with one accumulation step. The maximum budgets
125/250/375/750 are ten sampled equivalent passes for F100/200/300/600.
Thirteen validation checkpoints scale with the forget-set size.

Primary generation uses three ID-only prompt families per fact in each partition.
Safe-background diagnostics exclude all requested forget facts and the queried fact.
Gender MCQs are binary and reported separately; an empty same-patient retain set is
`null`, not zero accuracy. Typed scoring and MMLU decoding reuse the frozen v5 code.
The SFT reference covers all 2,925 facts, once per model and partition.

All fixed runs evaluate the endpoint and validation-selected matched-forgetting
checkpoints on test. The 0–100% ceilings, in 5 percentage point increments, and
selected weights are frozen before test evaluation. Grid runs use validation only.
Historical-v5 scores remain a separate protocol; C0 weights, when available, are
reevaluated under selective-v1 without replacing the original archived v5 scores.

LoRA checkpoints and native updated-weight checkpoints save optimizer/scheduler,
per-rank RNG and sampling coverage. Native checkpoints also preserve non-persistent
buffers. All weight snapshots are retained; only this new campaign's optimizer/RNG
archives older than its latest two resumable checkpoints are pruned.
Completed validation/raw predictions are reusable after interruption.

Before claiming a task, workers reserve a conservative estimate for its checkpoints
and three optimizer archives, other workers' reservations, and 100 GB headroom.
Insufficient capacity produces `storage_paused`; no old campaign files are deleted.
This extensive campaign may need additional storage beyond the prior experiments.

Confirmed administrative termination retries after 20 minutes. Other recoverable PBS
failures use the existing bounded backoff policy. Manual `qdel` does not automatically
restart that job; ordinary training failures remain visible for inspection.

## Launch

Run from an ABCI login shell. The first command can prepare and arm while SFT is running.

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"
sel_base="$PWD/results/validated_v2/0390"
sel_campaign="$sel_base/selective-extensive-20261011"
git fetch origin codex/0390-selective-training-20261011
if [ -f "$sel_campaign/PREPARING.json" ]; then
  sel_ref="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["ref"])' "$sel_campaign/PREPARING.json")"
else
  sel_ref="$(git rev-parse FETCH_HEAD)"
fi
sel_boot="$(mktemp -d /tmp/0390_selective_boot.XXXXXX)"
git show "${sel_ref}:scripts/abci/0390_prepare_selective.py" > "$sel_boot/prepare.py"
python -I "$sel_boot/prepare.py" \
  --root "$PWD" \
  --parent "$sel_base/conrep-llama-positive-completion-20261010" \
  --campaign "$sel_campaign" --ref "$sel_ref"
python -I "$sel_campaign/code/scripts/abci/0390_selective.py" prepare --campaign "$sel_campaign"
python -I "$sel_campaign/code/scripts/abci/0390_selective.py" arm --campaign "$sel_campaign"
)
```

`arm` is idempotent. It starts a detached supervisor; it does not promise immediate
PBS allocation. The scheduler decides when a submitted job actually starts.

## Status, resume and export

Each block is independently copyable.

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"
sel_campaign="$PWD/results/validated_v2/0390/selective-extensive-20261011"
python -I "$sel_campaign/code/scripts/abci/0390_selective.py" status --campaign "$sel_campaign"
qstat
tail -n 30 "$sel_campaign/events.jsonl"
)
```

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"
sel_campaign="$PWD/results/validated_v2/0390/selective-extensive-20261011"
python -I "$sel_campaign/code/scripts/abci/0390_selective.py" resume --campaign "$sel_campaign"
)
```

The same entry supports `stop` for a cooperative pause of selective workers.
After inspecting a training failure, explicitly use `resume --retry-failed` to retry
those tasks. A manually cancelled allocation requires `resume --requeue-job JOB_ID`;
the ID must belong to this campaign and be confirmed finished. These options require
the supervisor to have stopped. They never act on the SFT job.

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"
sel_campaign="$PWD/results/validated_v2/0390/selective-extensive-20261011"
python -I "$sel_campaign/code/scripts/abci/0390_selective.py" export \
  --campaign "$sel_campaign" --include-predictions
)
```

Export works on partial campaigns and writes under the campaign's `exports/` directory.
`summary.zip` contains metrics/configs/hashes, task and seed summaries, matched-forgetting
selections, sampling states, calibration audits and failure states. Raw generations are
in a separate `predictions.zip`; neither archive contains model weights or optimizer state.
Omit `--include-predictions` for the small summary only. Use the printed group-filesystem
paths for SFTP; do not try to download a different login node's `/tmp` directory.

Patient bootstrap intervals in the export are explicitly single-training-seed descriptive
intervals. Full paper-level multi-seed inference and multiplicity corrections must be
performed when the complete results are reviewed; the exporter makes no significance claim.

## Verification boundary

Local tests cover account capacity 2→3, ambiguous submission, peer retry reservations,
fixed-first/grid counts, cancellation reconciliation, validation-only selection, partial
prediction recovery, and exact tiny-model interrupted/resumed weights for all seven
methods on Llama and Gemma architectures. Real rich data construction and the frozen-v5
generation/scoring/reference integration were exercised locally. Existing campaign
recovery tests also pass.

This environment has no ABCI/GPU access and disallows the sockets needed for a local
multi-process Gloo test. Actual eight-GPU execution is therefore **not claimed as tested**.
The first use of every model/recipe runs an automatic two-step save/resume smoke inside
its PBS allocation; failure blocks that task's production training. This implementation
has not submitted any ABCI jobs from the development environment.
