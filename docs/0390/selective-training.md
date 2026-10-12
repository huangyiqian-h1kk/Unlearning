# Selective unlearning: runnable campaign

The training implementation is available on `codex/0390-selective-training-20261011`.

The October 12 [evaluation repair](selective-evaluation-repair.md) adds binary
gender-MCQ scoring to existing frozen campaigns and recovers affected evaluation
without retraining. It also records bounded continuation of incomplete ReLearn
preparation; completed calibration remains unchanged.

The optional [T0 general-only supplement](t0-general-only.md) adds 14 separate
trainings without changing this campaign's scientific snapshot or running jobs.
It freezes the server's existing Llama positive-completion campaign and overlays the new
controller, selective adapters, and checkpoint support. It does not check out a branch,
change the environment, or modify the running Qwen/SFT campaign.

## Admission and ordering

- Shared PBS submission lock; account-wide limit of three active jobs and three nodes.
  Queued, held, running, transitioning and exiting jobs all consume capacity.
  The separately authorized T0-General-only 3+1 mode allows exactly one additional
  companion allocation (four account jobs/nodes total); see its linked launch instructions.
- With one SFT allocation, at most two selective allocations are submitted. After it
  finishes, another selective worker is admitted automatically when work is ready.
- The existing Qwen campaign reserves one slot while armed or waiting for a retry.
  The supervisor polls every 30 seconds and survives a normal shell logout.
  A login-node reboot requires rerunning `resume`; no external service is assumed.
- Every selective worker requests one `rt_HF` node, eight GPUs, 12 hours,
  project `gcg51557`, queue `R9920261000`. Workers serially claim experiments and
  use the same allocation for training, calibration, and per-GPU evaluation.
- Fixed historical recipes come first: 180 main trainings have priority over the
  116 seed/request/mechanism trainings, with ready work filling idle workers.
  Only when their training **and evaluation** are complete
  are grids appended: 32 learning-rate, six forget-weight, four retain-LM runs.
  Total: 338 trainings, plus shared SFT evaluation and up to two historical-weight
  C0 bridges. These are not 338 simultaneous PBS jobs.
- A failed baseline does not block other fixed experiments. It remains visibly
  failed and prevents opening the grid gate. Plain `G256B32W2P4` is excluded;
  the previously agreed `P4C70V4IB2N20` insertion candidate remains.
- Reference evaluation and fixed training may run concurrently. Within fixed work,
  stages 0/5/10/20/30/40 are priorities rather than global barriers. A finished
  training whose model's reference is incomplete stays `pending` with
  `waiting_for_reference=true`; its weights are kept and the worker takes another
  task. When that reference completes, evaluation resumes without retraining.
  Failed references block new work for that model until explicitly repaired.

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

## Repair the initial MMLU reference reader failure

The first deployment's two SFT references completed validation but failed during
test MMLU overlap checking with `JSONDecodeError: Unterminated string`.
The selective JSONL reader used `str.splitlines()`, which also splits Unicode
U+0085/U+2028/U+2029 inside JSON strings. It now delegates to the same `read_rows`
function that the frozen MMLU evaluator uses. Genuine malformed data and overlap
are still errors; no question, answer, split, parser or generation setting changes.

The correct reader was already present in `config.py` in the original 0390
framework and used by both the frozen v5 MMLU scorer and its overlap check. The
selective wrapper introduced a duplicate reader, causing this regression. The
earlier tiny-model evaluation integration fixture disabled both MMLU paths, so
that test did not validate the failing branch. Reader regressions now include
Unicode separators and preserve rejection of genuinely overlapping partitions.
The CPU design planner also imports the shared reader. This last change applies
to future preparation; the stage-zero repair only changes `selective_data.py` in
the existing frozen campaign and leaves its already-prepared data intact.

For a campaign blocked at the two reference tasks, run the following once. The
repair refuses active selective allocations or any already-started downstream
task. It verifies the old source and inputs, checks both MMLU sets without a GPU,
and only permits the exact reviewed reader edit. It saves before/after files in
`repairs/mmlu-reader-v1`, updates source/config/training provenance, and preserves
the original state and every prediction byte. The evaluation implementation and
its signature do not change, so validated reference partitions are reused and
test resumes from cached generations. No validation marker is synthesized.
An interrupted repair can be rerun from the same command.

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"
sel_campaign="$PWD/results/validated_v2/0390/selective-extensive-20261011"
git fetch origin codex/0390-selective-training-20261011
sel_fix_ref="$(git rev-parse FETCH_HEAD)"
sel_fix_dir="$(mktemp -d /tmp/0390_selective_fix.XXXXXX)"
git show "${sel_fix_ref}:scripts/abci/0390_repair_selective_mmlu.py" > "$sel_fix_dir/repair.py"
python -I "$sel_fix_dir/repair.py" \
  --root "$PWD" --campaign "$sel_campaign" --ref "$sel_fix_ref"
python -I "$sel_campaign/code/scripts/abci/0390_selective.py" resume \
  --campaign "$sel_campaign" --retry-failed
python -I "$sel_campaign/code/scripts/abci/0390_selective.py" status \
  --campaign "$sel_campaign"
qstat
)
```

After this repair, use the frozen `status`/`resume`/`export` entry points below;
do not rerun the initial bootstrap, whose original provenance deliberately differs.
The Qwen campaign is unaffected. An empty `qstat` does not establish SFT success;
inspect its own final state and PBS history independently.

## Upgrade a running campaign to dependency scheduling

The first deployment admitted only the two model-reference jobs, and then only
the two historical bridge evaluations, even after SFT released the third slot.
This update removes that unintended fixed-stage barrier. It uses the same shared
account lock, three-job/three-node limit, queue, walltime, and recovery policy.

The upgrade freezes a separate controller under `controllers/`. It does not edit
the original frozen code, scientific configs, training identities, checkpoints,
or evaluation caches. Every file outside the two dispatch modules must match the
original source manifest; the calibration/smoke/training block must also have an
identical AST. The original scientific source hash and the controller hash are
recorded separately in `plan.json`.

The old supervisor exits at its next poll via a temporary campaign handoff state.
No STOP signal or qdel is sent to workers. Current children finish their work;
queued/running old workers remain accounted for and can share the task queue with
the new controller. New allocations use the new worker entry. A task that finishes
during handoff is retained, never reset to pending. The handoff may take about one
poll interval (30 seconds); the script waits for it and starts the new supervisor.

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"
sel_campaign="$PWD/results/validated_v2/0390/selective-extensive-20261011"
git fetch origin codex/0390-selective-training-20261011
sel_sched_ref="$(git rev-parse FETCH_HEAD)"
sel_sched_dir="$(mktemp -d /tmp/0390_selective_scheduler.XXXXXX)"
git show "${sel_sched_ref}:scripts/abci/0390_upgrade_selective_scheduler.py" > "$sel_sched_dir/upgrade.py"
python -I "$sel_sched_dir/upgrade.py" \
  --root "$PWD" --campaign "$sel_campaign" --ref "$sel_sched_ref"
sel_entry="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["entry"])' "$sel_campaign/plan.json")"
python -I "$sel_entry" status --campaign "$sel_campaign"
qstat
)
```

With two selective allocations and a completed SFT campaign, the next dispatch tick
admits the third allocation without waiting for the references or historical bridges.
Actual start time still belongs to PBS. `dispatch.policy` should be
`selective-dependencies-v2`; status also reports ready work, reference status,
waiting evaluations, account nodes and SFT peer reservations. Repeating the same
upgrade reuses the controller and never duplicates submissions.

## Status, resume and export

Each block is independently copyable.

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"
sel_campaign="$PWD/results/validated_v2/0390/selective-extensive-20261011"
sel_entry="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["entry"])' "$sel_campaign/plan.json")"
python -I "$sel_entry" status --campaign "$sel_campaign"
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
sel_entry="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["entry"])' "$sel_campaign/plan.json")"
python -I "$sel_entry" resume --campaign "$sel_campaign"
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
sel_entry="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["entry"])' "$sel_campaign/plan.json")"
python -I "$sel_entry" export \
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

Dependency dispatch regressions additionally cover the real 300-task topology,
two existing queued allocations plus exactly one new allocation, reference/model
dependencies, the full fixed/grid completion gate, and deferred evaluation without
retraining. A concurrent handoff test preserves a task completed by an old process
during supervisor replacement. Scheduler tests mock PBS and GPU work; they do not
claim to measure ABCI runtime or eight-GPU performance.

This environment has no ABCI/GPU access and disallows the sockets needed for a local
multi-process Gloo test. Actual eight-GPU execution is therefore **not claimed as tested**.
The first use of every model/recipe runs an automatic two-step save/resume smoke inside
its PBS allocation; failure blocks that task's production training. This implementation
has not submitted any ABCI jobs from the development environment.
