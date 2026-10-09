# ConRep overnight campaign: three nodes, ten hours

This is an additive entry for the existing ABCI server checkout. It freezes
the server's current `src/` tree and the new launcher, including uncommitted
changes. It does not replace the existing trainer, data builder, evaluator,
parallel launcher, or selection/storage code. Validation calls the frozen
server `experiments.validation.run` directly; no checkpoint selection, pruning,
or final-test evaluation is invoked.

## Install on the login node

The server has newer uncommitted code than the historical branch. **Do not
reset, checkout, or pull that branch over the server working tree.** Fetch the
new branch, then install only the new files with the collision-checking helper:

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"

git fetch origin codex/0390-conrep-night-20261008
night_ref="$(git rev-parse FETCH_HEAD)"
night_install_dir="$(mktemp -d /tmp/0390_conrep_install.XXXXXX)"
night_installer="$night_install_dir/install.py"
git show "$night_ref:scripts/abci/0390_install_conrep_night.py" > "$night_installer"
python -I "$night_installer" --root "$PWD" --ref "$night_ref" \
  --upgrade-from 33d5f9c22106d43bbaa88cc3bc57ee23afbc4ac9

campaign="$PWD/results/validated_v2/0390/conrep-night-20261008-3nodes"
python scripts/abci/0390_conrep_night.py prepare --campaign "$campaign" --hours 10
python scripts/abci/0390_conrep_night.py status --campaign "$campaign"
)
```

The installer uses only the standard library. `python -I` and a separate
temporary directory prevent unrelated `/tmp/math.py` files from shadowing
standard Python modules. The `--upgrade-from` option is also safe for a first
installation: existing package files may only be replaced when their bytes
exactly match that old revision. Replaced files are backed up under
`logs/0390/installations/`. Other server modifications are refused, not overwritten.
For an existing campaign, use the controller-only recovery procedure below.
It preserves its task/checkpoint identity and applies the current three-node
account cap. Installing checkout files alone does not update a frozen controller.

The commands above prepare the plan without submitting jobs. Start a new
campaign with:

```bash
python scripts/abci/0390_conrep_night.py start --campaign "$PWD/results/validated_v2/0390/conrep-night-20261008-3nodes"
```

The defaults read the two **existing server-resolved** configs:

- `results/validated_v2/0390/final-unlearn-v5-parallel3-seed42/gemma2_9b/config.json`
- `results/validated_v2/0390/final-unlearn-v5-parallel3-seed42/llama8b/config.json`

Override `--gemma-config` / `--llama-config` if those configurations have moved.
The required setup is the approved full SFT checkpoint, global batches 8/16/32,
`specified_positive=dropout`, and the full validation partition. Prepared data,
probe files, the source snapshot, and generated configs are fingerprinted.
SFT asset filenames, sizes and modification times are also checked before each
allocation; this is a metadata guard, not a claim of hashing every model shard.
An existing campaign is never overwritten; repeat `start` to adopt an existing
supervisor, or use explicit `resume` after it has stopped.

No jobs are submitted by `prepare`. It writes `positive-audit.json`, including
30 example original/positive pairs and actual augmentation coverage. If no
safe word deletion is possible in either model's retain data, preparation
fails before submitting anything. This avoids silently running a purported
positive ablation that never changes the input.

## Matrix and order

All runs use 125 optimizer steps, checkpoints 10,20,...,120,125, the fixed SFT
checkpoint, the existing data/validation protocol, and general Wiki LM weight 1.
Every checkpoint is retained and validated. All new training starts from SFT;
only retries of the *same* experiment resume an unlearning checkpoint.

| ID | Difference from A unless stated |
|---|---|
| A | rank32/alpha64/dropout.05, forget weight1, views4, corruption.7, both LM weights1 |
| B | forget weight5 |
| C | protected specified-retain positive |
| D | B + protected specified-retain positive |
| E | forget weight2 |
| F | B + views8; other-instance negative-view budget stays4 |
| G | B + specified LM0 |
| H | D + specified LM0 |
| I | B + global LoRA dropout.10 |
| J | B + rank64/alpha128 |
| K | B + rank128/alpha256 |
| L | B + forget corruption.50 |

The 32 task entries are ordered as Gemma A-D/42, E-H/42, I-L/42, A-H/43,
A-D/44, Llama A-D/42, A-D/43. The three workers prefer A→D→G→J,
B→E→H→K, or C→F→I→L; an idle worker can claim another pending entry.
Priority wins over preference: after A/B/C start, the first available worker
takes D before moving to E-H. More seeds are unlearning repeats with the same
SFT checkpoint, not repeated SFT training. A is intentionally rerun because the
new execution path must have a contemporary control.

The submission plan has exactly three persistent worker slots. Each allocation
uses one HF node (eight GPUs) and runs training plus checkpoint validation
serially across multiple experiments:

| Job slot | First fresh experiment | Same-priority preferences |
|---|---|---|
| 0 | Gemma A, seed42: control | A, D, G, J |
| 1 | Gemma B, seed42: forget CL weight5 | B, E, H, K |
| 2 | Gemma C, seed42: protected retain positive | C, F, I, L |

D (weight5 plus protected positive) goes to the first free slot after A/B/C
are claimed. Paused experiments take precedence, then the original task-group
priority, then slot preference. This is a shared queue: a slow experiment does
not force another node to idle. Checkpoint identities and the 32-task pool are
unchanged. Retries receive new PBS IDs but reuse these same three slots.

Protected positives are a conservative, reproducible **single article deletion**
before the value-introducing boundary. Supplied protected spans and entity,
attribute and value fields are preserved. All text from the first is/has/colon
boundary onward is protected, including values, negation, temporal qualifiers
and units. A plain A is not deleted as though it were an article. Unknown
structures are unchanged. Eligible positives are selected with probability .5,
and only one view is used. No validation/test labels are used to construct them.
This deliberately tests a light valid augmentation; it is not arbitrary token
replacement or proof that stronger augmentation has been exhausted. Wiki's
positive construction stays unchanged. Training logs record changed fraction
and positive cosine similarity so low effective coverage is visible.

## Scheduling and stopping

Each PBS allocation requests `-P gcg51557`, `-q R9920261000`,
`RTYPE=rt_HF`, `select=1`, twelve hours (`walltime=12:00:00`), and a `0390_` job name. Three slots are
available, with **all active/queued/held account jobs and requested nodes**
counted. For example, two existing single-node jobs leave one slot; one
three-node job leaves none. Both new and recovered campaigns use worker IDs
0, 1 and 2. The upgrader moves the old fourth worker into `retired_workers`,
retaining its PBS evidence and making its interrupted task available to the
three remaining workers after reconciliation and the retry wait. It cannot
submit another fourth-worker job. Existing jobs are not
cancelled. This launcher uses the existing
single-job submitter's submission lock. Independently running external
submitters must also respect the shared account limit.

The detached login-node supervisor checks every 60 seconds. A worker trains one
configuration on eight GPUs, then uses eight independent GPU processes for its
13 checkpoint validations. A finished validation worker immediately takes the
next checkpoint, without a wave barrier. Completed experiments are skipped.
There is no dependence on a ChatGPT window remaining open.

The ten-hour deadline starts after the first `start` successfully checks the
scheduler, including subsequent queue wait. Failed scheduler access during
startup does not consume the budget. Accepted jobs temporarily missing from
`qstat` continue to reserve their account slots until PBS confirms termination. The
initial per-run estimate is one hour and is updated conservatively using
completed runs. New runs need time for training and validation; after the last
90 minutes only the highest-priority fresh runs or unfinished validation are
eligible. Workers pause before the shorter of PBS allocation end and campaign
deadline. A new twelve-hour allocation can continue the same task list. At the
campaign deadline remaining owned PBS jobs are cancelled; no new work is added.
The PBS walltime and campaign budget are independent: `--hours 10` still sets a
ten-hour campaign, even though each PBS request allows twelve hours. Changing
walltime does not extend the campaign deadline. Use an explicit `--hours 12`
on recovery to grant a fresh twelve-hour campaign budget when intended.
The three-node, ten-hour budget is at most 30 node-hours (240 GPU-hours),
including startup, validation and recovery. The 32 experiments remain a
prioritized task pool; at the initial one-node-hour estimate they exceed this
budget. Later seed/model repeats run only when measured throughput leaves
enough time. Unfinished entries remain recorded for an explicit later resume.

Each allocation first runs a real two-step, eight-GPU save/resume smoke: stop
after step1, load its checkpoint, finish step2. Failure blocks that worker and
is not hidden by changing training parameters. This checks the actual server
CUDA/NCCL/model path before full experiments. Filesystem free-space reserve is
100 GB by default (`prepare --reserve-gb`); this is not a filesystem quota
measurement. Check project quota before a large campaign. No checkpoints are
automatically deleted to make room.

## Recovery and traceability

- A checkpoint is assembled in an incomplete directory. Only after all ranks
  save RNG state and synchronize is its completion marker published and the
  directory renamed. The model/adapter, optimizer, scheduler, step, CPU/CUDA,
  NumPy/Python and separate corruption/positive RNG states are restored.
- Resuming requires the same training fingerprint and world size. The planned
  max_steps and learning-rate schedule are not shortened for a pause. A
  `PAUSED.json` marker distinguishes a planned pause from success or a crash;
  `TRAINING_COMPLETE.json` appears only at the real last optimizer step.
- Forced interruption falls back to the latest complete checkpoint. Incomplete
  writes do not count as candidates. Extra checkpoints saved solely at a pause
  are recovery snapshots; the common 13-point validation schedule is unchanged.
- Validation resumes at checkpoint granularity. A valid completion includes the
  metrics and raw prediction files. An unfinished checkpoint is reevaluated;
  there is no claim of per-example inference resume.
- Confirmed PBS walltime/preemption/node failure can retry. The approved
  recovery policy also retries administrator termination, which is an expected
  interruption in this environment. A terminal queued job with no start/exit
  record and a retained placement comment followed by `terminated` gets the
  same retry schedule without changing its reservation or resource request.
  An active queued placement comment is not treated as a failed allocation.
- Administrator and terminated-before-start interruptions wait **20 minutes
  from each allocation's termination**, with **no retry-count limit**. This
  continues while the campaign budget admits work, until its deadline or an
  explicit stop. The same waiting time follows the interrupted task so another
  slot cannot bypass it. Polling or lack of a free account slot can delay the
  next submission; an already queued job is not resubmitted every 20 minutes.
- Other confirmed scheduler failures retain their separate bounded budget
  (three retries, with 60/120/240-second allocation backoff). Administrator
  interruptions do not consume that budget. All retry counts remain recorded.
  Manual user cancellation, explicit stop, access
  denial and deterministic training failures are not automatically revived.
  Exit143 or Exit271 alone is not sufficient to infer the reason. Unknown
  scheduler state and ambiguous qsub responses never create duplicate jobs.
- A replacement worker prioritizes paused tasks and loads the latest complete
  checkpoint of the same experiment. Finished validations remain reusable.
  Events and PBS reasons are printed to the supervisor/worker console as well
  as recorded in JSON; exhausted or unclassified failures produce `BLOCKED.json`.
- A paused task remains unclaimable until PBS confirms its previous allocation's
  ending reason, including a graceful checkpoint saved just before qdel.
- An interrupted supervisor adopts recorded PBS IDs. A unique job name helps
  recover a qsub response lost before its job ID was recorded. If no reliable
  scheduler evidence exists, it waits rather than risk duplicate training.
- Per-job logs use `logs/0390/runs/<jobID>/`, with submitted PBS, environment,
  launcher exit, final PBS state, experiment configs, training logs and metrics.
  The campaign event log maps multiple job IDs back to one experiment.

## Inspect, stop, resume

Run these from the existing server checkout after activating its environment:

```bash
campaign="$PWD/results/validated_v2/0390/conrep-night-20261008-3nodes"
python scripts/abci/0390_conrep_night.py status --campaign "$campaign"
tail -n 40 "$campaign/supervisor.log"
tail -n 40 "$campaign/events.jsonl"
qstat -u "$USER"
```

`status` also shows each worker's PBS ID, status and final outcome, and regenerates
`checkpoint-results.csv` and `summary.json`. A progress summary is printed to
`supervisor.log` every five minutes. The CSV
keeps every configuration/seed/checkpoint and the evaluator's original metrics
and scalar MMLU diagnostics; it does not choose a winner. Review Llama's raw
MMLU outputs and strict/main discrepancy before drawing cross-model utility
conclusions. This package does not silently replace the established parser.

```bash
# Graceful stop; no automatic restart. Add --cancel-jobs for immediate PBS cancellation.
python scripts/abci/0390_conrep_night.py stop --campaign "$campaign"

# Explicitly grant another four hours and resume unfinished work.
python scripts/abci/0390_conrep_night.py resume --campaign "$campaign" --hours 4
```

Cancelled or deterministically failed experiments are preserved for inspection,
not reset automatically. Changed code/configuration needs a new campaign.

## Recover a campaign prepared with the original launcher

The original launcher classified administrator interruption as unknown and
left its workers blocked. The previous repair still imposed three retries;
this revision removes that limit for the two expected interruption types,
migrates old four-worker campaigns to exactly three slots, and sets new PBS
requests to twelve hours. Fetch the updated branch and run the dedicated
upgrader instead of rerunning `prepare` or the additive installer:

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"
git fetch origin codex/0390-conrep-night-20261008
recovery_ref="$(git rev-parse FETCH_HEAD)"
recovery_dir="$(mktemp -d /tmp/0390_conrep_recovery.XXXXXX)"
git show "${recovery_ref}:scripts/abci/0390_recover_conrep_night.py" > "$recovery_dir/recover.py"
campaign="$PWD/results/validated_v2/0390/conrep-night-20261008"
python -I "$recovery_dir/recover.py" --root "$PWD" --ref "$recovery_ref" --campaign "$campaign" --restart
)
```

This preserves the original deadline. To explicitly grant a fresh time budget,
append `--hours 10` to the recovery command. Without `--restart`, the upgrader
installs the controller and migrates scheduler metadata without submitting jobs
or changing experiment progress. An old fourth worker is retained as history.

The upgrader refuses active campaign allocations or edited package-owned
launcher files. It backs up the existing plan/state and the three launcher
files it replaces. The original `code/` snapshot, source hash, task configs,
data, and checkpoint identities are unchanged. The new controller lives in
`controllers/<commit>/` with its own hashes; model training/validation still
use the original frozen entry. Normal checkout `status/start/resume/stop`
commands dispatch to that controller. Repeating the same upgrade is harmless.

Recovery reassesses the old final PBS records under the approved policy once,
including tasks blocked only by the previous administrator-interruption retry limit,
reopens only recoverable workers/tasks, and keeps completed results, manual
cancellations, failure evidence, and retry counters. It does not switch queue,
alter `node_group`, change batch sizes, or relabel a failed training run as done.

If you deliberately cancelled jobs to change the allocation settings, first
stop the old supervisor and then explicitly name those finished PBS IDs with
`--requeue-jobs`. For example, after fetching and extracting the upgrader as
above, replace the three example IDs with the actual cancelled allocation IDs:

```bash
python scripts/abci/0390_conrep_night.py stop --campaign "$campaign"
python -I "$recovery_dir/recover.py" \
  --root "$PWD" --ref "$recovery_ref" --campaign "$campaign" \
  --restart --requeue-jobs 1234561.pbs1 1234562.pbs1 1234563.pbs1
```

This waits up to 90 seconds for the stopped supervisor to release its lock.
Only the named, confirmed-finished campaign allocations are reopened; active
jobs and allocations blocked by other failures are rejected. Other cancelled
experiments, completed results, deterministic training failures, and the
existing deadline are preserved. `explicit_requeues` records the previous PBS
outcome and comment. This explicit action does not change automatic handling
of future manual cancellations. Add `--hours` only to grant a fresh time budget.

## Verification boundary

CPU tests use real tiny Llama and Gemma2 models and compare uninterrupted versus
paused/resumed final LoRA tensors exactly. Other tests exercise objective and
gradient equivalence, fixed negative budgets, protected values, configuration
freezing, account/node limits, duplicate submissions, walltime recovery,
manual cancellation, deadline admission and collision-safe installation.
The two-process Gloo test can be explicitly skipped only when the executor
denies network-interface access; that is not a distributed-training pass.
Full-backbone ABCI/NCCL execution remains a server check, enforced by the real
GPU smoke before full runs.

```bash
PYTHONPATH=src python -m pytest -q tests/0390/test_conrep_night.py
```
