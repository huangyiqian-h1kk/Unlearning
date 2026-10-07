# ConRep overnight campaign: four nodes, ten hours

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
night_installer="$(mktemp /tmp/0390_conrep_install_XXXXXX.py)"
git show "$night_ref:scripts/abci/0390_install_conrep_night.py" > "$night_installer"
python "$night_installer" --root "$PWD" --ref "$night_ref"

campaign="$PWD/results/validated_v2/0390/conrep-night-20261008"
python scripts/abci/0390_conrep_night.py prepare --campaign "$campaign" --hours 10
python scripts/abci/0390_conrep_night.py start --campaign "$campaign"
python scripts/abci/0390_conrep_night.py status --campaign "$campaign"
)
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
A-D/44, Llama A-D/42, A-D/43. Each worker initially prefers A→G→J,
B→E→K, C→H→I, or D→F→L; an idle worker can claim another pending entry.
Priority wins over preference. More seeds are unlearning repeats with the same
SFT checkpoint, not repeated SFT training. A is intentionally rerun because the
new execution path must have a contemporary control.

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
`RTYPE=rt_HF`, `select=1`, six hours, and a `0390` job name. Four slots are
available, with **all active/queued/held account jobs and requested nodes**
counted. Existing jobs are not cancelled. This launcher uses the existing
single-job submitter's submission lock. Independently running external
submitters must also respect the shared account limit.

The detached login-node supervisor checks every 60 seconds. A worker trains one
configuration on eight GPUs, then uses eight independent GPU processes for its
13 checkpoint validations. A finished validation worker immediately takes the
next checkpoint, without a wave barrier. Completed experiments are skipped.
There is no dependence on a ChatGPT window remaining open.

The ten-hour deadline starts at the first `start`, including queue wait. The
initial per-run estimate is one hour and is updated conservatively using
completed runs. New runs need time for training and validation; after the last
90 minutes only the highest-priority fresh runs or unfinished validation are
eligible. Workers pause before the shorter of PBS allocation end and campaign
deadline. A new six-hour allocation can continue the same task list. At the
campaign deadline remaining owned PBS jobs are cancelled; no new work is added.
32 is a task-pool target, not a promise of throughput.

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
- Confirmed PBS walltime/preemption/node failure can retry, at most three
  failure attempts. Manual qdel is never automatically revived. Exit143 alone
  is not enough to infer walltime. Unknown scheduler state, ambiguous qsub
  responses, deterministic configuration failures, OOM and NaN do not cause
  blind submissions or hidden hyperparameter changes.
- An interrupted supervisor adopts recorded PBS IDs. A unique job name helps
  recover a qsub response lost before its job ID was recorded. If no reliable
  scheduler evidence exists, it waits rather than risk duplicate training.
- Per-job logs use `logs/0390/runs/<jobID>/`, with submitted PBS, environment,
  launcher exit, final PBS state, experiment configs, training logs and metrics.
  The campaign event log maps multiple job IDs back to one experiment.

## Inspect, stop, resume

Run these from the existing server checkout after activating its environment:

```bash
campaign="$PWD/results/validated_v2/0390/conrep-night-20261008"
python scripts/abci/0390_conrep_night.py status --campaign "$campaign"
tail -n 40 "$campaign/supervisor.log"
tail -n 40 "$campaign/events.jsonl"
qstat -u "$USER"
```

`status` also regenerates `checkpoint-results.csv` and `summary.json`. The CSV
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
