# Gemma retain insertion grid and unattended handoff

This implements the approved **nine new Gemma experiments**. No new dropout-only,
replacement, P1, Llama, or seed-replication experiments are included.

| Retain positives per anchor | Insertion modes |
|---|---|
| 2 | fixed1, fixed2, binomial2p20 |
| 3 | fixed1, fixed2, binomial2p20 |
| 4 | fixed1, fixed2, binomial2p20 |

`fixed1`/`fixed2` insert exactly one/two vocabulary tokens per positive.
`binomial2p20` independently draws the count from Binomial(2,.2): probabilities
.64/.32/.04 for 0/1/2 insertions, mean .4. It is a low-intensity exploration
cell, not a claim of a controlled insertion-versus-replacement ablation.
The six fixed-count runs have priority 0; the three low-intensity runs have
priority 1. Three workers share the pool without a wave-completion barrier.

All runs fix G256/B32/W2, forget corruption .7, forget positives 4,
unlearning seed42, and the existing seed42 SFT and data. Batch sizes are
forget/retain/general = 8/32/32. Optimizer, LoRA, loss weights, temperatures,
125-step training, validation partition and v5 scoring are inherited from
the verified Gemma control. Save/evaluate 10,20,...,120,125: 13 checkpoints
per run, 117 new validations. Fixed125, common120 and full curves remain
the reporting protocol; there is no automatic checkpoint selection.

## Insertion construction

Only specified-retain positives change. Fast-tokenizer offsets protect every
token overlapping the complete entity, attribute, copula/value clause or
explicit training-record span. Legal gaps never split those token spans.
Prefix and suffix gaps lie after BOS and before EOS. One/two distinct legal
gaps are sampled uniformly; each receives a uniform legal non-special
vocabulary ID. There is no decode/re-tokenize step, rejection sampling for
unique views, or forced change in the binomial mode. Original IDs and order
are preserved; protected spans remain contiguous. No validation answers are
used. This does not guarantee semantic equivalence of random noise.

Attention and content-pooling masks are rebuilt for each variable-length view,
respecting left/right padding. Inserted tokens participate in attention and
mean pooling; padding/special IDs do not participate in pooling. Inputs with
insufficient headroom fail the CPU preflight instead of truncating originals.
The full 900-row audit, examples, legal gaps, count histograms and vocabulary
provenance are frozen in `retain-insertion-audit.json`.

Anchors, LM inputs, general retain and negative banks remain clean. The
dedicated clean dropout negative bank contributes a fixed **70 negatives**
per anchor for P2/P3/P4. Positive losses are averaged, not summed. Every branch
receives gradients. The existing checkpointed positive CPU generator supplies
all insertion randomness; save/resume also restores all-rank RNG and sampling
state. Logs record counts, zero/one/two-insertion histograms, actual content
fractions, unchanged/unique views, cosine, pair cosine, time and peak memory.
Every new allocation first passes an actual eight-GPU save/resume smoke;
workers test P4/fixed2, P4/binomial2p20 and P4/fixed1, respectively.

## Handoff contract

Prepare the successor while `conrep-positive-grid-20261010` is still running.
Preparation copies the parent's frozen evaluator/model/sampler helpers and
overlays the pinned night package. Neither the server checkout nor the old
campaign's code, configs, tasks, state or jobs are rewritten.

`chain` starts a detached login-node watcher with an exclusive process lock,
durable request, heartbeat and log. It waits for **all parent tasks to be
completed, every training/checkpoint/validation marker to be committed, and
all parent allocations to be reconciled and no longer active**. A generic
`finished` status alone is insufficient. It observes current campaign state,
not a PBS dependency on the original three job IDs. Administrator requeues
therefore do not falsely release the successor. Query errors never mean free
slots. Failed/cancelled tasks, an explicit stop, storage pause, budget pause or
deadline stop are reported as blocked; the watcher does not silently resume
or extend the parent. Resolve the parent condition, then re-run `chain`.

Only after readiness and available account capacity does it invoke the normal
idempotent supervisor start. Waiting leaves successor `started_at` and
`deadline` null. Its default 10-hour budget begins at handoff/start and includes
subsequent PBS queue waiting; there is no guarantee of immediate GPU allocation.
All account jobs/nodes, including queued and other-project jobs, count toward
the existing maximum of three. Queue `R9920261000`, project `gcg51557`, `0390_`
job names, eight GPUs per node, 12-hour PBS walltime, 20-minute administrator
retry and the storage reserve are preserved. Intentional qdel is not revived.
The existing bounded handling of deterministic failures is unchanged.

Repeated `chain` calls do not create duplicate watchers/jobs. The watcher
survives ordinary SSH disconnection, like the existing supervisor; a login
host/process failure still requires re-running `chain`. Once launched, the
normal successor supervisor owns recovery. `chain-status` reports the watcher
lock and persisted heartbeat as well as successor state.

## Prepare and arm now

Run this once on ABCI while the old three jobs continue. No checkout switch,
pull, reset, qdel or manual wait is needed.

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"

git fetch origin codex/0390-conrep-insertion-grid-20261010
ig_ref="$(git rev-parse FETCH_HEAD)"
ig_boot="$(mktemp -d /tmp/0390_conrep_insertion.XXXXXX)"
ig_base="$PWD/results/validated_v2/0390"
ig_campaign="$ig_base/conrep-insertion-grid-20261010"
git show "${ig_ref}:scripts/abci/0390_prepare_conrep_followup.py" > "$ig_boot/prepare.py"

python -I "$ig_boot/prepare.py" \
  --root "$PWD" --ref "$ig_ref" \
  --source-campaign "$ig_base/conrep-positive-grid-20261010" \
  --campaign "$ig_campaign" --profile insertion-grid --hours 10

ig_entry="$ig_campaign/code/scripts/abci/0390_conrep_night.py"
python -I "$ig_entry" chain --campaign "$ig_campaign"
python -I "$ig_entry" chain-status --campaign "$ig_campaign"
)
```

Preparation prints `tasks=9`, `gemma=9`, `llama=0`, `retain_positives=[2,3,4]`.
While the parent is running, expect `watcher_active: true`, successor status
`prepared`, and null successor start/deadline. The main log is
`results/validated_v2/0390/conrep-insertion-grid-20261010/handoff.log`.
If the parent has already completed, status may instead be `launched/running`.

## Inspect, re-arm or recover

From the activated checkout, set:

```bash
ig_campaign="$PWD/results/validated_v2/0390/conrep-insertion-grid-20261010"
ig_entry="$ig_campaign/code/scripts/abci/0390_conrep_night.py"
python -I "$ig_entry" chain-status --campaign "$ig_campaign"
python -I "$ig_entry" status --campaign "$ig_campaign"
```

Before successor start, re-run `chain --campaign "$ig_campaign"` to re-arm a
lost watcher. After start, `recover --campaign "$ig_campaign"` adopts/restarts
the supervisor using the original deadline. Only an explicit
`recover --campaign "$ig_campaign" --hours 10` extends that budget.
`chain-stop --campaign "$ig_campaign"` stops only the waiting handoff; it does
not cancel PBS allocations or stop an already-launched successor.

## Lightweight export

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"
ig_base="$PWD/results/validated_v2/0390"
ig_campaign="$ig_base/conrep-insertion-grid-20261010"
ig_ref="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["ref"])' "$ig_campaign/PREPARING.json")"
ig_export="$(mktemp -d /tmp/0390_conrep_insertion_export.XXXXXX)"
git show "${ig_ref}:scripts/abci/0390_collect_conrep_results.py" > "$ig_export/collect.py"
python -I "$ig_export/collect.py" --root "$PWD" \
  --campaign "$ig_base/conrep-night-20261008" \
  --campaign "$ig_base/conrep-followup-20261008" \
  --campaign "$ig_base/conrep-mixed-grid-20261009" \
  --campaign "$ig_base/conrep-positive-grid-20261010" \
  --campaign "$ig_campaign" --no-include-predictions
)
```

The export includes insertion settings, audit, per-step augmentation metrics
and handoff evidence, without prediction payloads or model/optimizer weights.
Use `--include-predictions` only when individual outputs are needed.

## Local validation

The focused regression suite passed 169 tests; two additional insertion
diagnostic-isolation/resume cases passed separately (171 total). Two CPU
distributed tests skipped because this executor denies Gloo interface access.
Coverage includes token preservation and protected-span contiguity, left/right
padding with BOS/EOS, insertion-count distributions, actual tiny Gemma training
and exact resume, diagnostic isolation, bootstrap provenance, parent-job
reconciliation, delayed budget start, duplicate arming, frozen legacy profiles
and lightweight result exports. Real eight-GPU smoke remains an ABCI launch
gate; it has not been run in this CPU environment.
