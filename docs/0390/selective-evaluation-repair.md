# Recover selective evaluation and incomplete ReLearn preparation

The October 12 T0-General-only diagnostics showed two independent failures:

- Thirteen of fourteen trainings have valid native `TRAINING_COMPLETE.json` and
  all planned checkpoints. Evaluation generated its primary responses, then
  crashed because gender probes have two options while the frozen v5 extraction
  function accepts only four. A normal PBS exit does not mean tasks succeeded.
- Llama ReLearn did not start training: the original three augmentation rounds
  covered 76/100 facts, leaving 24 uncovered. Its failure is separate from MCQ
  parsing. The two completed records are reused SFT references.

The parent selective campaign has the same binary probes. Repair both campaigns
so future experiments do not hit the same failure.

## What the repair preserves

`0390_repair_selective_evaluation.py` freezes a small Python launcher and an
audited extension, then updates only the operational Python dispatch and its
manifest pointer. Scientific sources, original entry points, worker shells,
dependency controllers, configs, training identities, inputs, SFT references,
weights and optimizer archives retain their existing bytes. Native verification
still runs. Training and torchrun commands pass through to the original Python.

The installer hands off the two login-node supervisors using their existing lock
and state protocol. It sends no cancellation or signal to a PBS training process.
Existing workers read the updated plan before claiming their next task. A worker
finishing a task during supervisor handoff can drain its allocation normally;
the existing supervisor replenishes it. An already running old evaluation can
still encounter the old parser once; the new supervisor detects that precise
failure and requeues its completed-training evaluation.

Capacity remains the authorized **parent 3 + companion 1**, at most four account
jobs and four nodes, including both controllers' queued/unseen reservations.
No fourth parent worker or extra companion worker is introduced.

## Scoring and cache audit

Only binary A/B extraction is added. It is the binary counterpart of the frozen
direct-choice parser, without a gold-answer argument. C/D choices, conflicting
choices and hedged answers abstain. Four-option MCQ, generation and MMLU scoring
continue through the original functions. Binary accuracy remains separate from
four-option accuracy; primary F-any/R-any still use three generation prompts.

Before accepting an existing prediction prefix, the repair checks the original
evaluation identity, every saved probe's metadata, and its score under the audited
parser. Foreign or inconsistent caches fail closed. Only missing predictions are
generated; weights and completed primary generation are reused.

The old protocol hash remains a **base compatibility key**, allowing native
workers and immutable SFT references to validate the same cache. It is not a claim
that runtime code is identical. `EVALUATION_EXTENSION.json` records the repair
revision, file hashes, base protocol hash, preserved prediction count/prefix hash,
and a separate `execution_identity`. Metrics and the final validation marker bind
that execution identity. Repaired exports include the frozen extension, its
manifest and evaluation receipts. Compare repaired results using both identities.

## Bounded ReLearn continuation

Only incomplete, untrained augmentation is extended. The original three rounds
and accepted rows are retained; completed calibration is never replaced. At most
six additional fixed paraphrase rounds request only facts still uncovered. The
original teacher, augmentation templates, literal leakage check, teacher verdict,
typed-answer filter and full forget set remain in force. No fabricated fallback
answer, dropped fact or relaxed filter is used.

Each additional round records request/output/audit hashes and can resume from its
completed cache. `CALIBRATED.json` is emitted only at full coverage and records the
preparation extension. `recovery-v1-coverage.json` reports progress. If coverage
is still incomplete after six additional rounds, ReLearn stays failed with
`blocked_augmentation_recovery_exhausted`; it must not be reported as completed.
Extra preparation time is part of calibration cost, not a matched-compute claim.

## Install and resume

Run this from an ABCI login node. It verifies both snapshots before handoff,
requeues only supported failures once, and arms both supervisors. Do not add
`--retry-failed`: that would reset unrelated failures too.

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"

fix_base="$PWD/results/validated_v2/0390"
fix_parent="$fix_base/selective-extensive-20261011"
fix_campaign="$fix_base/selective-t0-general-only-20261012"
git fetch origin codex/0390-selective-training-20261011
fix_ref="$(git rev-parse FETCH_HEAD)"
fix_ref="$(python -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p.get("evaluation_repair",{}).get("ref",sys.argv[2]))' "$fix_campaign/plan.json" "$fix_ref")"
fix_tmp="$(mktemp -d /tmp/0390_evaluation_repair.XXXXXX)"
git show "${fix_ref}:scripts/abci/0390_repair_selective_evaluation.py" > "$fix_tmp/repair.py"
python -I "$fix_tmp/repair.py" --root "$PWD" --parent "$fix_parent" \
  --campaign "$fix_campaign" --ref "$fix_ref"
qstat
)
```

Repeat installation uses the already installed revision and does not submit
duplicate jobs. An explicit STOP is respected; resume it deliberately first.
Preflight failures leave existing supervisors and scientific work untouched.

Status and export must use the Python path in each campaign's plan, so the
execution audit is included:

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"
fix_base="$PWD/results/validated_v2/0390"
for fix_name in selective-extensive-20261011 selective-t0-general-only-20261012; do
  fix_campaign="$fix_base/$fix_name"
  fix_python="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "$fix_campaign/plan.json")"
  fix_entry="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["temporary_extra_node"]["entry"])' "$fix_campaign/plan.json")"
  "$fix_python" -I "$fix_entry" status --campaign "$fix_campaign"
done
qstat
)
```

For the companion summary, use the same block with its campaign only and replace
`status` with `export`. `--include-predictions` produces a separate predictions ZIP.
Successful completion requires all 16 state records completed, a current
`COMPLETE.json`, and nonzero validated rows; queue disappearance alone is not proof.

## Verification

Regression tests use the actual frozen v5 parser via
`CONREP_FROZEN_TEST_ROOT=/path/to/campaign/code`. Tiny CPU Gemma and Llama models
first reproduce the binary crash, then complete both validation and test with the
generation prefix unchanged. Tests also cover four-choice/generation equivalence,
foreign-cache rejection, bounded missing-fact augmentation and one-time targeted
requeue. These CPU checks do not claim that ABCI's remaining ReLearn teacher calls
will achieve full coverage.

Local verification: 81 tests plus three subtests passed across the evaluation
repair, reader repair, dependency scheduler, T0-general-only and 3+1 admission
suites. An integration run used the actual frozen supervisors with mocked PBS:
handoff, 14 evidence-backed retries, a late parent failure, repeated installation,
parent/companion refill and native snapshot verification passed. Separate tiny
Gemma/Llama command-line evaluations and the audited summary export also passed.
PBS allocation/checkpoint markers in that scheduling test were fixtures, not an
ABCI GPU training run.
