# ConRep follow-up: 27 experiments and sparse diagnostics

This continues `conrep-night-20261008` in a separate campaign. It inherits that
campaign's frozen SFT paths, prepared data, model helpers and full validation
evaluator. The bootstrap copies its fingerprinted source and overlays the night
package and launcher. It never checks out a branch over the server's working
tree. Previous experiments and their checkpoint identities remain intact.

## Approved matrix

All experiments start from the same SFT checkpoint for their model. They use
125 optimizer steps, global batches 8/16/32, checkpoints 10,20,...,120,125 and
validation of every checkpoint. There is no checkpoint selection or test run.

| Tasks | Seeds | Count |
|---|---|---:|
| Llama E, F, G, H, I, J, K, L | 42, 43 | 16 |
| Gemma J, K | 43, 44 | 4 |
| Gemma M, N, O, P, Q, R, S | 42 | 7 |

E–L have exactly the definitions in [the original matrix](conrep-night.md).
J is forget CL weight 5 with rank64/alpha128; K uses rank128/alpha256. Their
learning rate is 1e-5, LoRA dropout .05 and both retain/general LM weights are 1.

| New ID | Reference | Only change from reference |
|---|---|---|
| M | K | rank256/alpha512; alpha/r remains 2 |
| N | K | learning rate 5e-6 |
| O | J | learning rate 2e-5 |
| P | K | forget CL weight 1 |
| Q | J | one fact-preserving retain paraphrase instead of the same-text dropout positive |
| R | J | detach the corrupted-forget representations in forget CL |
| S | J | detach specified-retain and general-retain representations in forget CL |

Q preserves the entity, attribute, copula and **complete value clause**, including
negation, units and time qualifiers. It reorders supported training assertions;
an existing `views` entry is accepted only if its complete parsed assertion
matches. It does not sample validation/test labels, add positive views, or change
global dropout. Preparation audits every retain row and refuses unsupported
grammars before submitting any PBS jobs; examples are in `fact-positive-audit.json`.

R detaches the entire corrupted branch. Those representations also occur as
other-instance negatives, so this is not exclusively a positive-target detach.
S detaches r/g only where they enter forget CL; their own CL and LM gradients
remain active. Both controls preserve the scalar loss value at fixed inputs.

## Three workers and recovery

Three persistent worker slots share one locked task pool. The first wave is
Llama J/42, Llama K/42, Gemma K/43. The next priority is Llama J/K/43 and the
remaining Gemma J/K replications, followed by Llama E/F/G/H/I/L, then Gemma M–S.
Workers run multiple training-plus-validation experiments serially within an
allocation. All 20 completion/replication tasks have priority over fresh
explorations; an exploration can begin once those 20 have been claimed, even if
another worker is still finishing one. Paused experiments resume first.

The existing account-wide cap of three active jobs/nodes counts queued and held
jobs too. Each allocation requests one eight-GPU HF node, queue `R9920261000`,
project `gcg51557`, **12:00:00 PBS walltime**. Administrator termination and
termination while queued retry every 1,200 seconds with no retry-count ceiling,
while the campaign budget remains open. Manual cancellation remains intentional;
deterministic training errors are not treated as unlimited interruptions.

`--hours 10` is the separate campaign budget, including queue wait. It does not
change the 12-hour PBS request or promise all 27 runs will finish in ten hours.
Use an explicit `resume --hours 10` to grant another ten-hour window; completed
experiments and checkpoint validations are skipped. Each new allocation first
runs the existing eight-GPU save/resume smoke with diagnostics enabled.

## Diagnostics and their interpretation

Each model gets the same fixed training-only sample across all variants and
seeds: 16 forget facts, 16 retain facts and 8 general texts, selected by a fixed
hash order. The files are fingerprinted with the other inputs. The sampler
requires complete supported facts and records any rejected grammar examples.

At step 0 and after steps 25/50/75/100/125 it records:

- Cosine with the same run's immutable step-zero SFT representations for all
  groups, plus within-group off-diagonal cosine to expose representational collapse.
- Forget alignment with current corrupted representations, alignment with the
  fixed SFT corrupted representations, and drift of those corrupted representations.
  The diagnostic corruption is always four fixed views at probability .7,
  independent of the training corruption setting and RNG.
- Mean and summed correct-answer token log probability on the 32 training facts,
  using native chat prompts. Only answer tokens are scored. Answers exceeding the
  context are explicitly marked unscored, never silently truncated.
- Frobenius norms of the adapter deltas for the sampled LoRA modules.

At steps 1/25/50/75/100/125, before the update, it also records each active loss
component's raw/weighted gradient norm and pairwise gradient cosine. This covers
**four named LoRA B matrices**, not the full model. Matrices are selected evenly
over eligible parameter names, with a two-million-element limit per matrix.
Gradients are averaged across ranks and reported per accumulation microbatch,
before clipping; they are not the norm of the sum across multiple microbatches.
JSON records include the exact names and parameter count. Rank changes therefore
change the sampled parameter count; do not interpret unnormalised cross-rank
norm differences as evidence of a better objective by themselves.

These probes diagnose training dynamics and embedding/generation consistency.
They are not held-out semantic-quality evidence or substitutes for the paper's
validation metrics. A higher corrupted-positive cosine alone does not establish
successful forgetting; compare fixed-reference drift, answer probabilities and
the unchanged validation metrics together.

Diagnostics preserve Python, NumPy, torch CPU/current-GPU RNG and model train/eval
state. They do not write optimizer gradients or step the optimizer. Step-zero
references are hashed and retained across retries; a resumed run refuses to
rebuild a missing reference using already-unlearned weights.

## Prepare and start on ABCI

Run this from a login node. Only `start` submits jobs. Preparation is idempotent
for the same revision and source; rerunning it does not reset progress.

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"

git fetch origin codex/0390-conrep-night-20261008
followup_ref="$(git rev-parse FETCH_HEAD)"
followup_dir="$(mktemp -d /tmp/0390_conrep_followup.XXXXXX)"
git show "${followup_ref}:scripts/abci/0390_prepare_conrep_followup.py" > "$followup_dir/prepare.py"

source_campaign="$PWD/results/validated_v2/0390/conrep-night-20261008"
campaign="$PWD/results/validated_v2/0390/conrep-followup-20261008"
python -I "$followup_dir/prepare.py" --root "$PWD" --ref "$followup_ref" \
  --source-campaign "$source_campaign" --campaign "$campaign" --hours 10

entry="$campaign/code/scripts/abci/0390_conrep_night.py"
python -I "$entry" start --campaign "$campaign"
python -I "$entry" status --campaign "$campaign"
qstat -u "$USER"
)
```

For a later status refresh, CSV export, or budget extension, activate the same
environment and use the new campaign's frozen entry:

```bash
campaign="$PWD/results/validated_v2/0390/conrep-followup-20261008"
entry="$campaign/code/scripts/abci/0390_conrep_night.py"
python -I "$entry" status --campaign "$campaign"
python -I "$entry" summarize --campaign "$campaign"
# After the budget has expired, explicitly grant a new window:
python -I "$entry" resume --campaign "$campaign" --hours 10
```

`checkpoint-results.csv` contains the original validation metrics for every
completed checkpoint. `diagnostic-results.csv` contains sparse representation and
answer-probability summaries. `gradient-diagnostics.csv` contains sampled gradient
norms and cosines. Their full per-example/per-parameter records live under
`experiments/<id>/training/diagnostics/`. Live optimizer progress is in
`experiments/<id>/training/train.jsonl`; the worker console reports task transitions.

## Local validation

The follow-up tests cover the exact 27-run matrix, single-factor overrides,
first-wave and late-budget task admission, directional detach gradients,
fact-clause preservation, answer token masking, snapshot inheritance, preparation
idempotence and refusal of mutated inputs. Tiny real Llama and Gemma models cover
default/Q/R/S: enabling diagnostics produces bit-identical final adapters, and
interrupted/resumed training reproduces continuous weights and diagnostic records.

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q \
  tests/0390/test_conrep_night.py \
  tests/0390/test_conrep_night_recovery.py \
  tests/0390/test_conrep_night_supervision.py \
  tests/0390/test_conrep_followup.py
```

The two-rank CPU tests explicitly skip if the host denies Gloo network-interface
access. These local checks do not claim to validate eight-GPU bf16 execution;
the mandatory ABCI allocation smoke is still the deployment gate.
