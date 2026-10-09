# ConRep positive grid and seed completion

The `positive-grid` profile implements the approved **31 new runs**: Gemma 13,
Llama 18. Its source is the completed `conrep-mixed-grid-20261009` campaign.
The bootstrap copies that campaign's frozen model, sampler and v5 evaluator,
then overlays only the pinned night training/controller package. The server
checkout and completed campaigns are preserved. No test evaluation, automatic
checkpoint selection or automatic grid expansion is performed.

## Exact experiment scope

All runs use the existing model-specific seed42 SFT final and prepared data.
Seeds 42/43/44 below refer only to unlearning. Global forget/general batches are
8/32. Training uses 125 optimizer updates, accumulation 1, AdamW LR 1e-5,
weight decay 0, clipping 1, warmup `round(125*.05)=6` updates, linear decay,
max length 512, BF16, gradient checkpointing, and 8 GPUs per run.
LoRA alpha is twice rank, dropout .05, and target modules are q/k/v/o/gate/up/down.
The frozen Gemma attention implementation remains eager; Llama remains SDPA.

Forget/retain/general CL temperatures remain .08/.1/.1; retain-negative weight
is 2 and margin .1. All CL/LM weights are 1 except the specified forget-CL
weight W. All gradient branches remain active. Save and validate at
10,20,...,120,125: **13 checkpoints per run, 403 new validations**. Report fixed
125 endpoints, common step120 and full curves. With 100 forget examples,
`125*8/100=10` forget-equivalent epochs; sampling does not guarantee ten complete
passes through every example.

Gemma exploration fixes rank256 / retain batch32 / W2 and unlearning seed42:

| Label | Forget probability | Forget positives | Retain positives | Retain probability |
|---|---:|---:|---:|---:|
| GF1 | .5 | 4 | 1 | 0 |
| GF2 | .5 | 8 | 1 | 0 |
| GF3 | .7 | 8 | 1 | 0 |
| GF4 | .9 | 4 | 1 | 0 |
| GF5 | .9 | 8 | 1 | 0 |
| GR1 | .7 | 4 | 1 | .1 |
| GR2 | .7 | 4 | 1 | .2 |
| GR3 | .7 | 4 | 4 | .1 |
| GR4 | .7 | 4 | 4 | .2 |

Exploration IDs encode every factor, e.g. GF2 is
`gemma2_9b-G256B32W2P1C50V8N00-s42`, GR4 is
`gemma2_9b-G256B32W2P4C70V4N20-s42`. C is forget probability in percent,
V is forget positive count, N is retain probability in percent.

The four additional Gemma runs are `G256B32W2P1/P4 × seeds43,44`, with
forget .7/4 and retain noise 0. Existing P1/P4 seed42 supply their controls.

Llama seed42 completes the full
`rank{64,256} × retain batch{16,32} × W{2,5} × P{1,4}` factorial.
Historical J (`G64B16W5P1`) and M (`G256B16W5P1`) supply two cells;
the remaining **14** cells are new. Every cell uses forget .7/4 and retain
noise 0. Add `G256B32W2P1/P4 × seeds43,44` for **18** Llama runs.
Already completed M-S runs are not resubmitted.

Priority 0 admits the four Gemma repeats and Llama P1/P4 seed42 targets.
Priority 1 admits the remaining 9 Gemma and 12 Llama exploration runs.
Priority 2 admits the four Llama repeats. Three workers share one locked queue;
model preferences only break ties. Recoverable paused tasks retain priority.
No wave-completion barrier holds an otherwise available worker idle.

## Positive and negative construction

| Configuration field | Meaning |
|---|---|
| `conrep.corruption_rate` | Forget per-eligible-token replacement probability |
| `conrep.views` | Forget positive count, 4 or 8 |
| `conrep.negative_views` | Other forget instances contribute exactly 4 corrupted views |
| `conrep.specified_views` | Retain positive count, 1 or 4 |
| `conrep.specified_negative_views` | One extra clean dropout negative view per other retain anchor |
| `conrep.specified_noise_probability` | Retain per-eligible-token probability, absent/0, .1 or .2 |
| `conrep.specified_noise_policy` | `training-fact-offset-protection-v1` for noisy runs |
| `conrep.specified_negative_source` | `clean_dropout` for noisy runs |

For **zero noise**, the trainer executes the existing positive construction,
forward order and loss path. P1 uses legacy `paired_loss`; P4 uses the existing
fixed-negative-budget multi-positive loss. Explicit probability 0 and an absent
field produce identical updates and random-generator states. This preserves
the historical controls.

For **nonzero noise**, only the specified-retain positive branch changes.
The anchor and both LM branches retain the original text. A fast tokenizer's
character offsets protect every token overlapping the entire attribute,
entity/identifier or copula/value clause, as well as explicit training-record
protected spans. The complete value includes negation, dates, units and
qualifiers. Unknown grammar, a truncated fact or disagreement with the frozen
`text_batch` tokenisation fails closed. No validation/test answers are read to
construct or protect positives.

Each eligible position/view draws independently from the checkpointed positive
CPU generator. Replacement is uniform over legal non-special vocabulary IDs,
excluding the original ID. There is no forced replacement, distinct-view
resampling or attempt to reach a percentage of all sentence tokens. Fully
protected rows remain unchanged. For the common structured sentence, only
leading `The` and linking `of` may remain eligible, so the effective fraction
of all tokens can be much smaller than .1/.2. Preserving protected token IDs
does not prove semantic equivalence of randomly inserted tokens; audit examples
and the measured changed/unchanged fractions remain part of interpretation.

No noisy positive becomes another anchor's negative. An additional clean-input
dropout forward supplies the fixed negative bank. An anchor's negatives are:
all other clean retain anchors, one clean dropout view of each, and clean
forget anchors. Its own clean dropout view is excluded. With B32/F8 this is
`2*(32-1)+8=70` negatives for both P1 and P4; B16 gives 38. All bank and positive
forwards receive gradients. Extra positives are averaged, not summed. Nonzero
noise incurs an additional clean retain forward; elapsed time and peak memory
remain logged rather than assuming equal compute with the legacy controls.

## Preflight, recovery and evidence

Preparation verifies four historical controls: Gemma P1/P4 seed42 and Llama
J/M seed42. It checks configuration identity and semantic compatibility,
frozen core/evaluator hashes, data/probe hashes, training completion and all
13 committed validations including required prediction files. An absent or
incompatible control stops preparation without publishing a plan or adding a
replacement run. Evidence is frozen in `positive-grid-design.json` and in the
new campaign input manifest.

Before PBS submission, a CPU-only audit uses the actual Gemma SFT tokenizer
and the preserved `text_batch` helper on **all 900 retain training rows**.
`retain-noise-audit.json` records protected spans, eligible counts, replacement
pool provenance, examples and sampled .1/.2 diagnostics. It is included in
the frozen input hashes. Unsupported/truncated rows or zero aggregate
eligibility stop preparation. No model weights are loaded for this audit.

Each new allocation must pass a real 8-GPU two-update save/resume smoke before
claiming a full task. Worker0 tests GR4 (P4, noise .2), worker1 GF5 (eight
forget positives), worker2 Llama G256B32W5P4. Smoke configs are independent of
task-admission preference and recorded with the smoke pass marker.

Existing atomic checkpointing retains all-rank optimizer/scheduler, Python,
NumPy, torch/CUDA and corruption/positive RNG states, plus committed sampling
counts. There is no new uncheckpointed augmentation RNG. Validation resumes
from missing committed checkpoints. Training-only representation/answer and
sampled-gradient diagnostics retain their frozen probe definitions. Per-step
noise logs additionally record global eligible/replaced counts, actual
replacement fractions over eligible and all content tokens, unchanged-view
fraction, unique token views, positive cosine and pair cosine where P>1.

The controller retains three active jobs/nodes maximum for the current user,
including queued/held and other jobs. Each allocation uses one rt_HF node,
8 GPUs, queue R9920261000, project gcg51557, `0390_` names and a 12-hour walltime.
The initial 10-hour campaign budget includes queue waiting. Administrator
termination retries after 1,200 seconds while the budget is open; intentional
qdel is not automatically revived, and deterministic failures are bounded.
The existing storage reserve defaults to 100 GB. No checkpoint pruning is added.

## Deploy and start on ABCI

Run from the existing checkout. Fetching and extracting the bootstrap does not
pull/reset that checkout or install over its server-only evaluator changes.
The fetched commit is recorded in `PREPARING.json` and the frozen provenance.

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"

git fetch origin codex/0390-conrep-positive-grid-20261010
pg_ref="$(git rev-parse FETCH_HEAD)"
pg_boot="$(mktemp -d /tmp/0390_conrep_positive.XXXXXX)"
git show "${pg_ref}:scripts/abci/0390_prepare_conrep_followup.py" > "$pg_boot/prepare.py"
pg_campaign="$PWD/results/validated_v2/0390/conrep-positive-grid-20261010"
python -I "$pg_boot/prepare.py" \
  --root "$PWD" --ref "$pg_ref" \
  --source-campaign "$PWD/results/validated_v2/0390/conrep-mixed-grid-20261009" \
  --campaign "$pg_campaign" --profile positive-grid --hours 10

pg_entry="$pg_campaign/code/scripts/abci/0390_conrep_night.py"
python -I "$pg_entry" start --campaign "$pg_campaign"
python -I "$pg_entry" status --campaign "$pg_campaign"
qstat -u "$USER"
)
```

Preparation must print `tasks=31, gemma=13, llama=18, historical_controls=4,
workers=3`. If preparation fails, `set -e` prevents `start`. Fix the reported
cause without modifying completed results or overriding a hash check. The
same revision/profile preparation is idempotent and never resets state.
Only `start` submits PBS work; this repository implementation does not itself
claim that the ABCI runs or GPU smokes have executed.

For status or an explicit budget extension, restore the environment and use:

```bash
pg_campaign="$PWD/results/validated_v2/0390/conrep-positive-grid-20261010"
pg_entry="$pg_campaign/code/scripts/abci/0390_conrep_night.py"
python -I "$pg_entry" status --campaign "$pg_campaign"
# If the budget has ended with pending work, resume the same campaign:
python -I "$pg_entry" resume --campaign "$pg_campaign" --hours 10
```

## Export and analysis

Extract the collector from the exact overlay revision recorded in the campaign:

```bash
pg_campaign="$PWD/results/validated_v2/0390/conrep-positive-grid-20261010"
pg_ref="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["ref"])' "$pg_campaign/PREPARING.json")"
pg_export="$(mktemp -d /tmp/0390_conrep_export.XXXXXX)"
git show "${pg_ref}:scripts/abci/0390_collect_conrep_results.py" > "$pg_export/collect.py"
python -I "$pg_export/collect.py" --root "$PWD" --campaign "$pg_campaign" --include-predictions
```

The ZIP includes all committed per-example PMC/MMLU predictions, configs,
fixed-step metrics, full curves, input/source hashes, historical control
evidence, retain-noise audit, sampling coverage and smoke pass metadata.
`experiment-configurations.csv` now includes both corruption probabilities,
positive/negative view budgets and negative-source policy.
`augmentation-diagnostics.csv` contains per-step positive/noise observations;
abandoned/replayed observations are labelled, while `sampling-coverage.csv`
uses committed checkpoints. Model and optimizer weights are excluded.

Without explicit campaigns, the collector includes the original, follow-up,
mixed and positive campaigns when present. Verified predictions are included
by default for the positive campaign; `--include-predictions` includes them
for all selected campaigns, allowing read-only audit of historical MMLU
parsing and paired per-example comparisons. `--no-include-predictions` opts
out explicitly. Existing v5/strict metrics and scoring remain unchanged.

## Code and verification

`positive_grid.py` owns the matrix, historical controls, audits and immutable
configs. `noise.py` owns token protection/alignment and observed noise counts.
`losses.py` implements the independent clean negative bank; `trainer.py`
selects it only for nonzero noise. The existing bootstrap/controller supply
the new profile and explicit smoke mapping. The collector preserves the new
provenance and per-example evidence.

The relevant regression suite covers explicit loss/gradient formulas,
positive-count averaging, unchanged zero-noise updates/RNG, real tiny
Llama/Gemma noise training and exact checkpoint resume, full preparation and
its failure gates, 31 unique dynamic claims, historical-source preservation,
account job limits/recovery and committed exports. CPU distributed tests skip
when Gloo interfaces are unavailable; actual 8-GPU success is checked on ABCI.

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q tests/0390/test_conrep_*.py
```
