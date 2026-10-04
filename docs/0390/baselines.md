# Baseline integration and source mapping

All six methods start from the same selected SFT checkpoint per backbone and
receive the same prepared `F`, `R_s`, and `R_g` pools. Method-specific data
augmentation/calibration is recorded separately. The common evaluator is
`scripts/0390_experiment.py validate`; training adapters do not substitute their
upstream benchmark evaluators. These are ClinicIA adaptations, not claims of
bitwise reproduction of the original papers' complete pipelines.

## Sources and execution

| Method | Pinned source | Preserved algorithm | Parameterization |
| --- | --- | --- | --- |
| NPO | [OpenUnlearning d33c476](https://github.com/locuslab/open-unlearning/tree/d33c4762941731cb397494f605f71528d8ce9dce) | Sequence-summed NPO against a frozen start model + retain CE | Shared rank-32 LoRA configuration |
| RMU | Same OpenUnlearning snapshot, `src/trainer/unlearn/rmu.py` | Random normalized activation target + frozen-reference retain activation MSE | Selected layers' `mlp.down_proj.weight` |
| SAGO | [SAGO 2363ee7](https://github.com/sustech-nlp/SAGO/tree/2363ee74ec45cf467e7c28127b0b82b3b94690f7) | NPO + retain CE, elementwise sign-aligned gradient synthesis | Shared rank-32 LoRA configuration |
| FALCON | [FALCON 12e09a3](https://github.com/CharlesJW222/FALCON/tree/12e09a337cb5f8cec60f24f29c76c2911b15468c) | MI layer selection, original SVD steering function, contrastive unalignment, cosine retention, gradient conflict projection, SophiaG | Selected layers' down projections |
| LUNAR | [LUNAR dfa56eb](https://github.com/facebookresearch/LUNAR/tree/dfa56eb0291a93e967284a9ec2d28d5572d235b1) | Refusal-minus-forget activation direction; fit local down projection to shifted forget / original retain targets | Native full local down-projection fit |
| ReLearn | [ReLearn e0d6938](https://github.com/zjunlp/unlearn/tree/e0d6938a0b7adcaf6d8e9e869c598bf100b8849e) | Question augmentation, fuzzy-answer generation and disclosure filtering; `relearn_klr_gdr` training | Shared rank-32 LoRA configuration |

The upstream FALCON numeric kernels, MI estimator, ReLearn prompt templates,
and all relevant licenses are stored in `third_party/0390_baselines/` with
SHA-256 values. SophiaG is the exact `decoupled_sophia.py` exported by
`zetascale==2.7.7`, the version pinned by FALCON. Its standalone source is used
to avoid importing the unrelated model dependencies in the entire zeta package.
The wheel hash and source hash are recorded in the manifest.

NPO, RMU and SAGO have small, explicit ports in `baselines/common.py` and the
shared trainer; LUNAR fitting is in `baselines/lunar.py`. This avoids the older
repositories' incompatible global Transformers/Trainer dependencies while
making the changed interfaces inspectable. No baseline silently falls back to
another loss if a required asset is missing.

## Important adaptations

- NPO uses the sum of supervised token NLL per sequence, not a length-normalized
  SimNPO objective. The reference model is frozen at the shared SFT start.
- `R_s` and `R_g` losses are separately averaged and weighted. Their values use
  each baseline's native retention objective: CE, activation MSE, cosine or
  CE+KL. Shared data access does not mean those mathematically different losses
  should have identical numerical weight scales.
- SAGO implements the official `sago` variant: at an elementwise conflict use
  the retain gradient; otherwise use the forget gradient. It does not add both
  gradients at nonconflicting elements. Each task gradient is accumulated and
  averaged across ranks before applying the gate. Zero signs are nonconflicting.
- FALCON's MI estimator is called on separately recorded `F`/`R_s` and `F`/`R_g`
  activations, then combined with the declared retain weights. Invalid KDE
  estimates are not treated as evidence for a low-MI layer. The selected block
  and up to its two predecessors train their down projections. Parameter names
  replace fragile numeric parameter indices across Llama/Qwen architectures.
  POVs are computed from the frozen model on the current forget minibatch,
  rather than the upstream WMDP-specific two-topic prebuilt list. Its original
  steering, loss, projection, and optimizer kernels remain unchanged.
- LUNAR uses native backbone chat templates. Refusal directions are differences
  of final-prompt-token block outputs, equivalent to the next block's pre-hook
  location. Features and original targets always come from the frozen start
  model. Sampling features on demand avoids caching a large Wiki activation
  dataset. The direction is added to all valid-token targets, matching the
  pinned implementation's operation despite its last-token-only docstring.
  Local fitting uses AdamW and the declared step-based decay schedule.
- ReLearn uses the original question and fuzzy-answer prompts. The official
  filter hard-codes private/public attribute categories for public figures;
  the adapter instead asks whether the designated IA answer is disclosed.
  Augmentation is executed with a local teacher so it can run on ABCI. Literal
  answer inclusion additionally fails the filter. The resulting file records
  the teacher and an audit; incomplete forget coverage stops training.
- Padding is excluded from activation and KL losses. NPO/SAGO/ReLearn use the
  shared LoRA interface; RMU/FALCON/LUNAR retain their local full-weight update
  mechanisms. Report these parameterizations alongside results.

## Commands

All commands below run from the repository root after environment/data/SFT
preparation. `CONREP_SFT_CHECKPOINT` must be the selected checkpoint for the
backbone in the command. The queue submitter accepts at most two active jobs;
run later commands after earlier jobs have completed.

### NPO, RMU, SAGO

```bash
python scripts/abci/0390_submit.py baseline --model llama3b --run-id rmu42 \
  --method rmu --checkpoint "$CONREP_SFT_CHECKPOINT" \
  --set run.output_dir=results/validated_v2/0390/llama3b/rmu-seed42

python scripts/abci/0390_submit.py baseline --model llama3b --run-id sago42 \
  --method sago --checkpoint "$CONREP_SFT_CHECKPOINT" \
  --set run.output_dir=results/validated_v2/0390/llama3b/sago-seed42
```

NPO uses the identical command with method/output `npo`. Default RMU target
block is 7 and trainable blocks are 5/6/7; tune the layer and steering strength
using the same validation constraints rather than assuming these are optimal
for new backbones.

### FALCON

```bash
python scripts/abci/0390_submit.py falcon-layers --model llama3b --run-id falconmi \
  --method falcon --checkpoint "$CONREP_SFT_CHECKPOINT" \
  --output results/validated_v2/0390/llama3b/falcon-layers.json

# After the MI job has completed:
python scripts/abci/0390_submit.py baseline --model llama3b --run-id falcon42 \
  --method falcon --checkpoint "$CONREP_SFT_CHECKPOINT" \
  --set baseline.layer_selection=results/validated_v2/0390/llama3b/falcon-layers.json \
  --set run.output_dir=results/validated_v2/0390/llama3b/falcon-seed42
```

Input: shared checkpoint, fixed calibration samples from each training pool.
Output: per-layer MI/entropy estimates and the chosen layer, then full-model
checkpoints. Layer selection has not been validated on actual 3B/7B activations.

### LUNAR

```bash
python scripts/abci/0390_submit.py baseline --model llama3b --run-id lunar42 \
  --method lunar --checkpoint "$CONREP_SFT_CHECKPOINT" \
  --set run.output_dir=results/validated_v2/0390/llama3b/lunar-seed42
```

Input: same training pools and the downloaded official refusal instruction
file. The canonical question is reconstructed from IA metadata only when its
identifier and answer are present in the canonical training text; no additional
evaluation formulations are imported. If this join cannot resolve a row, provide
`data.forget_requests` / `data.retain_requests` JSONL with `fact_id`, `prompt`,
`answer`, `kind: qa`. This is an explicit data-adaptation requirement, not an
automatic expansion of the forget knowledge scope.

Output: checkpoints containing the model with the fitted down projection, plus
the local fitting optimizer/scheduler state. Default block 22, coefficient 2 and
200 fitting steps are starting values to validate on both backbones.

### ReLearn

```bash
# Generate once with a fixed teacher; share the resulting file across backbones.
python scripts/abci/0390_submit.py relearn-augment --model qwen7b --run-id relearnprep \
  --method relearn --output data/processed/0390/relearn.jsonl

# After reviewing the generated coverage/audit:
python scripts/abci/0390_submit.py baseline --model llama3b --run-id relearn42 \
  --method relearn --checkpoint "$CONREP_SFT_CHECKPOINT" \
  --set run.output_dir=results/validated_v2/0390/llama3b/relearn-seed42
```

Input: canonical forget requests, fixed teacher and original augmentation
templates. Output: approved QA training rows and `.audit.json`, then adapters.
If the automatic filter leaves a fact uncovered, the trainer reports it rather
than replacing missing answers with a generic refusal. Teacher output quality
and actual PMC request joins still need inspection before numerical experiments.

## Fair tuning and validation

Use the same Wiki pool and specified retain pool, source checkpoint, data split,
seeds, checkpoint-validation protocol and utility constraints. Record trainable
parameter counts, forget/retain exposure and walltime, since local weight fitting
and gradient surgery have different costs. Give each method an explicit tuning
budget; a useful first small search changes learning rate and a shared multiplier
on its two retain coefficients. Keep all validation outputs, including unsuccessful
trade-offs. Current defaults are executable initial settings, not tuned SOTA claims.
