# ConRep v2 implementation specification

## Data and model

Let `F` be the designated forget facts, `R_s` specified retain facts, and `R_g`
the fixed general retain text pool. One fact is one canonical training row;
existing paraphrase columns provide additional expressions of an `R_s` fact.
`h_theta(x)` is the L2-normalized mean of final-layer **causal** decoder states
at the non-special content tokens of text `x`. `theta` denotes the shared model
parameters, including the trainable LoRA adapter. The same decoder and adapter
generate text with the ordinary causal LM head. There is no projection head,
bidirectional attention conversion, frozen embedding teacher or MNTP stage.

Both backbones use rank-32, alpha-64 LoRA on q/k/v/o and gate/up/down projections,
with dropout 0.05. Generation uses the backbone's native chat template. Embedding
uses plain fact/document tokens and pools only content positions. This is a
joint causal embedder-generator training configuration, not a claim to reproduce
GritLM or MAGNET training in full.

## Corruption and loss

For each forget fact `f_i` (index `i`), draw `K=4` independent views
`c_i^k`, where `k` is the view index. Each eligible content token is independently
replaced with probability `rho=0.7`, uniformly over legal non-special vocabulary
IDs other than itself. BOS/EOS/padding and any non-content positions are protected.
Replacement length equals original length. Masks and noise are redrawn on each
training encounter. No answer-span parsing or mandatory value replacement is used.
Both original and corrupted branches receive gradients.

For an anchor representation `a`, its positive set `P(a)`, its negative set
`N(a)`, and temperature `tau > 0`, the implemented per-positive loss is

\[
\ell(a)=-\frac{1}{|P(a)|}\sum_{p\in P(a)}
\log\frac{\exp(s(a,p)/\tau)}{\exp(s(a,p)/\tau)+\sum_{n\in N(a)}\exp(s(a,n)/\tau)},
\]

where `s` is cosine similarity and `|P(a)|` is the number of positives. Other
positives are excluded from each positive's negative denominator.

- Forget: positives are only the `K` own corrupted views. Negatives are other
  forget facts, other facts' controls, specified retain facts and general retain
  texts. The anchor itself is excluded. Retain negative cosine logits receive
  margin `m=0.1` **before** division by `tau_f=0.08`, and multiplicative
  denominator weight `w=2` (implemented as `log(w)` in logit space).
- Specified retain: the positive is an existing paraphrase of the same pivot.
  Other retain pivots/views and all forget pivots are negatives; self is excluded.
  Temperature `tau_s=0.1`.
- General retain: two forward passes of the same text, with independent LoRA
  dropout, are positives; other general texts/views are negatives.
  Temperature `tau_g=0.1`.

The main objective is

\[
L=L_F+\lambda_s L_s^{CL}+\lambda_g L_g^{CL}
  +\mu_s L_s^{LM}+\mu_g L_g^{LM}.
\]

`L_F` is mean forget contrastive loss; `L_s^CL` and `L_g^CL` are specified/general
contrastive losses; `L_s^LM` and `L_g^LM` are ordinary next-token cross-entropy
on the two retain streams. `lambda_s`, `lambda_g`, `mu_s`, `mu_g` are their
nonnegative weights, initially all 1. Forget/corrupted texts receive no LM loss.
There is no hidden NPO or gradient-ascent term in ConRep v2.

Relative to the historical code, this deliberately changes shared random text
to independent token-ID noise, changes sum-of-positive exponentials to the
per-positive objective above, fixes the self-negative mask, and applies the
margin in cosine rather than temperature-scaled units. These changes must be
reported when comparing to historical ConRep; this is not byte-identical replay.

## Training and checkpoint state

Global per-forward batches are 8 forget, 16 specified retain and 32 general
retain. They are divided equally among `torchrun` ranks. Each stream is sampled
without replacement within a global batch; the RNG is a deterministic function
of seed and update/micro-step. The three streams use independent draws.
Training length is measured in expected forget exposures, not Wiki epochs.

All CL embeddings are gathered with autograd support before forming the loss.
LM losses are local means; parameter gradients are averaged across ranks.
Gradient accumulation does not enlarge the contrastive negative pool: that
pool is the global per-forward batch. Model replicas are explicitly synchronized
through gradient averaging; do not wrap this loop in an additional DDP reducer.
Gradient checkpointing uses the non-reentrant implementation.

Checkpoints include model/adapter, tokenizer, optimizer, scheduler, per-rank
CPU/CUDA/corruption RNG state, config and a completion marker. Resume requires
the original world size and method/data/model configuration. Full training
completion does not automatically select a checkpoint.

## Diagnostics and ablations

`analyze` records geometry separately for `F`, `R_s`, `R_g`; mixing the groups
into one rank statistic would hide local collapse. The centered effective rank
is `exp(-sum_j p_j log p_j)`, where `p_j` is singular value `j` divided by the
sum of singular values of the centered representation matrix. A constant matrix
has rank statistic 0. Pairwise cosine and centered norms accompany this statistic.
Retain own-view vs hardest-other cosine and STS-B Spearman assess embedding
quality; own-control cosine measures steering. None alone establishes forgetting.

The implemented ablations are shared target, removal of inter-forget negatives,
removal of specified CL, removal of general CL, and removal of both retain LM
terms. In the shared-target ablation all identical controls are positives, so
they are not simultaneously labeled as negatives. Two-stage warmup, bidirectional
embedding and answer-span corruption are not part of this first execution path.
