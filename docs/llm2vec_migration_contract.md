# Phase 3D-1 LLM2Vec migration contracts

Phase 3D-1 turns the `llm2vec`/`llm22vec` relationship identified in the
Phase 3D-0 ownership audit into executable migration contracts. This batch does
not move or edit model source, delete a package, download dependencies, load a
model, or claim numerical equivalence.

The exact source, import-consumer, AST, and behavior records are in
[`llm2vec_migration_contract.json`](llm2vec_migration_contract.json). Tests use
small dependency stubs because Torch, Transformers, PEFT, and tqdm are not part
of the repository's validated lightweight environment.

## Why both packages must remain

The two packages expose a class named `LLM2Vec`, but they do not have the same
model contract:

| Contract | `llm2vec` | `llm22vec` |
| --- | --- | --- |
| Public implementation | `llm2vec.llm2vec.LLM2Vec` | `llm22vec.llm22vec.LLM2Vec` |
| `enable_bidirectional` default | `True` | `False` |
| Non-bidirectional loader | `AutoModel` | `AutoModelForCausalLM` |
| Forward request | Normal model call | Adds `output_hidden_states=True` |
| Representation used | `last_hidden_state` | Last item of `hidden_states` |
| Direct tracked consumers | 13 | 11 |

The shared bidirectional mapping is Mistral, Llama, Gemma, and Qwen2 to their
respective `*BiModel` implementations. Both packages currently share the same
pooling implementation, but that fact does not make their construction or
forward contracts equivalent.

The causal derivative also passes an empty `token` keyword to tokenizer,
configuration, and model loaders. This is recorded as observed legacy behavior,
not recommended credential handling.

## Pooling contract

Both implementations require left padding and support:

- `mean`: mean of the rightmost attention-length positions;
- `weighted_mean`: weights those positions from 1 through sequence length;
- `eos_token` and `last_token`: use the final hidden-state position;
- `bos_token`: select hidden states where input IDs equal the tokenizer BOS ID.

When `skip_instruction=True`, `embed_mask` replaces `attention_mask` before
pooling. The dependency-stubbed tests execute each pooling branch with small
array fixtures and also prove that the normalized `get_pooling` AST is identical
between the two files.

## Forward contract

Both implementations remove `embed_mask` before calling the underlying model
and restore it before pooling. The canonical implementation pools
`reps.last_hidden_state`. The causal derivative requests all hidden states and
pools `reps.hidden_states[-1]`.

The tests execute these calls against recording doubles. They verify exact
keywords and representation selection without initializing any real model.

## Historical ConRep contract

The selected `historical_v1` ConRep records resolve to:

```text
llm2vec/ContrastiveUnlearning_Adaptive_RandomToken_LMloss_margin.py
```

That entry point imports `LLM2Vec` from `llm22vec`, declares
`bidirectional=False`, and explicitly forwards the value to
`LLM2Vec.from_pretrained`. A later canonical ConRep entry point must preserve a
traceable compatibility route for this behavior rather than silently switching
to the canonical LLM2Vec defaults.

## OpenUnlearning wrapper contract

`LLM2Vec2CausalLM` exposes the derivative's inner causal model to the
OpenUnlearning evaluation path. The wrapper:

- exposes the inner model's tokenizer and configuration;
- delegates `generate(*args, **kwargs)` unchanged;
- calls the inner model with `input_ids`, `attention_mask`, `labels`, and
  `return_dict=True`;
- returns a `CausalLMOutputWithPast` containing copied `loss` and `logits`.

The current wrapper accepts extra `forward` keyword arguments but does not pass
them to the inner model. The contract test preserves this observed behavior so
a future adapter change must be explicit and reviewed.

## Preserved parse finding

`llm2vec/ContrastiveUnlearning_Adaptive_RandomToken_LMloss_EpochEval.py` cannot
currently be parsed as Python: line 581 contains adjacent duplicate `json_file`
keyword assignments without a separator. It remains a direct `llm22vec`
consumer and is included by the conservative import scan. Phase 3D-1 records
the defect but does not repair this legacy entry point.

## What the tests prove

The lightweight suite proves:

1. the source and package-export blobs still match the Phase 3D-0 baseline;
2. the exact canonical, causal-derivative, and wrapper consumer sets remain
   visible;
3. package imports resolve to their distinct implementation modules under
   dependency stubs;
4. loader selection, constructor defaults, forward selection, pooling, and
   wrapper delegation match the recorded legacy behavior;
5. the Phase 3D-1 diff contains only its three additions and the update that
   makes the Phase 3D-0 test read its immutable completion tree.

These checks do not prove that two real Transformer models produce equivalent
hidden states, embeddings, gradients, losses, or generations. They also do not
validate a combined dependency environment.

## Remaining gates

`llm22vec` removal and source movement remain blocked. The next source-layout
batch should introduce a project-owned adapter boundary while preserving these
imports and behaviors. Real-model equivalence, dependency reconciliation,
third-party attribution, and project licensing require separate review.

Run the contract tests with:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python -m unittest -v tests.test_llm2vec_migration_contract
```
