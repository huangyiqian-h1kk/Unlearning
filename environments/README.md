# Environment boundaries

There is intentionally no single model-environment lock file in this release.
The two preserved upstream-derived stacks declare incompatible Transformers
versions.

For LLM2Vec-facing work, start from `llm2vec/setup.py`. For
OpenUnlearning-facing work, start from
`llm2vec/open_unlearning/requirements.txt` and its `setup.py`. Build those in
separate environments and review hardware-specific packages such as
FlashAttention, BitsAndBytes, and DeepSpeed for the target platform.

These declarations preserve dependency evidence; they are not a promise that
the historical experiments can be rerun. Do not install model dependencies as
part of lightweight artifact validation. Exact declarations and the known
conflict are recorded in [`docs/dependency_matrix.json`](../docs/dependency_matrix.json).
