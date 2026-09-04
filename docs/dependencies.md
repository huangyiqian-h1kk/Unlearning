# Dependency boundaries

There is no single truthful model environment for every historical component.

| Component | Path | Key constraint |
| --- | --- | --- |
| Project metadata/table tools | root + `src/` | Python 3.10+, standard library for table rebuild |
| ConRep + pinned LLM2Vec | `src/conrep/`, `third_party/llm2vec/` | Transformers 4.43.1–4.44.2 in the pinned package |
| Historical baselines | `third_party/open-unlearning/` | Transformers 4.45.1 and its pinned stack |

The LLM2Vec and OpenUnlearning Transformers constraints do not intersect.
Prepare separate environments and choose one explicitly. No repository command
downloads a model, creates an environment, or performs an LFS pull
automatically.

The exact upstream revisions, requirements, and license identities are in
[`dependency_matrix.json`](dependency_matrix.json).
