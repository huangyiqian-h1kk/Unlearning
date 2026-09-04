# Third-party components

Vendored dependency code is isolated under `third_party/`; the project-facing
API is under `src/`. These are historical working snapshots anchored to
inferred upstream revisions, not falsely presented as pristine upstream trees.

| Component | Current path | Upstream | Inferred revision | Audited relation | License |
| --- | --- | --- | --- | --- | --- |
| LLM2Vec | `third_party/llm2vec/` | McGill-NLP/llm2vec | `312adcfb578b4023bfc9d3bcc83b6b87448ab059` | 104 common paths: 99 identical, 5 modified; one upstream-only path absent | [MIT](third_party/llm2vec/LICENSE) |
| OpenUnlearning | `third_party/open-unlearning/` | locuslab/open-unlearning | `d33c4762941731cb397494f605f71528d8ce9dce` | 199 identical common paths, 5 modified common paths, and 70 retained local-only integration paths | [MIT](third_party/open-unlearning/LICENSE) |

The exact comparisons are frozen in
[`docs/repository_source_inventory.json`](docs/repository_source_inventory.json).
Use the inferred revisions to inspect provenance, not to claim that these
directories are clean upstream checkouts.

The causal `llm22vec` adapter is project-specific and now lives at
`src/conrep/backends/llm22vec/`. It reuses byte-identical support modules from
the pinned LLM2Vec package through an explicit package-path fallback.

Historical baseline launch scripts are preserved separately under
`legacy/open_unlearning_jobs/`; their presence is provenance, not proof that
they produced a selected paper result.

The two upstream licenses do not choose a license for project-owned ConRep,
ClinicIA, documentation, or data. The absence of a root project license remains
an explicit release decision.
