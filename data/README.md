# Data layout and availability

Paper-facing data is organized semantically under
[`data/clinicia/`](clinicia/README.md):

- `regime_a/diagnosis/` and `regime_a/deaths/` for pre-existing celebrity
  attributes;
- `regime_a/shared/retain/` for complementary retention data;
- `regime_b/pmc/` for injected clinical relations and retain/forget probes.

`data/clinicia/catalog.json` maps all 28 files back to their historical
`llm2vec/` paths. `data/legacy/` contains unrelated cyber/wiki artifacts
retained only for provenance.

Twenty-five files are Git LFS pointers. Their current and historical paths,
pointer blob identity, content OID, and size are recorded in
`lfs_manifest.json`. The repository does not automatically fetch them and
does not claim redistribution permission.
