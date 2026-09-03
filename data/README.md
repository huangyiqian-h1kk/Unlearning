# Data release inventory

The scientific data remains at its historical tracked paths so provenance and
existing configuration evidence are not rewritten. This directory contains a
release-oriented inventory rather than duplicate payloads.

[`lfs_manifest.json`](lfs_manifest.json) records the release policy and exact
base tree for every tracked Git LFS pointer under `llm2vec/UnlearnData/` and
`llm2vec/cache/`. The release validator compares current pointer paths and Git
blobs directly with that immutable base tree, so per-object identifiers and
sizes do not need to be republished in a second inventory. It works even when
LFS smudge has materialized a working-tree file.

Important boundaries:

- an LFS pointer is identity evidence, not proof that its object is locally or
  publicly retrievable;
- no LFS download is performed by validation;
- all redistribution statuses remain unresolved and default to “do not
  redistribute” pending researcher/legal review;
- model, dataset, publication, privacy, and clinical-data terms can be
  independent of source-code licenses;
- the eleven paper MCQ datasets have stable semantic IDs through
  `src/clinicia/registry.py` and integrity facts through
  `results/paper/mcq_dataset_inventory.json`.

No dataset bytes were moved or edited in Phase 3D-3.
