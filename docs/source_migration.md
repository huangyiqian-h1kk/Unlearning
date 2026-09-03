# Phase 3D-2 canonical source migration

Phase 3D-2 separates project-owned ConRep and ClinicIA Python source from the
vendored LLM2Vec/OpenUnlearning trees. It is a path migration, not a scientific
refactor: every moved implementation retains its exact Git blob from the
Phase 3D-1 completion tree.

Machine-readable paths and blob identities are recorded in
[`source_migration_manifest.json`](source_migration_manifest.json).

## Canonical ownership

| Component | Canonical location | Preserved source | Stable launcher |
| --- | --- | ---: | --- |
| ConRep | `src/conrep/` | 12 training variants under `src/conrep/legacy/` | `scripts/train_conrep.py` |
| ClinicIA | `src/clinicia/` | 11 evaluation modules under `src/clinicia/legacy/` | `scripts/evaluate_clinicia.py` |

The `legacy/` label means that the implementation bytes and historical
behavior are preserved. It does not mean the code is third-party. Both trees
are project-owned; the label distinguishes preserved paper-era programs from
new stable registry and protocol interfaces.

## Compatibility and traceability

Every moved file keeps a compatibility launcher at its former path. The
launcher compiles and executes the canonical file in the existing namespace,
so archived commands continue to select the same source implementation. In
particular, the five protected historical ConRep records still refer to:

```text
llm2vec/ContrastiveUnlearning_Adaptive_RandomToken_LMloss_margin.py
```

That path now redirects to the blob-identical canonical source at:

```text
src/conrep/legacy/ContrastiveUnlearning_Adaptive_RandomToken_LMloss_margin.py
```

The known syntax error in the epoch-evaluation variant is intentionally moved
without repair and remains recorded at line 581. Phase 3D-2 does not silently
change an unvalidated historical program.

The new launchers can enumerate their targets without importing Torch,
Transformers, PEFT, or loading a model:

```bash
python scripts/train_conrep.py --list
python scripts/evaluate_clinicia.py --list
```

Actual training and evaluation still require an explicitly validated runtime;
the launchers do not claim that such an environment exists.

## ClinicIA registry and protocols

`src/clinicia/registry.py` assigns stable semantic IDs to the eleven datasets
already anchored by `results/paper/mcq_dataset_inventory.json`.
`src/clinicia/adapters.py` verifies the recorded SHA-256 and record count before
returning JSONL records. It never downloads LFS objects; a verified pointer is
reported as not materialized.

Two disjoint protocol namespaces are explicit:

- `historical_v1` describes compatibility with archived paper evidence under
  `results/paper/`;
- `validated_v2` reserves corrected future evaluation under
  `results/validated_v2/` and is deliberately marked not runnable.

This prevents future measurements from overwriting or being confused with the
paper-era evidence.

## Deliberately unchanged

Phase 3D-2 does not edit:

- `configs/historical/`, `results/paper/`, or either root archive;
- the canonical `llm2vec` package, causal `llm22vec` derivative, or the nested
  OpenUnlearning tree;
- dependency declarations, model behavior, dataset bytes, or reported metrics.

Dependency reconciliation, any `llm22vec` consolidation, portable reproduction
configuration, dataset/license release decisions, and the root README remain
Phase 3D-3 work.
