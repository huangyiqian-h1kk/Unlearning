# Phase 3D-3 release integration

Phase 3D-3 integrates the reviewed repository surfaces needed for a readable,
auditable research artifact. It does not run or reinterpret the science.

## Completed in this phase

- eighteen byte-identical `llm22vec` support modules are replaced by package
  fallback to the canonical `llm2vec` implementations;
- the behaviorally distinct causal core, public import, and OpenUnlearning
  wrapper remain in `llm22vec`;
- incompatible model dependency surfaces are recorded and intentionally kept
  separate;
- portable component, reproduction-candidate, and compact sweep records gain
  an offline structural validator;
- all tracked LFS pointer paths and Git blobs are checked against the immutable
  Phase 3D-2 tree without downloading data or republishing per-object metadata;
- third-party attribution, local modifications, unresolved project licensing,
  and unresolved dataset redistribution rights are explicit;
- the reviewed README draft becomes the root README and now describes the
  post-Phase-3D-2 layout;
- manuscript placement is reserved without inventing a citation.

## Deliberately not completed

No real dependency installation, model initialization, training, inference,
numerical equivalence run, LFS download, or scheduler work occurred. Candidate
`validated_v2` configuration remains non-runnable. No license was selected for
project-owned code, and no dataset redistribution decision was inferred.

`configs/historical/`, `results/paper/`, and both root evidence archives remain
scientifically protected. Phase 3E must perform final clean-clone, link,
licensing, recovery, and researcher-sign-off checks before the overall
reorganization can be called complete.
