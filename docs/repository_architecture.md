# Repository architecture

The current layout follows research ownership rather than the directory in
which a script happened to be written.

| Layer | Paths | Rule |
| --- | --- | --- |
| Project method | `src/conrep/` | ConRep concepts, stable dispatch, preserved implementations |
| Project benchmark | `src/clinicia/`, `data/clinicia/` | ClinicIA probes, metrics, registry, and data |
| Selected experiments | `experiments/paper_runs/` | Only evidence-identified paper runs |
| Paper evidence | `configs/paper/historical/`, `results/paper/` | Immutable historical records and reconstruction |
| Vendored dependencies | `third_party/` | Upstream-derived snapshots with licenses and audited local deltas |
| Non-selected history | `legacy/` | Exploratory/superseded code; not a public entry point |

The old `llm2vec/` root mixed all six layers. Phase 3D-R moves blobs without
changing scientific data, then adds a small project-owned explanation and
dispatch layer. Historical evidence may still mention old paths; current tools
resolve them through explicit catalogs instead of rewriting provenance.

See [`paper-to-code.md`](paper-to-code.md) for scientific concepts and
[`reproduction.md`](reproduction.md) for commands.
