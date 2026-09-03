# Dependency boundaries

Phase 3D-3 records dependency facts without downloading packages or claiming a
model environment was executed.

The vendored LLM2Vec packaging metadata allows Transformers
`>=4.43.1,<=4.44.2`. The nested OpenUnlearning snapshot pins Transformers
`==4.45.1`. Their intersection is empty. A single merged model environment
would therefore be misleading and is not published.

The machine-readable record is
[`dependency_matrix.json`](dependency_matrix.json). It copies declarations
from the tracked `setup.py` and `requirements.txt` files, anchors upstream code
revisions and license identities, and records the conflict explicitly.

## Environment policy

- Archived metric extraction and table construction use the Python standard
  library and do not need either model stack.
- Repository unit tests exercise model-facing contracts with lightweight
  dependency stubs; they are not numerical model validation.
- LLM2Vec-facing work and OpenUnlearning-facing work must use isolated
  environments built from their own tracked declarations.
- `validated_v2` remains non-runnable until an environment, dataset access,
  checkpoint lineage, and evaluation protocol are reviewed together.
- Historical configurations remain immutable even where they contain private
  absolute paths or obsolete runtime assumptions.

See [`../environments/README.md`](../environments/README.md) for preparation
guidance and [`source_ownership_audit.md`](source_ownership_audit.md) for the
upstream comparison method.
