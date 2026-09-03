# Configuration layers

`historical/` contains immutable, evidence-backed descriptions of past
experiments. These records can be incomplete and are not promises that a run is
executable.

Phase 3D-3 adds three separate future-facing layers:

- `components/` provides portable model, method, dataset, and protocol IDs;
- `reproduction/` contains structurally validated candidate run descriptions;
- `sweeps/` contains compact review matrices and never expanded job products.

The candidate reproduction records are deliberately marked `runnable: false`.
Structural validation proves references and portability, not dependency,
checkpoint, dataset, compute, or numerical readiness.

Validate the paper record set with:

```bash
python scripts/configs/validate_historical_experiments.py \
  --index configs/historical/paper/index.json \
  --manifest results/paper/manifest.json
```

Validate future-facing records separately with:

```bash
python scripts/configs/validate_reproduction_configs.py \
  --index configs/reproduction/index.json
```
