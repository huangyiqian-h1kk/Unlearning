# Configuration layers

`historical/` contains immutable, evidence-backed descriptions of past experiments. These records can be incomplete and are not promises that a run is executable. Future runnable reproduction, reusable component, and sweep configurations will live in separate directories after validation.

Validate the paper record set with:

```bash
python scripts/configs/validate_historical_experiments.py \
  --index configs/historical/paper/index.json \
  --manifest results/paper/manifest.json
```
