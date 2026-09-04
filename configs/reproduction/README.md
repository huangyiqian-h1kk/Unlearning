# Future portable runs

These records describe the `validated_v2` contract for new measurements. They
are deliberately separate from the byte-identical selected paper capsules in
[`experiments/paper_runs/`](../../experiments/paper_runs/).

The current candidate is not declared runnable: dependency, dataset-rights,
checkpoint-lineage, model smoke-test, and numerical-protocol gates remain
false. When those gates close, write outputs under `results/validated_v2/`,
never over `results/paper/`.

Validate structure offline with:

```bash
python scripts/configs/validate_reproduction_configs.py \
  --index configs/reproduction/index.json
```
