# Archived results policy

No training, inference, evaluation, or model download was rerun to prepare the release tables. The two root tar archives are immutable experimental evidence. The extraction tool verifies archive and member SHA-256 values and writes compact metric snapshots; it does not restore checkpoints, logs, tokenizers, weights, or detailed generations.

Resolved configuration evidence takes precedence over historical directory and task labels. Missing, unresolved, and non-comparable cells remain visibly marked rather than being regenerated or silently replaced. Table reconstruction from archived measurements is distinct from complete end-to-end training reproduction.

The compact snapshots under `results/paper/raw_metrics/` are the durable input to table construction. Once created, table building does not read the archives or historical directories.
