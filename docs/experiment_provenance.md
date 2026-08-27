# Experiment provenance

The machine-readable source of truth is [`results/paper/manifest.json`](../results/paper/manifest.json). It records archive and member hashes, exact tar member names, resolved semantic identities, status flags, starting checkpoints, and evidence limitations.

## Principal findings

- All five selected ConRep runs have `lm_weight = 0`; the auxiliary LM term did not contribute to these released main-result runs.
- The selected runs completed fixed configured epoch counts. Archived trainer states do not demonstrate automated early stopping or best-checkpoint selection.
- Historical `Llama3` labels on the selected Regime A runs resolve to `meta-llama/Llama-2-7b-chat-hf` from configuration-generation evidence and are marked `label_corrected`.
- Historical `Mixtral` PMC labels resolve to `mistralai/Mistral-7B-Instruct-v0.2` in Hydra configurations and are marked `label_corrected`.
- PMC GradDiff starts from the earlier CSV checkpoint, while the universal baseline, NPO, RMU, and selected ConRep use the universal continuation. GradDiff is retained but marked `non_comparable`.
- Exact PMC generation pooling is `(900 × retain + 100 × forget) / 1000`. Exact pooled MCQ success counts are not present in the aggregate metrics, so exact significance masking is unresolved.

Private cluster paths present inside immutable evidence are normalized in the manifest to semantic checkpoint and dataset identifiers.
