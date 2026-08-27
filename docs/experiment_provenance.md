# Experiment provenance

The machine-readable source of truth is [`results/paper/manifest.json`](../results/paper/manifest.json). It records archive and member hashes, exact tar member names, resolved semantic identities, status flags, starting checkpoints, and evidence limitations.

## Principal findings

- All five selected ConRep runs have `lm_weight = 0`; the auxiliary LM term did not contribute to these released main-result runs.
- The selected runs completed fixed configured epoch counts. Archived trainer states do not demonstrate automated early stopping or best-checkpoint selection.
- Historical `Llama3` labels on the selected Regime A runs resolve to `meta-llama/Llama-2-7b-chat-hf` from configuration-generation evidence and are marked `label_corrected`.
- Historical `Mixtral` PMC labels resolve to `mistralai/Mistral-7B-Instruct-v0.2` in Hydra configurations and are marked `label_corrected`.
- PMC GradDiff starts from the earlier CSV checkpoint, while the universal baseline, NPO, RMU, and selected ConRep use the universal continuation. GradDiff is retained but marked `non_comparable`.
- Exact PMC generation pooling is `(900 × retain + 100 × forget) / 1000`. MCQ sizes are verified as retain `100/100/100` and forget `50/50/46` for ATT/IDeq/ID. The checkout could not reach the public LFS transport, so these six counts use the reviewed pointer-OID inventory; the five Regime A datasets were counted locally.
- Full-precision baseline accuracies yield exact pooled PMC successes ATT `54/150`, IDeq `48/150`, and ID `41/146`. ATT and IDeq exceed chance; ID does not and is masked. Regime A masking is model-specific: Llama-2 masks Deaths IDeq and Diagnosis ID; Mistral masks Deaths IDeq, Deaths ID, and Diagnosis ID.
- Row averages exclude MMLU, missing cells, significance-masked cells, and—in Regime B—every derived Delta column.
- Each selected ConRep record cites and parses its tracked JSON configuration and job script. Terminal/root `trainer_state.json` is selected once per trained run; fixed-epoch completion is not described as early stopping.
- The two intended RMU/Deaths cells are quarantined: intended identity is separate from the source-resolved identities (Mistral/Diagnosis/10 epochs and Llama-2/Deaths/3 epochs respectively), and neither source populates a canonical cell.

Private cluster paths present inside immutable evidence are normalized in the manifest to semantic checkpoint and dataset identifiers.
