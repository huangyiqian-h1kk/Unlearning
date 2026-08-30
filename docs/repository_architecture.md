# Repository architecture record

This document records the target architecture; it does not claim that historical end-to-end reproduction is complete.

## Ownership boundaries

ConRep is project-owned method code and will eventually live in `src/conrep/`. ClinicIA's data registry, adapters, and shared evaluation are project-owned benchmark code and will eventually live in `src/clinicia/`. Future ConRep and baseline measurements must use the same versioned ClinicIA protocol.

GradDiff, NPO, and RMU remain backed by OpenUnlearning and must not be copied into `src/conrep/methods/`. The target is a pinned, attributable OpenUnlearning snapshot plus a project-owned ClinicIA adapter/registration layer. Dataset handlers, configurations, scripts, logs, and debug edits currently mixed into that tree are not all upstream source. Source-tree comparison identifies the inferred base as `locuslab/open-unlearning@d33c4762941731cb397494f605f71528d8ce9dce`; this is an inference, not direct Git-ancestry evidence. The audit comparison observed the expected 204 common paths (199 identical, 5 modified), 104 project-only additions, and no upstream-only paths. Modified paths are `configs/eval/tofu.yaml`, `configs/experiment/finetune/tofu/default.yaml`, `configs/trainer/finetune.yaml`, `src/data/__init__.py`, and `src/trainer/unlearn/npo.py`.

LLM2Vec is an upstream dependency with a small project-specific causal/generative ConRep adapter. The inferred comparison base is `McGill-NLP/llm2vec@312adcfb578b4023bfc9d3bcc83b6b87448ab059`. The audit comparison observed the expected 104 common paths (99 identical, 5 modified) and one upstream-only path. Modified paths are `experiments/run_simcse.py`, `llm2vec/loss/HardNegativeNLLLoss.py`, `llm2vec/loss/__init__.py`, `llm2vec/loss/utils.py`, and `train_configs/simcse/Mistral.json`. Future organization should pin upstream and isolate the causal/generative adapter. Upstream code must remain distinguishable from project modifications, and duplicated `llm22vec` cannot be removed until adapter-equivalence tests exist.

## Configuration and data policy

`configs/historical/` stores evidence-backed, possibly incomplete and non-runnable records. Future `configs/reproduction/` will hold validated runnable configurations; `configs/components/` reusable model, dataset, method, and evaluation fragments; and `configs/sweeps/` compact grids whose expanded products are ignored.

Dataset IDs and manifests replace private absolute paths. Training corpora remain separate from evaluation probes. LFS object IDs and hashes must be retained, while data movement waits for redistribution and license review. ConRep and OpenUnlearning may have different adapters, but must resolve the same versioned dataset IDs and splits.

## Evaluation, results, and artifacts

`historical_v1` documents compatibility with archived measurements. `validated_v2` will correct future evaluation without changing historical paper measurements. MMLU uses a separately pinned `lm_eval` configuration. Archived paper evidence therefore remains distinct from corrected future protocols.

Both root tar archives remain immutable at their current paths. `results/paper/` remains the tracked paper-result layer. Checkpoints, full logs, expanded sweeps, generated outputs, scheduler logs, and detailed generations belong in ignored local or external artifact storage. Table reconstruction remains independent of training and expanded historical directories.
