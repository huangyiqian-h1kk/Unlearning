# Implementation verification — 0390

Verification date: 2026-10-05. This records code and execution checks, not experiment results.

## Llama 8B single-allocation SFT pipeline

- Added the Llama-3.1-8B-Instruct configuration and `sft-pipeline` stage. One
  coordinator launches SFT and then a fresh validation process group in the same
  PBS allocation; selection runs only after every validation worker succeeds.
- Nine new CPU tests cover process ordering, launcher dispatch, PBS resource/name
  settings, real constraint/tie-break checkpoint selection, worker failures,
  missing validation reports, no eligible checkpoint, completed-training reuse,
  output protection and job-ID-linked artifact capture.
- The tests exercise JSON configuration reload through the real config loader:
  scientific-notation learning rates remain numbers when workers load a snapshot.
- The existing 7 submitter, 9 log, 2 environment, 6 HG validation and 6 MMLU
  protocol tests also pass: 39 relevant CPU tests in total.
- Python compilation, PBS/shell syntax and patch whitespace checks pass. These
  are orchestration checks; actual Llama 8B GPU memory, runtime and SFT quality
  remain to be measured on ABCI. No model weights or private run results were
  generated or published by these checks.

```bash
python tests/0390/test_sft_pipeline.py -v
```

Commands, expected outputs and restart behavior: [llama8b-sft.md](llama8b-sft.md).

## Job-log management checks (2026-10-05)

- Nine new CPU-only tests in `tests/0390/test_joblogs.py` pass. They exercise the
  actual PBS shell wrapper with a simulated module/venv and real child processes:
  merged live stdout/stderr, successful exit, a child exit of 7, CUDA setup failure,
  SIGTERM forwarding and exit 143, and lightweight artifact/config capture.
- Archive tests cover historical `$HOME` log recovery without deleting originals,
  filtered PBS JSON, unavailable scheduler history, immutable result snapshots,
  distinct PBS IDs for repeated run labels, a job starting before its submission
  record is written, and archive failure after successful qsub without inviting a
  duplicate submission. Environment/PBS secret fields are excluded.
- The two existing ABCI environment tests and seven submitter tests also pass:
  18 related tests in total. Shell syntax and patch whitespace checks pass.
- Private history ledgers can be imported locally. Full raw logs and live PBS
  records must be collected on ABCI using the command in [logging.md](logging.md);
  these local tests do not claim server-side recovery or publish private ledgers.

```bash
python tests/0390/test_joblogs.py -v
python tests/0390/test_abci_env.py -v
python tests/0390/test_abci_submit.py -v
```

## Completed locally

- A Python 3.11.16 environment was installed with the pinned training dependencies,
  including Transformers 4.48.3, PEFT 0.14.0, Accelerate 1.3.0 and DeepSpeed 0.16.2.
  Local PyTorch was the CPU build of 2.5.1; `pip check` reported no conflicts.
- `tests/0390/test_core.py`: 16 passed, 1 skipped. The tests initialize small real
  Llama/Qwen causal models, perform optimization and load the saved artifacts.
- ConRep updates run on both model families. Resuming checkpoint 1 reproduces
  checkpoint 2 adapter weights within absolute tolerance `1e-7`.
- Each of NPO, RMU, FALCON, LUNAR, SAGO and ReLearn executes a real parameter
  update. FALCON uses the vendored SophiaG optimizer. LUNAR's saved model changes
  only the selected block's down-projection weight.
- Ordinary SFT and retain-only SFT execute; QA prompt tokens are loss-masked.
  Multi-format data fixtures preserve injection content and distinguish patient
  identifiers such as `P1` from `P10` when constructing retain-only data.
- Corruption protects structural tokens, actually changes sampled token IDs,
  and varies independently across views and examples. The contrastive test checks
  the exact one-positive loss and gradients to corrupted/retain branches.
- Generation validation decodes only new tokens. Cache reuse rejects a changed
  evaluation protocol. Checkpoint selection rejects utility-violating candidates.
- PBS rendering checks project/queue/RTYPE, `0390` names, argument quoting and
  the two-job limit. Shell syntax and first-party patch whitespace checks pass.
  Vendored source retains upstream whitespace to preserve its recorded hashes.
- The repository's 121 pre-existing unittest checks pass.

The CPU test command is:

```bash
python -m pip install -r environments/0390/requirements-test.txt
python -m pytest tests/0390/test_core.py -q
python -m unittest discover -s tests -p 'test_*.py' -q
```

On the current execution host, DeepSpeed's CPU auto-detection encounters restricted
process information (`psutil.NoSuchProcess`). The local Python 3.11 test process
therefore used `DS_ACCELERATOR=cuda` solely to select its import implementation;
all model tensors and tested updates remained on CPU. This setting is **not**
added to ABCI scripts and does not verify CUDA or DeepSpeed execution.

## ABCI evidence supplied by the user (2026-10-05)

- The server venv passes 16 CPU tests (the distributed test was deselected).
- Both backbone downloads and local tokenizer/data preflight completed. PMC data
  counts are forget 100, retain 900, injection 3,000, retain-only injection 2,700,
  with zero unresolved injection rows; the deduplicated general pool has 49,959 rows.
- Job `2505727.pbs1` (`0390_l3smk1`) completed in 1m45s with exit status 0.
  Its eight-rank Llama 3B ConRep smoke reports `all_reduce: passed` and two updates;
  this execution includes adapter/checkpoint saving. The tail also contains a
  non-fatal warning about process-group destruction during exit.
- Job `2505728.pbs1` (`0390_l3sft1`) exited with status 1 during DeepSpeed import,
  before training, because the launch script had not initialized `CUDA_HOME`.
  The startup now loads ABCI's CUDA 12.4.1 module (matching PyTorch cu124), checks
  `nvcc`, and imports DeepSpeed once before spawning SFT workers.
- The rerun `2505944.pbs1` (`0390_l3sft2`, commit `a460d4e6`) completed in 2m03s
  with exit status 0. Its log confirms CUDA Toolkit 12.4.1, DeepSpeed 0.16.2,
  PyTorch CUDA 12.4, eight training ranks and NCCL initialization. Losses were
  4.8771 and 4.5173; `TRAINING_COMPLETE.json` reports `global_step: 2` and the
  saved final model under `llama3b/gpu-sft-smoke-v2/final`. This verifies the
  Llama full-SFT/ZeRO-2 execution and save path, not knowledge-injection quality.
- Two local shell integration tests pass (`python tests/0390/test_abci_env.py -v`).
  They execute the batch wrapper with a simulated module environment, checking
  that the toolkit, venv and project-local caches reach Python, and that a missing
  compiler prevents Python startup. These tests do not execute CUDA or DeepSpeed.
- Resource selection now supports `rt_HF` and `rt_HG` in the same reserved queue.
  `validate`, `analyze`, `falcon-layers` and `relearn-augment` default to HG;
  training and parallel checkpoint-series validation retain their HF defaults.
  This changes allocation only: training batch sizes and objectives are unchanged.
  Six local submitter tests pass (`python tests/0390/test_abci_submit.py -v`),
  covering default allocations, explicit overrides, preserved training arguments,
  the one-GPU HG limit and rejection of multiple writers for unsharded evaluation.
- Llama formal SFT `2506096.pbs1` (`0390_l3sft42`) exited 0 in 8m48s;
  the completion marker reports 465 updates and the saved final model.
- Qwen SFT smoke `2506097.pbs1` exited 0 in 3m23s with a two-step completion
  marker. Formal Qwen SFT `2506252.pbs1` (`0390_q7sft42`) then exited 0 in
  15m25s, completed 465 updates and saved the final model. The logged loss fell
  from 5.1495 to 0.3007; this is execution evidence, not validation quality.
- Llama original-backbone validation `2506253.pbs1` (`0390_l3baseval`) ran on
  HG and exited 0 in 10m18s. PMC QA/cloze/background accuracies were zero on
  both pools; current-protocol MMLU was 0.3605263158 (57 subjects, 20 each).
  Because this original-backbone score is unexpectedly low, checkpoint selection
  is deferred pending an answer-format diagnostic. It is not an SFT regression.
- The new `audit-mmlu` stage compares the unchanged likelihood scorer with
  short generated answers on a fixed balanced subset. Five CPU-only tests cover
  sampling, strict extraction, invalid-output accounting, aggregation and CLI;
  the six submitter tests also cover its single-GPU HG default.
- Audit jobs `2506422.pbs1` (Llama, 1m13s) and `2506423.pbs1` (Qwen, 1m40s)
  both exited 0 on HG and evaluated 285 rows across 57 subjects. Llama accuracy
  changed from 0.3438596491 (old likelihood) to 0.5824561404 (instructed generation);
  Qwen changed from 0.6807017544 to 0.7122807018. The latter mode had one invalid
  Llama output and zero invalid Qwen outputs. Neither model had input truncation.
  Uninstructed short generation was unparseable for all rows in both models;
  supplied raw outputs begin explanations or attempt all questions. Llama's old
  likelihood predicted A 227/285 times, with mean A/B/C/D probability mass 0.050449.
- Formal validation now shares the audited instruction/parser, greedy decoding,
  ten-token generation limit and context budget. It uses all 1,140 fixed MMLU
  validation rows and saves raw MMLU generations and invalid/truncation diagnostics.
  The protocol hash includes the shared MMLU code and rejects old caches. PMC
  prompts and scoring and SFT weights are unchanged.
- `validate-series --include-backbone` places original-model scores in `base/`
  alongside checkpoint scores and assigns distinct candidates to distinct ranks.
  Six CPU tests in `test_mmlu_protocol.py`, five audit tests and seven submitter
  tests pass (18 total). They check the exact prompt/token budget, macro averaging,
  invalid denominators, output files, cache incompatibility, tokenizer restoration
  and rank assignment. Generation is stubbed in these new integration tests;
  the actual generator/prompt ran in the two ABCI audits above. Full-series GPU
  validation with the integrated protocol is the next server step.

## Not yet verified

The two-process Gloo test is skipped only when this execution host specifically
rejects the socket setup with `Operation not permitted`; other failures still fail
the test. The server Llama 3B smoke above covers NCCL and the eight-rank
contrastive gather/backward; the SFT rerun covers Llama's ZeRO-2 execution.
The supplied logs now verify Qwen GPU SFT, HG inference, and both full-length SFT
runs. Knowledge-injection quality and retained utility remain unverified.

The local checkout contains Git LFS pointers; the user materialized the original
assets on ABCI and ran PMC preparation as recorded above. Selected SFT quality,
FALCON MI selection and ReLearn teacher-generated augmentation remain unverified.
Tiny-model tests use synthetic fixtures and are not evidence of baseline quality.

No new unlearning results or selected SFT checkpoints are available yet. ABCI commands and jobs
are executed by the user; the server evidence above comes from their supplied logs.
The walltimes remain initial requests to adjust from server measurements. Baseline
ports preserve the named core algorithms with adaptations listed in
[baselines.md](baselines.md); their numeric equivalence and final hyperparameter
tuning remain experimental work.
