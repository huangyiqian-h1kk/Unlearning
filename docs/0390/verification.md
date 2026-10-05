# Implementation verification — 0390

Verification date: 2026-10-05. This records code and execution checks, not experiment results.

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
  `nvcc`, and imports DeepSpeed once before spawning SFT workers. Its actual
  GPU/ZeRO-2 validation still requires a new SFT smoke job.
- Two local shell integration tests pass (`python tests/0390/test_abci_env.py -v`).
  They execute the batch wrapper with a simulated module environment, checking
  that the toolkit, venv and project-local caches reach Python, and that a missing
  compiler prevents Python startup. These tests do not execute CUDA or DeepSpeed.

## Not yet verified

The two-process Gloo test is skipped only when this execution host specifically
rejects the socket setup with `Operation not permitted`; other failures still fail
the test. The server Llama 3B smoke above covers NCCL and the eight-rank
contrastive gather/backward. ZeRO-2 SFT and Qwen 7B GPU execution remain unverified.

The local checkout contains Git LFS pointers; the user materialized the original
assets on ABCI and ran PMC preparation as recorded above. Selected SFT quality,
FALCON MI selection and ReLearn teacher-generated augmentation remain unverified.
Tiny-model tests use synthetic fixtures and are not evidence of baseline quality.

No new scientific benchmark results have been measured. ABCI commands and jobs
are executed by the user; the server evidence above comes from their supplied logs.
The walltimes remain initial requests to adjust from server measurements. Baseline
ports preserve the named core algorithms with adaptations listed in
[baselines.md](baselines.md); their numeric equivalence and final hyperparameter
tuning remain experimental work.
