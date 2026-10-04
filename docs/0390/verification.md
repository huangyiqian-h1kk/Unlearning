# Implementation verification — 0390

Verification date: 2026-10-04. This records code checks, not experiment results.

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

## Not yet verified

The two-process Gloo test is skipped only when this execution host specifically
rejects the socket setup with `Operation not permitted`; other failures still fail
the test. NCCL, the eight-rank contrastive gather/backward and ZeRO-2 SFT must be
tested on ABCI. The supplied `0390_smoke` performs dependency/data/tokenizer checks,
GPU all-reduce and two actual ConRep updates on the selected 3B/7B backbone.

The checkout contains Git LFS pointers for the released ClinicIA assets. Actual
PMC parsing, metadata joins, selected SFT quality, FALCON MI selection and ReLearn
teacher-generated augmentation have not been validated here. Tiny-model tests
use synthetic fixtures and are not evidence of data coverage or baseline quality.

No ABCI login, environment installation, GPU allocation, `qsub`, real-backbone
training or new benchmark measurement has been performed. The walltimes are initial
requests to adjust after the first server logs. Baseline ports preserve the named
core algorithms with adaptations listed in [baselines.md](baselines.md); their
numeric equivalence and final hyperparameter tuning remain experimental work.
