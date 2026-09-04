# Model environments

Run commands from the repository root. Archived table reconstruction needs only
Python 3.10+; model work needs two isolated environments because LLM2Vec pins
`transformers>=4.43.1,<=4.44.2` while the vendored OpenUnlearning snapshot pins
`transformers==4.45.1`.

## ConRep, ClinicIA, and MMLU

```bash
conda create -n conrep-clinicia python=3.11 -y
conda activate conrep-clinicia
pip install -e third_party/llm2vec
pip install -e .
pip install lm-eval==0.4.8
pip install --no-build-isolation flash-attn
```

- **Required inputs:** a host-compatible CUDA/PyTorch stack; Hugging Face access
  to the selected Llama-2/Mistral and LLM2Vec adapters.
- **Output:** environment only.
- **Paper experiments:** five ConRep cells, all ClinicIA evaluations, and MMLU.
- **Verification:** package boundaries and dependency constraints are validated
  offline. No public-release GPU smoke test has been recorded.

## PMC SFT and OpenUnlearning baselines

```bash
conda create -n conrep-openunlearning python=3.11 -y
conda activate conrep-openunlearning
pip install -e './third_party/open-unlearning[lm-eval]'
pip install --no-build-isolation flash-attn==2.6.3
```

- **Required inputs:** a CUDA stack compatible with pinned `torch==2.4.1`.
- **Output:** environment only.
- **Paper experiments:** Regime B PMC SFT and the 15 GradDiff/NPO/RMU cells.
- **Verification:** the complete pinned requirement set is committed and
  validated offline. No public-release GPU smoke test has been recorded.

## Data and model caches

The public commands accept explicit dataset, model, and output paths. Standard
Hugging Face variables such as `HF_HOME` may be set for local cache placement,
but no repository command assigns a machine-specific cache path or downloads a
model automatically.

Before training, run:

```bash
python scripts/reproduce.py data-status --require-materialized
```

The five selected historical jobs requested one GPU, bf16, and
FlashAttention 2. Their exact GPU model, CUDA module, and VRAM headroom were not
preserved, so this repository does not state an invented hardware guarantee.
The historical environment commands and old absolute paths remain visible in
`experiments/paper_runs/*/historical/job.sh` as provenance only.
