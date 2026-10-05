#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ -f local.env ]]; then source local.env; fi
: "${CONREP_ENV:?Set CONREP_ENV in local.env}"
: "${CONREP_WORK_ROOT:?Set CONREP_WORK_ROOT under /groups}"
# The supervisor starts before CUDA initialization, so setup failures are logged too.
# Only this outer process writes job metadata; torchrun ranks share its console log.
if [[ -n "${PBS_JOBID:-}" && "${CONREP_LOG_ACTIVE:-0}" != "1" ]]; then
    exec "$CONREP_ENV/bin/python" scripts/abci/0390_logs.py run -- "$@"
fi
# The PyTorch wheel includes CUDA runtime libraries, but DeepSpeed also probes
# the system toolkit when imported on a GPU node. Initialize it before Python.
if ! type module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh
fi
module load "${CONREP_CUDA_MODULE:-cuda/12.4/12.4.1}"
: "${CUDA_HOME:?The CUDA module must set CUDA_HOME}"
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    printf '[0390] CUDA compiler not found: %s/bin/nvcc\n' "$CUDA_HOME" >&2
    exit 1
fi
source "$CONREP_ENV/bin/activate"
printf '[0390] CUDA_HOME=%s\n' "$CUDA_HOME"
"$CUDA_HOME/bin/nvcc" --version
export HF_HOME="${CONREP_HF_CACHE:-$CONREP_WORK_ROOT/cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${CONREP_OMP_THREADS:-8}"
export PYTHONUNBUFFERED=1
export TORCH_EXTENSIONS_DIR="$CONREP_WORK_ROOT/cache/torch_extensions"
export TRITON_CACHE_DIR="$CONREP_WORK_ROOT/cache/triton"
mkdir -p "$CONREP_WORK_ROOT/tmp" "$HF_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"
export TMPDIR="$CONREP_WORK_ROOT/tmp"
exec "$CONREP_ENV/bin/python" scripts/abci/0390_launch.py "$@"
