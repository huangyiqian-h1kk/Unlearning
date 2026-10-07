#!/bin/bash
set -euo pipefail
project_root="$1"
campaign_root="$2"
worker_id="$3"
cd "$project_root"
: "${PBS_JOBID:?This worker must run inside its own PBS allocation}"
archive_root="$project_root/logs/0390/runs/$PBS_JOBID"
mkdir -p "$archive_root"
exec >>"$archive_root/console.log" 2>&1
trap 'status=$?; printf "{\"exit_code\":%s}\n" "$status" >"$archive_root/launcher-exit.json"' EXIT
source "$project_root/local.env"
: "${CONREP_ENV:?Set CONREP_ENV in local.env}"
: "${CONREP_WORK_ROOT:?Set CONREP_WORK_ROOT in local.env}"
if ! type module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi
module load "${CONREP_CUDA_MODULE:-cuda/12.4/12.4.1}"
source "$CONREP_ENV/bin/activate"
export HF_HOME="${CONREP_HF_CACHE:-$CONREP_WORK_ROOT/cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${CONREP_OMP_THREADS:-8}"
export TORCH_EXTENSIONS_DIR="$CONREP_WORK_ROOT/cache/torch_extensions"
export TRITON_CACHE_DIR="$CONREP_WORK_ROOT/cache/triton"
export TMPDIR="$CONREP_WORK_ROOT/tmp"
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"
python -m pip freeze >"$archive_root/environment.txt"
python "$campaign_root/code/scripts/abci/0390_conrep_night.py" worker \
    --campaign "$campaign_root" --worker "$worker_id"
