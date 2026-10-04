#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ -f local.env ]]; then source local.env; fi
: "${CONREP_ENV:?Set CONREP_ENV in local.env}"
: "${CONREP_WORK_ROOT:?Set CONREP_WORK_ROOT under /groups}"
export HF_HOME="${CONREP_HF_CACHE:-$CONREP_WORK_ROOT/cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${CONREP_OMP_THREADS:-8}"
export PYTHONUNBUFFERED=1
mkdir -p "$CONREP_WORK_ROOT/tmp" "$HF_HOME"
export TMPDIR="$CONREP_WORK_ROOT/tmp"
exec "$CONREP_ENV/bin/python" scripts/abci/0390_launch.py "$@"
