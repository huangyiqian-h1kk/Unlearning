#!/bin/bash
set -euo pipefail
q3_campaign="$1"
q3_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PBS_JOBID:?Run inside the submitted PBS allocation}"
mkdir -p "$q3_campaign/logs/$PBS_JOBID"
exec >>"$q3_campaign/logs/$PBS_JOBID/console.log" 2>&1
trap 'status=$?; printf "{\"exit_code\":%s}\n" "$status" >"$q3_campaign/logs/$PBS_JOBID/launcher-exit.json"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
q3_project="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["project_root"])' "$q3_campaign/plan.json")"
cd "$q3_project"
source local.env
q3_python="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["python"])' "$q3_campaign/plan.json")"
if ! type module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi
module load "${CONREP_CUDA_MODULE:-cuda/12.4/12.4.1}"
export HF_HOME="${CONREP_HF_CACHE:-$CONREP_WORK_ROOT/cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export TORCH_EXTENSIONS_DIR="$CONREP_WORK_ROOT/cache/torch_extensions_qwen3"
export TRITON_CACHE_DIR="$CONREP_WORK_ROOT/cache/triton_qwen3"
export TMPDIR="$CONREP_WORK_ROOT/tmp"
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"
"$q3_python" -m pip freeze > "$q3_campaign/logs/$PBS_JOBID/environment.txt"
"$q3_python" -I "$q3_script_dir/0390_qwen3_rich.py" worker --campaign "$q3_campaign"
