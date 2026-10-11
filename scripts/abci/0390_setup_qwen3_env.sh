#!/bin/bash
set -euo pipefail
# Run from the project root after sourcing local.env. No existing env is changed.
: "${CONREP_ENV:?Source local.env first}"
: "${CONREP_PYTHON:?Set CONREP_PYTHON to Python 3.11}"
: "${CONREP_WORK_ROOT:?Source local.env first}"
: "${QWEN3_ENV:=$(dirname "$CONREP_ENV")/conrep-qwen3-py311-cu124}"
export QWEN3_ENV
q3_source="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
"$CONREP_PYTHON" - "$QWEN3_ENV" "$CONREP_ENV" <<'PY'
import pathlib,sys
assert sys.version_info[:2] == (3,11), 'Python 3.11 required'
assert pathlib.Path(sys.argv[1]).resolve() != pathlib.Path(sys.argv[2]).resolve(), 'Use a separate QWEN3_ENV'
p=pathlib.Path(sys.argv[1])
assert not p.exists() or not any(p.iterdir()) or (p/'0390-qwen3-env').is_file(), 'Existing unrelated environment; choose a new QWEN3_ENV'
p.mkdir(parents=True,exist_ok=True)
(p/'0390-qwen3-env').touch()
PY
mkdir -p "$CONREP_WORK_ROOT/tmp"
export TMPDIR="$CONREP_WORK_ROOT/tmp"
"$CONREP_PYTHON" -m venv "$QWEN3_ENV"
"$QWEN3_ENV/bin/python" -m pip install 'pip<26'
"$QWEN3_ENV/bin/python" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
DS_BUILD_OPS=0 "$QWEN3_ENV/bin/python" -m pip install -r "$q3_source/environments/0390/requirements-qwen3.txt"
"$QWEN3_ENV/bin/python" -m pip check
"$QWEN3_ENV/bin/python" -I "$q3_source/scripts/abci/0390_repair_qwen3_arrow.py" --check
"$QWEN3_ENV/bin/python" -m pip freeze > "$QWEN3_ENV/0390-environment-freeze.txt"
printf 'QWEN3_ENV=%s\n' "$QWEN3_ENV"
