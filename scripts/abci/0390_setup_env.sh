#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ -f local.env ]]; then source local.env; fi
: "${CONREP_WORK_ROOT:?Set CONREP_WORK_ROOT under /groups}"
: "${CONREP_ENV:?Set CONREP_ENV to a dedicated environment under /groups}"
: "${CONREP_PYTHON:?Set CONREP_PYTHON to a Python 3.11 interpreter}"
mkdir -p "$CONREP_WORK_ROOT/tmp"
export TMPDIR="$CONREP_WORK_ROOT/tmp"
"$CONREP_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,11), "Use Python 3.11 for the ABCI environment"'
"$CONREP_PYTHON" -m venv "$CONREP_ENV"
"$CONREP_ENV/bin/python" -m pip install --upgrade 'pip<26'
"$CONREP_ENV/bin/python" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
DS_BUILD_OPS=0 "$CONREP_ENV/bin/python" -m pip install -r environments/0390/requirements-training.txt -c environments/0390/constraints-py311.txt
"$CONREP_ENV/bin/python" -m pip install --no-deps -e .
"$CONREP_ENV/bin/python" -m pip check
mkdir -p logs/0390
"$CONREP_ENV/bin/python" -m pip freeze > logs/0390/environment-freeze.txt
