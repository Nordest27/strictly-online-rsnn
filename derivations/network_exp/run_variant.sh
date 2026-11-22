#!/bin/bash
# usage: run_variant.sh NAME EPISODES [ENV=VAL ...]   (later ENV=VAL override earlier ones)
# Uses the patched ml_genn in ./ml_genn_exp via PYTHONPATH (nothing is pip-installed).
# Needs a Python with pygenn (CUDA) + numpy, scipy, opencv-python-headless, matplotlib, line_profiler.
# Set PYTHON to that interpreter if it isn't `python` (e.g. PYTHON=~/venvs/genn/bin/python).
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NAME=$1; EPS=$2; shift 2
PY=${PYTHON:-python}
[ -z "$PYTHON" ] && [ -x "$X/.venv/bin/python" ] && PY=$X/.venv/bin/python
D=$X/runs/$NAME; rm -rf "$D"; mkdir -p "$D/outputs/complete_value_traces"
cp $X/ml_genn_exp/examples/eprop/*.py "$D/"
cd "$D"
env CUDA_PATH=${CUDA_PATH:-/usr/local/cuda} PYTHONPATH="$X/ml_genn_exp/ml_genn:$PYTHONPATH" \
    MAX_EPISODES=$EPS CSV_PREFIX=$NAME REPETITION=0 "$@" "$PY" run_headless.py > "$D/log.txt" 2>&1
echo "$NAME exit $?"
