#!/bin/zsh
cd /home/user/repos/personal/analysis/drift_sel/volume
V='{"depth": 8, "input_depth": 2, "output_depth": 2, "sigma_cells": 1.0, "p_max": 0.8, "p_global": 0.01}'
for g in 0.5 1.0 2.0 4.0; do
  env PYTHONPATH=/home/user/repos/ml_genn/ml_genn OMP_NUM_THREADS=2 MPLBACKEND=Agg /home/user/repos/personal/derivations/network_exp/.venv/bin/python -u probe_volume.py out v8_g01_ee$g "$V" "{\"internal\": {\"mean_scale\": $g}}" --slots 5 2>&1 | grep -v -E "make|WARN|Init time" > out/v8_g01_ee$g.log &
  sleep 60
  while [ "$(ps -eo args | awk '$1 ~ /python/ && $0 ~ /run_headless.py/' | wc -l)" -ge 5 ]; do sleep 20; done
done
wait
