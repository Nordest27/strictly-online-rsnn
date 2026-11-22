#!/bin/zsh
cd /home/user/repos/personal/analysis/drift_sel/volume
run() {  # name volume layer
  env PYTHONPATH=/home/user/repos/ml_genn/ml_genn OMP_NUM_THREADS=2 MPLBACKEND=Agg /home/user/repos/personal/derivations/network_exp/.venv/bin/python -u probe_volume.py out $1 "$2" "$3" --slots 5 2>&1 | grep -v -E "make|WARN|Init time" > out/$1.log &
  sleep 30
}
B='"depth": 8, "input_depth": 2, "output_depth": 2, "sigma_cells": 1.0, "p_max": 0.8, "p_global": 0.01'




run zs1_ee0.6 "{$B, \"z_shift\": 1.0}" '{"internal": {"mean_scale": 0.6}}'
run zs1.5_ee0.5 "{$B, \"z_shift\": 1.5}" '{"internal": {"mean_scale": 0.5}}'
run zs1.5_ee0.6 "{$B, \"z_shift\": 1.5}" '{"internal": {"mean_scale": 0.6}}'
B='"depth": 8, "input_depth": 2, "output_depth": 2, "sigma_cells": 1.0, "p_max": 0.8, "p_global": 0.03'
run zs1_ee0.5_g03 "{$B, \"z_shift\": 1.0}" '{"internal": {"mean_scale": 0.5}}'
wait
