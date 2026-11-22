#!/bin/bash
for d in /home/user/repos/personal/analysis/drift_sel/volume/estimators/search_replays/ours_r1_c37_s0 /home/user/repos/personal/analysis/drift_sel/volume/estimators/search_replays/eprop_r1_c09_s0; do
  cd $d && PYTHONPATH=/tmp/claude-1000/-home-user-repos-personal/443153b0-266e-4142-add4-6a0e6900d11b/scratchpad/mlg_wt/ml_genn SNAKE_CONFIG=$d/config.json MPLBACKEND=Agg OMP_NUM_THREADS=1 BUILD_JOBS=2 nice -n 5 /home/user/repos/personal/derivations/network_exp/.venv/bin/python -u /tmp/claude-1000/-home-user-repos-personal/443153b0-266e-4142-add4-6a0e6900d11b/scratchpad/mlg_wt/examples/eprop/run_headless.py > $d/log.txt 2>&1
  /home/user/repos/personal/derivations/network_exp/.venv/bin/python /tmp/claude-1000/-home-user-repos-personal/443153b0-266e-4142-add4-6a0e6900d11b/scratchpad/mlg_wt/examples/eprop/arm_replay.py $d/outputs/arm_episodes -o /mnt/c/Users/72459430/Downloads/arm_search_replays_2026-11-18/$(basename $d).gif --stride 2
done
