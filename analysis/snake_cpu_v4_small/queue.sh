#!/bin/zsh
while kill -0 3108079 2>/dev/null; do sleep 30; done
for n in mexhat_reservoir inverted_reservoir; do
  (cd /home/user/repos/personal/analysis/snake_cpu_v4_small/$n && env PYTHONPATH=/home/user/repos/ml_genn/ml_genn SNAKE_CONFIG=/home/user/repos/personal/analysis/snake_cpu_v4_small/$n/config.json MPLBACKEND=Agg OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 BUILD_JOBS=2 nohup nice -n 5 /home/user/repos/personal/derivations/network_exp/.venv/bin/python -u /home/user/repos/ml_genn/examples/eprop/run_headless.py > log.txt 2>&1 &)
  echo "started $n $(date)" >> /home/user/repos/personal/analysis/snake_cpu_v4_small/queue.log
done
