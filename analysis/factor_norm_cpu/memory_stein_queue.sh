#!/bin/zsh
cd /home/user/repos/personal/analysis/factor_norm_cpu
PY=/home/user/repos/personal/derivations/network_exp/.venv/bin/python
: > memory_stein_results.txt
for args in "5 standard" "5 resonator 25 1.7" "5 plateau 0.55" "1 standard" "1 resonator 25 2.8" "1 plateau 0.5"; do
  OMP_NUM_THREADS=2 nice -n 5 $PY -I toy_memory_stein.py ${=args} >> memory_stein_results.txt 2>&1
done
echo "STEIN DONE" >> memory_stein_results.txt
