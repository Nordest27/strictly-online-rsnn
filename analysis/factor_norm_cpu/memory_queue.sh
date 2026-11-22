#!/bin/zsh
cd /home/user/repos/personal/analysis/factor_norm_cpu
PY=/home/user/repos/personal/derivations/network_exp/.venv/bin/python
while ps -eo args | grep -v grep | grep -q "toy_resonator.py"; do sleep 30; done
for args in "1 plateau 0.5" "5 plateau 0.55"; do
  OMP_NUM_THREADS=2 nice -n 5 $PY -I toy_memory.py ${=args} 2>&1 | tail -3 >> memory_results.txt
done
echo "PLATEAU DONE" >> memory_results.txt
