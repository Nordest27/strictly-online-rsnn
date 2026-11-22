#!/bin/zsh
# once all backprop runs have started: the remaining contrast runs, then the two architecture searches
cd /home/user/repos/personal/analysis/drift_sel
until [ $(ls backprop/*/log.txt 2>/dev/null | wc -l) -ge 6 ]; do sleep 30; done
(cd herit && ./queue.sh jobs_contrast_rest.txt) &
sleep 60
(cd hebb_sel && ./queue.sh jobs_contrast.txt) &
sleep 60
search/start.sh fields_l1 --layers 1 --hours 14 --max 50 --refs /home/user/repos/personal/analysis/drift_sel/sep/base/config.json &
search/start.sh nofields_l5 --layers 5 --no-fields --hours 14 --max 50 --seed 1 &
wait
