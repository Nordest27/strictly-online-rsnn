#!/bin/zsh
# start the afterglow runs once the sweep's queue has finished all 6 runs
until [ "$(wc -l < progress.txt 2>/dev/null || echo 0)" -ge 6 ]; do sleep 60; done
xargs -P 4 -I{} ./run_one.sh {} < jobs2.txt
