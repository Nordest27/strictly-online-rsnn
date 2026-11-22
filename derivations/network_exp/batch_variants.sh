#!/bin/bash
# Original Snake 5x5 architecture: compare rule variants over the early-learning window.
#   EPS       episodes per run (default 2500)
#   SEEDS     runs per variant (default 3)
#   PARALLEL  runs at once (default 1). Each run is a separate process + GeNN build + CUDA context;
#             four at once crashed a 12 GB laptop, so raise only after checking one run's memory use.
#   VARIANTS  subset to run (default "full new drift_only grad_only new_rsign"); see args_for below for all
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); R=$X/run_variant.sh
EPS=${EPS:-2500}; SEEDS=${SEEDS:-3}; PARALLEL=${PARALLEL:-1}
VARIANTS=${VARIANTS:-"full new drift_only grad_only new_rsign"}

args_for() {
    case $1 in
        full)   echo "RULE_VARIANT=full" ;;
        drift)  echo "RULE_VARIANT=drift" ;;
        fluct)  echo "RULE_VARIANT=fluct" ;;
        frozen) echo "RULE_VARIANT=full FREEZE_READOUT=1" ;;
        new)        echo "RULE_VARIANT=new" ;;                                  # psi - 2 psibar (proposed rule)
        drift_only) echo "RULE_VARIANT=drift_only" ;;                           # psi - psibar  (gradient part removed)
        grad_only)  echo "RULE_VARIANT=grad_only" ;;                            # -psibar       (drift removed)
        new_sig03)  echo "RULE_VARIANT=new LOG_SYN_SIG=-1.2" ;;                # noise x0.3
        new_sig3)   echo "RULE_VARIANT=new LOG_SYN_SIG=1.1" ;;                 # noise x3
        full_sig03) echo "RULE_VARIANT=full LOG_SYN_SIG=-1.2" ;;
        full_sig3)  echo "RULE_VARIANT=full LOG_SYN_SIG=1.1" ;;
        new_const)  echo "RULE_VARIANT=new HIDDEN_DELTA=const" ;;              # hidden sees running |delta|: no reward info
        new_rsign)  echo "RULE_VARIANT=new HIDDEN_DELTA=randsign" ;;           # hidden sees delta with random sign
        new_lgrad)  echo "RULE_VARIANT=new LOCAL_REWARDS=gradient" ;;          # local rewards x gradient part only
        new_loff)   echo "RULE_VARIANT=new LOCAL_REWARDS=off" ;;               # local rewards off
        new_lq)     echo "RULE_VARIANT=new LOCAL_REWARDS=q_trace" ;;           # local rewards x lambda-traced gradient part (Q trace)
        new_lq_dv)  echo "RULE_VARIANT=new LOCAL_REWARDS=q_trace DV_MODE=correct" ;;   # + |dV| as membrane change, not the reset jump
        *) echo "unknown variant $1" >&2; exit 1 ;;
    esac
}

for s in $(seq 1 $SEEDS); do
    for v in $VARIANTS; do
        while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 5; done
        $R ${v}_s$s $EPS $(args_for $v) &
    done
done
wait
"${PYTHON:-python}" $X/summarise.py --block=250
