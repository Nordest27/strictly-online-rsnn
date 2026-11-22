"""Experiment 5: how the neuron-local rewards (voltage stability, voltage regulariser, firing-rate reward)
should enter the proposed rule. One run, five seeds:
  proposed             local rewards x the whole lambda-trace (as implemented)
  proposed_local_qtrace local rewards x the lambda-trace of the gradient part only (gate -psibar);
                       the TD error still uses drift and gradient part (P and Q traces)
  proposed_local_off   local rewards off (the e-prop-style firing-rate gradient term is kept)
Runtime ~20 min.  Usage: python3 exp5_local_rewards.py [env_steps]
"""
import sys, time
import numpy as np
from core import Config, Learner

VARIANTS = ["proposed", "proposed_local_qtrace", "proposed_local_off"]


def main(env_steps=20000):
    L = Learner(Config(), VARIANTS, seeds=[0, 1, 2, 3, 4], rng_seed=0)
    t0 = time.time()
    hist = L.run(env_steps, block=2500)
    print(f"\nreward per environment step, mean ± s.d. over 5 seeds, at 10k and 20k steps  [{time.time() - t0:.0f}s]")
    for i, v in enumerate(VARIANTS):
        sel = L.var_of == i
        print(f"   {v:22s} 10k {hist[3][sel].mean():+.3f} ± {hist[3][sel].std():.3f}   "
              f"20k {hist[-1][sel].mean():+.3f} ± {hist[-1][sel].std():.3f}")
    np.savez("results_exp5.npz", hist=hist, variants=VARIANTS, var_of=L.var_of)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
