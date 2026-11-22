"""Experiment 9: relative weighting of the drift (a) and the perturbation gradient (b),
gate = a (psi - psibar) - b psibar  (proposed rule: a = b = 1). One run, five seeds:
  fixed (a, b) = (1, 1), (1, 2), (2, 1), (1, 0.5)
  drift_first   (a, b) from (1.5, 0.5) to (0.5, 1.5) over training
  grad_first    (a, b) from (0.5, 1.5) to (1.5, 0.5) over training (control)
Runtime ~20 min.  Usage: python3 exp9_weighting.py [env_steps]
"""
import sys, time
import numpy as np
from core import Config, Learner, Variant

VARIANTS = [Variant("a1_b1", 1.0, -2.0), Variant("a1_b2", 1.0, -3.0), Variant("a2_b1", 2.0, -3.0),
            Variant("a1_b0.5", 1.0, -1.5), Variant("drift_first", sched=(1.5, 0.5, 0.5, 1.5)),
            Variant("grad_first", sched=(0.5, 1.5, 1.5, 0.5))]


def main(env_steps=20000):
    L = Learner(Config(), VARIANTS, seeds=[0, 1, 2, 3, 4], rng_seed=0)
    t0 = time.time()
    hist = L.run(env_steps, block=2500)
    print(f"\nreward per environment step, mean ± s.d. over 5 seeds  [{time.time() - t0:.0f}s]")
    for i, v in enumerate(VARIANTS):
        sel = L.var_of == i
        print(f"   {v.name:12s} 10k {hist[3][sel].mean():+.3f} ± {hist[3][sel].std():.3f}   "
              f"20k {hist[-1][sel].mean():+.3f} ± {hist[-1][sel].std():.3f}")
    np.savez("results_exp9.npz", hist=hist, var_of=L.var_of)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
