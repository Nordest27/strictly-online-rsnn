"""Experiment 13: centring the TD error for the drift only.
The drift descends sum A e; a critic bias (nonzero mean of delta) acts as a constant modulator that pushes
activity. Here the drift part and the gradient part use separate lambda-traces; the drift part is multiplied
by delta minus its running mean, the gradient part by delta. Local rewards off. One run, five seeds:
  drift_only / drift_only_centered     gate psi - psibar
  proposed   / proposed_centered       gate psi - 2 psibar (drift part psi - psibar, gradient part -psibar)
Runtime ~25 min.  Usage: python3 exp13_drift_centering.py [env_steps]
"""
import sys, time
import numpy as np
from core import Config, Learner, Variant

VARIANTS = [Variant("drift_only", 1.0, -1.0, local_rewards="off"),
            Variant("drift_only_centered", 1.0, -1.0, local_rewards="off", drift_center=True),
            Variant("proposed", 1.0, -2.0, local_rewards="off"),
            Variant("proposed_centered", 1.0, -2.0, local_rewards="off", drift_center=True)]


def main(env_steps=20000):
    L = Learner(Config(), VARIANTS, seeds=[0, 1, 2, 3, 4], rng_seed=0)
    acc = {"hz": 0, "dm": 0, "n": 0}

    def hook(l, out, ex):
        if l.progress >= 0.5:
            acc["hz"] = acc["hz"] + l.st.z[:, :l.lay.N].mean(1); acc["dm"] = acc["dm"] + ex["delta"][:, 0]; acc["n"] += 1
    L.hooks.append(hook)
    t0 = time.time()
    hist = L.run(env_steps, block=2500)
    n = acc["n"]
    print(f"\nsecond half of training, mean over 5 seeds  [{time.time() - t0:.0f}s]")
    print(f"   {'variant':22s} {'hidden Hz':>9s} {'mean delta':>11s}   reward rate 10k / 20k")
    for i, v in enumerate(VARIANTS):
        sel = L.var_of == i
        print(f"   {v.name:22s} {1000 * acc['hz'][sel].mean() / n:9.0f} {acc['dm'][sel].mean() / n:+11.5f}   "
              f"{hist[3][sel].mean():+.3f} ± {hist[3][sel].std():.3f} / {hist[-1][sel].mean():+.3f} ± {hist[-1][sel].std():.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
