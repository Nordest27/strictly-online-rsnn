"""Experiment 11: local rewards designed for each component (proposed rule as the task signal, one run, 5 seeds).
  local_off       no neuron-local rewards (the e-prop-style firing-rate gradient term is kept)
  local_q         local rewards through the lambda-trace of the gradient part only (|dV| as implemented)
  local_q_dv      same, with |dV| measured from the previous post-reset voltage (membrane change, not reset)
  homeo_c         plasticity homeostat through the drift: modulator c (psibar - 0.1) on the drift trace,
                  local rewards off; c = 0.01, 0.1, 1
Runtime ~25 min.  Usage: python3 exp11_homeostat.py [env_steps]
"""
import sys, time
import numpy as np
from core import Config, Learner, Variant

VARIANTS = [Variant("local_off", 1.0, -2.0, local_rewards="off"),
            Variant("local_q", 1.0, -2.0, local_rewards="q_trace"),
            Variant("local_q_dv", 1.0, -2.0, local_rewards="q_trace", dv_mode="correct"),
            Variant("homeo_0.01", 1.0, -2.0, local_rewards="off", homeo_c=0.01),
            Variant("homeo_0.1", 1.0, -2.0, local_rewards="off", homeo_c=0.1),
            Variant("homeo_1", 1.0, -2.0, local_rewards="off", homeo_c=1.0)]


def main(env_steps=20000):
    L = Learner(Config(), VARIANTS, seeds=[0, 1, 2, 3, 4], rng_seed=0)
    acc = {"hz": 0, "psi": 0, "dead": 0, "n": 0}

    def hook(l, out, ex):
        if l.progress >= 0.5:
            acc["hz"] = acc["hz"] + l.st.z[:, :l.lay.N].mean(1)
            acc["psi"] = acc["psi"] + out["psi"][:, :l.lay.N].mean(1)
            acc["dead"] = acc["dead"] + (l.st.psibar[:, :l.lay.N] < 0.005).mean(1); acc["n"] += 1
    L.hooks.append(hook)
    t0 = time.time()
    hist = L.run(env_steps, block=2500)
    n = acc["n"]
    print(f"\nsecond half of training, mean over 5 seeds  [{time.time() - t0:.0f}s]")
    print(f"   {'variant':12s} {'hidden Hz':>9s} {'gate':>6s} {'frozen neurons':>15s}   reward rate 10k / 20k")
    for i, v in enumerate(VARIANTS):
        sel = L.var_of == i
        print(f"   {v.name:12s} {1000 * acc['hz'][sel].mean() / n:9.0f} {acc['psi'][sel].mean() / n:6.3f} "
              f"{acc['dead'][sel].mean() / n:15.2f}   {hist[3][sel].mean():+.3f} ± {hist[3][sel].std():.3f} / "
              f"{hist[-1][sel].mean():+.3f} ± {hist[-1][sel].std():.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
