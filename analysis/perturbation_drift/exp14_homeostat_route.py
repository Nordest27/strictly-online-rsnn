"""Experiment 14: through which term does the plasticity homeostat help?
Forager; drift_only / gradient_only / proposed, each with and without the homeostat (c = 1, target 0.1,
on a separate drift trace), local rewards off. One run, five seeds. Also reports the fraction of hidden
neurons frozen in all states (average gate < 0.005).
Runtime ~40 min.  Usage: python3 exp14_homeostat_route.py [env_steps]
"""
import sys, time
import numpy as np
from core import Config, Learner, Variant

VARIANTS = []
for name, (cp, cb) in (("drift_only", (1.0, -1.0)), ("gradient_only", (0.0, -1.0)), ("proposed", (1.0, -2.0))):
    VARIANTS.append(Variant(name, cp, cb, local_rewards="off"))
    VARIANTS.append(Variant(name + "+homeo", cp, cb, local_rewards="off", homeo_c=1.0))


def main(env_steps=20000):
    L = Learner(Config(), VARIANTS, seeds=[0, 1, 2, 3, 4], rng_seed=0)
    acc = {"hz": 0, "dead": 0, "n": 0}

    def hook(l, out, ex):
        if l.progress >= 0.5:
            acc["hz"] = acc["hz"] + l.st.z[:, :l.lay.N].mean(1)
            acc["dead"] = acc["dead"] + (l.st.psibar[:, :l.lay.N] < 0.005).mean(1); acc["n"] += 1
    L.hooks.append(hook)
    t0 = time.time()
    hist = L.run(env_steps, block=2500)
    n = acc["n"]
    print(f"\nsecond half of training, mean over 5 seeds  [{time.time() - t0:.0f}s]")
    print(f"   {'variant':22s} {'hidden Hz':>9s} {'frozen':>7s}   reward rate 10k / 20k")
    for i, v in enumerate(VARIANTS):
        sel = L.var_of == i
        print(f"   {v.name:22s} {1000 * acc['hz'][sel].mean() / n:9.0f} {acc['dead'][sel].mean() / n:7.2f}   "
              f"{hist[3][sel].mean():+.3f} ± {hist[3][sel].std():.3f} / {hist[-1][sel].mean():+.3f} ± {hist[-1][sel].std():.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
