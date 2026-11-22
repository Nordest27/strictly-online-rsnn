"""Experiment 15: how to weight the plasticity homeostat (forager, local rewards off, one run, five seeds).
  none            proposed rule without homeostat
  fixed           modulator c (psibar - 0.1), c = 1, shared optimiser normalisation
  dscaled         c * mean|delta| * (psibar - 0.1), c = 5 (mean|delta| ~ 0.19 here, so ~ fixed c = 1)
  sep_0.1/0.3     c = 1 with its own AdaBelief state and 10% / 30% of the learning rate
and the same fixed / dscaled / sep_0.1 homeostats with the gradient-only rule.
Runtime ~60 min.  Usage: python3 exp15_homeostat_weighting.py [env_steps]
"""
import sys, time
import numpy as np
from core import Config, Learner, Variant

P, G = dict(c_psi=1.0, c_bar=-2.0, local_rewards="off"), dict(c_psi=0.0, c_bar=-1.0, local_rewards="off")
VARIANTS = [Variant("none", **P), Variant("fixed", homeo_c=1.0, **P), Variant("dscaled", homeo_c=5.0, homeo_dscale=True, **P),
            Variant("sep_0.1", homeo_c=1.0, homeo_sep=0.1, **P), Variant("sep_0.3", homeo_c=1.0, homeo_sep=0.3, **P),
            Variant("grad_fixed", homeo_c=1.0, **G), Variant("grad_dscaled", homeo_c=5.0, homeo_dscale=True, **G),
            Variant("grad_sep_0.1", homeo_c=1.0, homeo_sep=0.1, **G)]


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
    print(f"   {'variant':14s} {'hidden Hz':>9s} {'frozen':>7s}   reward rate 10k / 20k")
    for i, v in enumerate(VARIANTS):
        sel = L.var_of == i
        print(f"   {v.name:14s} {1000 * acc['hz'][sel].mean() / n:9.0f} {acc['dead'][sel].mean() / n:7.2f}   "
              f"{hist[3][sel].mean():+.3f} ± {hist[3][sel].std():.3f} / {hist[-1][sel].mean():+.3f} ± {hist[-1][sel].std():.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
