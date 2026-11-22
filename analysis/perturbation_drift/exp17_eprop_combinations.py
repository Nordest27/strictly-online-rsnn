"""Experiment 17: combining random-feedback e-prop with the parts of the perturbation rule.
E-prop (actor-critic, Bellec et al. 2020): hidden learning signal L = Bp (pi - onehot) - Bv with fixed random
feedback, update delta x lambda-trace(ebar L). Noise is injected in all variants. One run, five seeds:
  eprop, eprop+gradient, eprop+drift, eprop+proposed, eprop+homeostat (|delta|-scaled, c = 5),
  eprop+intrinsic (threshold/bias homeostasis, control), proposed (reference). Local rewards off.
Runtime ~60 min.  Usage: python3 exp17_eprop_combinations.py [env_steps]
"""
import sys, time
import numpy as np
from core import Config, Learner, Variant

E = dict(eprop=1.0, local_rewards="off")
VARIANTS = [Variant("eprop", 0.0, 0.0, **E), Variant("eprop+gradient", 0.0, -1.0, **E),
            Variant("eprop+drift", 1.0, -1.0, **E), Variant("eprop+proposed", 1.0, -2.0, **E),
            Variant("eprop+homeostat", 0.0, 0.0, homeo_c=5.0, homeo_dscale=True, **E),
            Variant("eprop+intrinsic", 0.0, 0.0, intrinsic=True, **E),
            Variant("proposed", 1.0, -2.0, local_rewards="off")]


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
    print(f"   {'variant':18s} {'hidden Hz':>9s} {'frozen':>7s}   reward rate 10k / 20k")
    for i, v in enumerate(VARIANTS):
        sel = L.var_of == i
        print(f"   {v.name:18s} {1000 * acc['hz'][sel].mean() / n:9.0f} {acc['dead'][sel].mean() / n:7.2f}   "
              f"{hist[3][sel].mean():+.3f} ± {hist[3][sel].std():.3f} / {hist[-1][sel].mean():+.3f} ± {hist[-1][sel].std():.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
