"""Experiment 12: making REINFORCE on the neuron-local objectives effective.
Task signal: the proposed rule. Local objectives: voltage regulariser, |dV| (membrane change, corrected)
and firing-rate reward (target 10 Hz), through the gradient part only. One run, five seeds:
  off               no local rewards
  shared            lambda-traced gradient part, shared optimiser normalisation (current)
  sep_0.1 / sep_0.3 own AdaBelief state with 10% / 30% of the learning rate
  sep_0.3_base      + per-neuron running-mean baseline of the local reward
  sep_0.3_base_inst + instantaneous window (noise trace only, no lambda-trace)
Measured in the second half: mean VReg, mean |dV|, mean |F - F*| (hidden neurons), firing rate, reward.
Runtime ~30 min.  Usage: python3 exp12_local_objectives.py [env_steps]
"""
import sys, time
import numpy as np
from core import Config, Learner, Variant

C = dict(c_psi=1.0, c_bar=-2.0, dv_mode="correct")
VARIANTS = [Variant("off", local_rewards="off", **C),
            Variant("shared", local_rewards="q_trace", **C),
            Variant("sep_0.1", local_rewards="q_trace", local_sep=0.1, **C),
            Variant("sep_0.3", local_rewards="q_trace", local_sep=0.3, **C),
            Variant("sep_0.3_base", local_rewards="q_trace", local_sep=0.3, local_baseline=True, **C),
            Variant("sep_0.3_base_inst", local_rewards="gradient", local_sep=0.3, local_baseline=True, **C)]


def main(env_steps=20000):
    L = Learner(Config(), VARIANTS, seeds=[0, 1, 2, 3, 4], rng_seed=0)
    hz = {"v": 0, "n": 0}
    snap = {}

    def hook(l, out, ex):
        if l.progress >= 0.5:
            if "start" not in snap:
                snap["start"] = {k: v.copy() for k, v in l.local_stats.items()}; snap["t0"] = l.t
            hz["v"] = hz["v"] + l.st.z[:, :l.lay.N].mean(1); hz["n"] += 1
    L.hooks.append(hook)
    t0 = time.time()
    hist = L.run(env_steps, block=2500)
    n = hz["n"]
    ls = {k: (L.local_stats[k] - snap["start"][k]) / n for k in L.local_stats}
    print(f"\nsecond half of training, mean over 5 seeds  [{time.time() - t0:.0f}s]")
    print(f"   {'variant':18s} {'VReg':>7s} {'|dV|':>7s} {'|F-F*|(Hz)':>11s} {'hidden Hz':>9s}   reward rate 10k / 20k")
    for i, v in enumerate(VARIANTS):
        sel = L.var_of == i
        print(f"   {v.name:18s} {ls['vreg'][sel].mean():7.4f} {ls['dv'][sel].mean():7.4f} "
              f"{1000 * ls['rate_err'][sel].mean():11.1f} {1000 * hz['v'][sel].mean() / n:9.0f}   "
              f"{hist[3][sel].mean():+.3f} ± {hist[3][sel].std():.3f} / {hist[-1][sel].mean():+.3f} ± {hist[-1][sel].std():.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
