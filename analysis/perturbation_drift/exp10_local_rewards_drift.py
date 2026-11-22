"""Experiment 10: how each neuron-local reward acts when routed through the drift vs the gradient part.

Task signal: the proposed rule (gate psi - 2 psibar, TD error). Each local reward alone, scaled x10 so that
its direction of action is visible in the small task, routed through the lambda-trace of the drift part
(gate psi - psibar, the implementation's sign) or of the gradient part (gate -psibar), plus local rewards off.
Local rewards: spike penalty (the |dV| term as implemented: -0.01 v_th z), voltage regulariser
(-c_reg VReg), firing-rate reward (c_reg (trace of |F - F*| - |F - F*|), target 10 Hz).
Measured: hidden / field firing rate, mean gate of the hidden neurons, reward rate. One run, three seeds.
Runtime ~20 min.  Usage: python3 exp10_local_rewards_drift.py [env_steps] [scale]
"""
import sys, time
import numpy as np
from core import Config, Learner, Variant, f32


def variants(scale):
    v = [Variant("off", 1.0, -2.0, local_rewards="off")]
    for name, k in (("spike", 0), ("vreg", 1), ("rate", 2)):
        terms = tuple(scale if j == k else 0.0 for j in range(3))
        v.append(Variant(f"{name}_drift", 1.0, -2.0, local_rewards="p_trace", local_terms=terms))
        v.append(Variant(f"{name}_grad", 1.0, -2.0, local_rewards="q_trace", local_terms=terms))
    return v


def main(env_steps=20000, scale=10.0):
    V = variants(scale)
    L = Learner(Config(), V, seeds=[0, 1, 2], rng_seed=0)
    acc = {"hz_h": 0, "hz_f": 0, "psi": 0, "n": 0}

    def hook(l, out, ex):
        if l.progress >= 0.5:                                   # second half of training
            acc["hz_h"] = acc["hz_h"] + l.st.z[:, :l.lay.N].mean(1)
            acc["hz_f"] = acc["hz_f"] + l.st.z[:, l.lay.N:].mean(1)
            acc["psi"] = acc["psi"] + out["psi"][:, :l.lay.N].mean(1); acc["n"] += 1
    L.hooks.append(hook)
    t0 = time.time()
    hist = L.run(env_steps, block=2500)
    n = acc["n"]
    print(f"\nlocal rewards x{scale}, second half of training, mean over 3 seeds  [{time.time() - t0:.0f}s]")
    print(f"   {'variant':14s} {'hidden Hz':>9s} {'field Hz':>9s} {'gate psi':>9s} {'reward rate (20k)':>18s}")
    for i, v in enumerate(V):
        sel = L.var_of == i
        print(f"   {v.name:14s} {1000 * acc['hz_h'][sel].mean() / n:9.0f} {1000 * acc['hz_f'][sel].mean() / n:9.0f} "
              f"{acc['psi'][sel].mean() / n:9.3f} {hist[-1][sel].mean():+11.3f} ± {hist[-1][sel].std():.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000, float(sys.argv[2]) if len(sys.argv) > 2 else 10.0)
