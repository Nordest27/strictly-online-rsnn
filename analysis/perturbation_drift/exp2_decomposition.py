"""Experiment 2 (Appendix E, Table "decomposition"): train the hidden layer with each part of the score.

One run, five seeds, all variants in the same batch (within-run comparison):
  original          gate psi, i.e. the full score P + Q with the optimiser's sign
  proposed          gate psi - 2 psibar
  twin_P            drift only (antithetic twins)
  twin_Q_code       noise part only, original sign
  twin_Q_reinforce  noise part only, REINFORCE sign
  twin_P_minus_Q    drift + correctly signed noise part (the target of the proposed rule)
  reservoir         hidden weights fixed, readouts learning
It also measures, along training of the proposed rule, how the two parts of its UPDATE (each part of the
score through the lambda-trace, times the TD error) align with those of the original score's update:
cos(noise parts) should be negative (REINFORCE sign) and cos(drift parts) positive.
Runtime ~50 min (single core, five seeds).  Usage: python3 exp2_decomposition.py [env_steps]
"""
import sys, time
import numpy as np
from core import Config, Learner, score

VARIANTS = ["original", "proposed", "twin_P", "twin_Q_code", "twin_Q_reinforce", "twin_P_minus_Q", "reservoir"]


def main(env_steps=20000):
    cfg = Config()
    L = Learner(cfg, VARIANTS, seeds=[0, 1, 2, 3, 4], rng_seed=0)
    sel = L.var_of == VARIANTS.index("proposed")
    acc = {k: 0.0 for k in ("QQ", "qq", "Qq", "PP", "pp", "Pp")}
    tr = {k: 0.0 for k in ("P", "Q", "p", "q")}             # lambda-traces of each part (proposed networks)

    def align(l, out, ex):
        st2, o2 = ex["twin_state"], ex["twin_out"]
        g1 = (out["psi"] - 2 * l.st.psibar)[sel]; g2 = (o2["psi"] - 2 * st2.psibar)[sel]
        s1 = score(cfg, g1, out["pi"][sel], out["xi"][sel]); s2 = score(cfg, g2, o2["pi"][sel], o2["xi"][sel])
        p, q = 0.5 * (s1 + s2), 0.5 * (s1 - s2)
        P, Q = ex["P"][sel], ex["Q"][sel]
        d = ex["delta"][sel][:, :, None]
        u = {k: d * tr[k] for k in tr}                       # update contributions (trace before update)
        acc["QQ"] += (u["Q"] ** 2).sum(); acc["qq"] += (u["q"] ** 2).sum(); acc["Qq"] += (u["Q"] * u["q"]).sum()
        acc["PP"] += (u["P"] ** 2).sum(); acc["pp"] += (u["p"] ** 2).sum(); acc["Pp"] += (u["P"] * u["p"]).sum()
        for k, x in (("P", P), ("Q", Q), ("p", p), ("q", q)):
            tr[k] = cfg.lam * tr[k] + x
    L.hooks.append(align)
    t0 = time.time()
    hist = L.run(env_steps, block=2500)
    print(f"\nreward per environment step after {env_steps} steps (mean ± s.d. over 5 seeds; random policy ≈ -0.13)"
          f"  [{time.time() - t0:.0f}s]")
    for name, (m, s) in L.summary(hist).items():
        print(f"   {name:18s} {m:+.3f} ± {s:.3f}")
    print("\nupdate of the proposed rule vs update of the original score, accumulated over training:")
    print(f"   cos(noise parts) = {acc['Qq'] / np.sqrt(acc['QQ'] * acc['qq']):+.2f}   "
          f"|noise part| ratio = {np.sqrt(acc['qq'] / acc['QQ']):.2f}")
    print(f"   cos(drift parts) = {acc['Pp'] / np.sqrt(acc['PP'] * acc['pp']):+.2f}")
    print(f"   |noise part| / |drift| in the original update = {np.sqrt(acc['QQ'] / acc['PP']):.1f}")
    np.savez("results_exp2.npz", hist=hist, variants=VARIANTS, var_of=L.var_of)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
