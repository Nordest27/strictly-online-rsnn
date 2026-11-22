"""Experiment 6: does the drift allocate sensitivity to moments before worse-than-expected outcomes?

The drift minimises sum_t A_t e_t (A_t = sum_k lambda^k delta_{t+k}): it should make neurons more sensitive
(gate open) before worse-than-expected outcomes and less sensitive before better-than-expected ones.
Measured per network along training, in one run with five seeds:
  1. correlation over time between sensitivity s_t (mean gate psi of core and field neurons) and A_t;
  2. sensitivity in the 3 environment steps before deaths, split by surprise (delta summed over the death
     step: below / above the median), and before food, relative to the network's mean sensitivity.
Controls: reservoir (hidden weights fixed: correlation from dynamics and inputs alone) and gradient_only
(no drift), against drift_only and proposed.
Runtime ~20 min.  Usage: python3 exp6_plasticity_allocation.py [env_steps]
"""
import sys, time
import numpy as np
from core import Config, Learner

VARIANTS = ["proposed", "drift_only", "gradient_only", "reservoir"]
BLOCK = 5000          # environment steps per analysis block
PRE = 3               # environment steps before an event


def main(env_steps=20000):
    cfg = Config()
    L = Learner(cfg, VARIANTS, seeds=[0, 1, 2, 3, 4], rng_seed=0)
    n_t = env_steps * cfg.K
    s = np.zeros((n_t, L.B), np.float32); d = np.zeros((n_t, L.B), np.float32)
    events = []                                                     # (sim step, batch idx, +1 food / -1 death)

    def hook(l, out, ex):
        s[l.t - 1] = out["psi"].mean(1); d[l.t - 1] = ex["delta"][:, 0]
    L.hooks.append(hook)
    act0 = L.act

    def act():
        rew, ate, dead = act0()
        for b in np.nonzero(ate | dead)[0]:
            events.append((L.t - 1, b, 1 if ate[b] else -1))
        return rew, ate, dead
    L.act = act
    t0 = time.time()
    hist = L.run(env_steps, block=BLOCK)
    print(f"[{time.time() - t0:.0f}s]")
    A = np.zeros_like(d)
    for t in range(n_t - 2, -1, -1):                                 # A_t = delta_t + lambda * A_{t+1}
        A[t] = d[t] + cfg.lam * A[t + 1]
    ev = np.array(events)
    nb = env_steps // BLOCK
    print(f"\n1. corr(sensitivity s_t, following advantage A_t) per block of {BLOCK} env steps "
          f"(mean ± s.d. over seeds)")
    print(f"   {'variant':16s} " + "  ".join(f"{(k + 1) * BLOCK // 1000:>6d}k" for k in range(nb)))
    for i, v in enumerate(VARIANTS):
        cols = []
        for k in range(nb):
            sl = slice(k * BLOCK * cfg.K, (k + 1) * BLOCK * cfg.K)
            c = [np.corrcoef(s[sl, b], A[sl, b])[0, 1] for b in np.nonzero(L.var_of == i)[0]]
            cols.append(f"{np.mean(c):+.3f}±{np.std(c):.3f}")
        print(f"   {v:16s} " + "  ".join(cols))
    print(f"\n2. sensitivity in the {PRE} env steps before an event, relative to the network's mean (second half "
          f"of training)")
    print(f"   {'variant':16s} {'mispredicted deaths':>20s} {'predicted deaths':>17s} {'food':>8s}")
    half = n_t // 2
    for i, v in enumerate(VARIANTS):
        rows = {"mis": [], "pred": [], "food": []}
        for b in np.nonzero(L.var_of == i)[0]:
            e = ev[(ev[:, 1] == b) & (ev[:, 0] >= half + PRE * cfg.K) & (ev[:, 0] < n_t - cfg.K)]
            base = s[half:, b].mean()
            sens = np.array([s[t - PRE * cfg.K:t, b].mean() / base for t in e[:, 0]])
            surprise = np.array([d[t:t + cfg.K, b].sum() for t in e[:, 0]])
            dth = e[:, 2] < 0
            if dth.sum() >= 4:
                med = np.median(surprise[dth])
                rows["mis"].append(sens[dth & (surprise <= med)].mean())
                rows["pred"].append(sens[dth & (surprise > med)].mean())
            if (~dth).sum() >= 4:
                rows["food"].append(sens[~dth].mean())
        f = lambda x: f"{np.mean(x):.3f}±{np.std(x):.3f}" if x else "--"
        print(f"   {v:16s} {f(rows['mis']):>20s} {f(rows['pred']):>17s} {f(rows['food']):>8s}")
    print("\nreward rate at the end:", {v: round(float(hist[-1][L.var_of == i].mean()), 3) for i, v in enumerate(VARIANTS)})


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
