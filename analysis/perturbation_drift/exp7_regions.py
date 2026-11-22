"""Experiment 7: does the drift freeze neurons that precede good outcomes and sensitise those that precede bad ones?

Two segregated regions of ALIF neurons (no connections between them). "Good" inputs project only to the
good region, "bad" inputs only to the bad region. Each trial shows one stimulus (50/50) for STIM steps and
then delivers reward +1 (good) or -1 (bad). There is no critic: the TD error is the reward itself, so the
advantage stays positive after good stimuli and negative after bad ones. Only input weights learn, with
each rule's gate, the implementation's sign convention, a lambda-trace and AdaBelief.
Prediction for rules with a drift: the good region freezes (gate closes: silent or saturated) and the bad
region becomes or stays sensitive. Rules without a drift should not split the regions.
Second argument: initial input weight mean (default 0.02, neurons near threshold; 0.006 or 0.004 start
almost quiet, to test whether the drift sensitises the bad region).
Runtime ~2 min.  Usage: python3 exp7_regions.py [trials] [w_mean]
"""
import sys
import numpy as np
from core import Config, Layout, NeuronState, advance, score, adabelief, f32

N_IN, N_REG = 20, 20                     # inputs per stimulus, neurons per region
STIM, TRIAL, REWARD_AT = 20, 40, 25      # stimulus steps, trial length, reward step
RATE, W_MEAN, W_SD = 0.2, 0.02, 0.005
GATES = {"drift_only": (1.0, -1.0), "proposed": (1.0, -2.0), "original": (1.0, 0.0),
         "gradient_only": (0.0, -1.0), "no_learning": (0.0, 0.0)}


def main(trials=3000, seeds=5, lr=1e-3, w_mean=W_MEAN):
    cfg = Config(regs=False)
    names = list(GATES); B = len(names) * seeds
    var_of = np.repeat(np.arange(len(names)), seeds)
    lay = Layout(NIN=2 * N_IN, NE=2 * N_REG, NI=0, NF=0)
    M = np.zeros((lay.PRE, lay.POST), f32)
    M[:N_IN, :N_REG] = 1; M[N_IN:2 * N_IN, N_REG:] = 1                  # good inputs -> good region, bad -> bad
    rng = np.random.default_rng(0)
    W = np.stack([(np.random.default_rng(10 + b % seeds).normal(w_mean, W_SD * w_mean / W_MEAN, M.shape) * M) for b in range(B)]).astype(f32)
    W0 = W.copy(); m_h, s_h = np.zeros_like(W), np.zeros_like(W)
    c_psi = np.array([GATES[n][0] for n in names], f32)[var_of][:, None]
    c_bar = np.array([GATES[n][1] for n in names], f32)[var_of][:, None]
    learn = np.array([0.0 if n == "no_learning" else lr for n in names], f32)[var_of][:, None, None]
    st = NeuronState(B, lay); p = np.zeros_like(W); t_glob = 0
    good_reg, bad_reg = slice(0, N_REG), slice(N_REG, 2 * N_REG)
    blocks = 6; per = trials // blocks
    stats = {k: np.zeros((blocks, B)) for k in ("psi_g", "psi_b", "hz_g", "hz_b", "dist_g", "dist_b")}
    cnt = np.zeros((blocks, B, 2))
    for tr in range(trials):
        good = rng.random(B) < 0.5                                       # each network sees its own sequence
        blk = min(tr // per, blocks - 1)
        for k in range(TRIAL):
            zin = np.zeros((B, lay.NIN), f32)
            if k < STIM:
                on = rng.random((B, N_IN)) < RATE
                zin[:, :N_IN] = on * good[:, None]; zin[:, N_IN:] = on * ~good[:, None]
            nz = rng.standard_normal((B, lay.POST)).astype(f32)
            out = advance(cfg, lay, st, W, M, zin, nz)
            t_glob += 1
            delta = np.zeros((B, 1), f32)
            if k == REWARD_AT:
                delta[:, 0] = np.where(good, 1.0, -1.0)
            g = delta[:, :, None] * p                                     # descend on delta * trace (as implemented)
            gate = c_psi * out["psi"] + c_bar * st.psibar
            p = cfg.lam * p + score(cfg, gate, out["pi"], out["xi"])
            adabelief(W, m_h, s_h, g * M, learn, W0, t_glob, cfg)
            if k < STIM:                                                  # measure each region during its own stimulus
                dist = np.abs(st.v - out["A"])
                for reg, key, sel in ((good_reg, "g", good), (bad_reg, "b", ~good)):
                    stats["psi_" + key][blk] += sel * out["psi"][:, reg].mean(1)
                    stats["hz_" + key][blk] += sel * st.z[:, reg].mean(1)
                    stats["dist_" + key][blk] += sel * dist[:, reg].mean(1)
                cnt[blk, :, 0] += good; cnt[blk, :, 1] += ~good
    for key in ("psi", "hz", "dist"):
        stats[key + "_g"] /= np.maximum(cnt[:, :, 0], 1); stats[key + "_b"] /= np.maximum(cnt[:, :, 1], 1)
    stats["hz_g"] *= 1000; stats["hz_b"] *= 1000
    print(f"initial input weight mean {w_mean}")
    print(f"{trials} trials, {seeds} seeds; per region, during its own stimulus: mean gate psi | firing rate (Hz) | "
          f"mean |V - A|  (first block -> last block)\n")
    print(f"   {'rule':14s} {'good region':>36s}   {'bad region':>36s}")
    for i, n in enumerate(names):
        sel = var_of == i
        f = lambda key: (f"{stats[key][0, sel].mean():.3f}->{stats[key][-1, sel].mean():.3f}" if "psi" in key else
                         f"{stats[key][0, sel].mean():.0f}->{stats[key][-1, sel].mean():.0f}" if "hz" in key else
                         f"{stats[key][0, sel].mean():.2f}->{stats[key][-1, sel].mean():.2f}")
        print(f"   {n:14s} {f('psi_g'):>13s} {f('hz_g'):>10s} {f('dist_g'):>11s}   "
              f"{f('psi_b'):>13s} {f('hz_b'):>10s} {f('dist_b'):>11s}")
    print("\nfinal gate psi, good minus bad region (mean ± s.d. over seeds):")
    for i, n in enumerate(names):
        dlt = stats["psi_g"][-1, var_of == i] - stats["psi_b"][-1, var_of == i]
        print(f"   {n:14s} {dlt.mean():+.3f} ± {dlt.std():.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000, w_mean=float(sys.argv[2]) if len(sys.argv) > 2 else W_MEAN)
