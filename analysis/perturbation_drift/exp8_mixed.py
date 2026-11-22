"""Experiment 8: does the drift freeze responses in good states and keep them variable in bad states, without
segregated regions?

Same trials as exp7 (a good or bad stimulus, then reward +1 / -1, no critic), but both input groups project
to ALL neurons with random weights (mixed selectivity). For each neuron, compare its responses during good
and during bad stimuli:
  gate:        mean psi during good vs bad stimuli (state-conditional sensitivity)
  variability: Fano factor of its spike count across trials of the same stimulus (frozen vs randomised)
Prediction for rules with a drift: gate and variability higher in bad states than in good states, for the
same neurons; the gradient part alone should do the reverse.
Runtime ~3 min.  Usage: python3 exp8_mixed.py [trials]
"""
import sys
import numpy as np
from core import Config, Layout, NeuronState, advance, score, adabelief, f32

N_IN, N_HID = 20, 40
STIM, TRIAL, REWARD_AT = 20, 40, 25
RATE, W_MEAN, W_SD = 0.2, 0.01, 0.006
GATES = {"drift_only": (1.0, -1.0), "proposed": (1.0, -2.0), "original": (1.0, 0.0),
         "gradient_only": (0.0, -1.0), "no_learning": (0.0, 0.0)}
HOMEO = {}                    # name -> (c, mode): plasticity homeostat c (psibar - target) on a separate drift trace;
                              # mode "fixed" (shared optimiser), "dscaled" (x running mean |delta|), "sep" (own optimiser, 10% lr)
HOMEO_TARGET = 0.1


def main(trials=3000, seeds=5, lr=1e-3, homeo=False):
    if homeo:
        GATES.clear(); GATES["proposed"] = (1.0, -2.0)
        for name, cm in (("fixed_1", (1.0, "fixed")), ("dscaled_5", (5.0, "dscaled")), ("sep_0.1", (1.0, "sep"))):
            GATES[name] = (1.0, -2.0); HOMEO[name] = cm
    cfg = Config(regs=False)
    names = list(GATES); B = len(names) * seeds
    var_of = np.repeat(np.arange(len(names)), seeds)
    lay = Layout(NIN=2 * N_IN, NE=N_HID, NI=0, NF=0)
    M = np.zeros((lay.PRE, lay.POST), f32); M[:2 * N_IN, :] = 1          # both stimuli -> all neurons
    rng = np.random.default_rng(0)
    W = np.stack([(np.random.default_rng(10 + b % seeds).normal(W_MEAN, W_SD, M.shape) * M) for b in range(B)]).astype(f32)
    W0 = W.copy(); m_h, s_h = np.zeros_like(W), np.zeros_like(W)
    c_psi = np.array([GATES[n][0] for n in names], f32)[var_of][:, None]
    c_bar = np.array([GATES[n][1] for n in names], f32)[var_of][:, None]
    learn = np.array([0.0 if n == "no_learning" else lr for n in names], f32)[var_of][:, None, None]
    st = NeuronState(B, lay); p = np.zeros_like(W); t_glob = 0
    hc = np.array([HOMEO.get(n, (0.0, ""))[0] for n in names], f32)[var_of][:, None]
    hds = np.array([HOMEO.get(n, (0.0, ""))[1] == "dscaled" for n in names])[var_of][:, None]
    hsep = np.array([HOMEO.get(n, (0.0, ""))[1] == "sep" for n in names])[var_of][:, None, None]
    absd = np.zeros((B, 1), f32); m_hh, s_hh = np.zeros_like(W), np.zeros_like(W)
    pd = np.zeros_like(W)                                                 # drift trace for the homeostat
    last = trials // 5                                                    # analyse the last 20% of trials
    psi_acc = np.zeros((2, 2, B, N_HID)); n_acc = np.zeros((2, 2, B))     # [phase first/last][good/bad]
    counts = {0: [], 1: []}                                                # spike counts per trial, last phase
    for tr in range(trials):
        good = rng.random(B) < 0.5
        ph = 0 if tr < last else (1 if tr >= trials - last else -1)
        cnt = np.zeros((B, N_HID))
        for k in range(TRIAL):
            zin = np.zeros((B, lay.NIN), f32)
            if k < STIM:
                on = rng.random((B, N_IN)) < RATE
                zin[:, :N_IN] = on * good[:, None]; zin[:, N_IN:] = on * ~good[:, None]
            out = advance(cfg, lay, st, W, M, zin, rng.standard_normal((B, lay.POST)).astype(f32))
            t_glob += 1
            delta = np.zeros((B, 1), f32)
            if k == REWARD_AT:
                delta[:, 0] = np.where(good, 1.0, -1.0)
            absd = 0.999 * absd + 0.001 * np.abs(delta)
            g_h = (hc * np.where(hds, absd, 1.0) * (st.psibar - HOMEO_TARGET))[:, None, :] * pd
            g = delta[:, :, None] * p + np.where(hsep, 0.0, g_h)
            if hsep.any():
                adabelief(W, m_hh, s_hh, np.where(hsep, g_h, 0.0) * M, learn * 0.1, W0, t_glob, cfg, l2=0.0)
            p = cfg.lam * p + score(cfg, c_psi * out["psi"] + c_bar * st.psibar, out["pi"], out["xi"])
            pd = cfg.lam * pd + score(cfg, out["psi"] - st.psibar, out["pi"], out["xi"])
            adabelief(W, m_h, s_h, g * M, learn, W0, t_glob, cfg)
            if k < STIM:
                cnt += st.z
                if ph >= 0:
                    for gi, sel in ((0, good), (1, ~good)):
                        psi_acc[ph, gi][sel] += out["psi"][sel]; n_acc[ph, gi][sel] += 1
        if ph == 1:
            counts[0].append(np.where(good[:, None], cnt, np.nan)); counts[1].append(np.where(~good[:, None], cnt, np.nan))
    psi_m = psi_acc / np.maximum(n_acc[..., None], 1)
    fano = {}
    for gi in (0, 1):
        c = np.array(counts[gi])                                          # trials x B x N
        mu, var = np.nanmean(c, 0), np.nanvar(c, 0)
        fano[gi] = np.where(mu > 0.5, var / np.maximum(mu, 1e-9), np.nan)   # neurons with < 0.5 spikes: undefined
    print(f"{trials} trials, {seeds} seeds, {N_HID} neurons receiving both stimuli\n")
    print(f"   {'rule':14s} {'gate good -> bad (start)':>26s} {'gate good -> bad (end)':>24s} "
          f"{'neurons gate bad > good':>24s} {'Fano good / bad (end)':>22s}")
    for i, n in enumerate(names):
        sel = var_of == i
        g0, b0 = psi_m[0, 0, sel].mean(), psi_m[0, 1, sel].mean()
        g1, b1 = psi_m[1, 0, sel].mean(), psi_m[1, 1, sel].mean()
        frac = (psi_m[1, 1, sel] > psi_m[1, 0, sel]).mean()
        fg, fb = np.nanmean(fano[0][sel]), np.nanmean(fano[1][sel])
        frozen = ((psi_m[1, 0, sel] < 0.005) & (psi_m[1, 1, sel] < 0.005)).mean()
        print(f"   {n:14s} {g0:11.3f} / {b0:.3f}  {g1:11.3f} / {b1:.3f}  {frac:23.2f} {fg:13.2f} / {fb:.2f}"
              f"   frozen in both states: {frozen:.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000, homeo=len(sys.argv) > 2 and sys.argv[2] == "homeo")
