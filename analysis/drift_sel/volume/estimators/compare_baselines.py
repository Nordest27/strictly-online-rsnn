"""Baselines for the reward version of the drift (local.sensitivity), in the open-loop replay of a recording.

Per-step reward (TD error = 1): r_t = -SensD_{t-1}; credited through Gtrace (lambda-trace of -psibar * ZF * xi), Adam
momentum at the end of the window. Advantage variants:
  none     : r_t
  constant : r_t - mean(r)                                        (a long running mean, the rule's SensMean/LocMean)
  ncritic  : d_t = r_t + gamma V_t - V_{t-1}, the code's neuron critic (critic_level "neuron"): V = w . phi, phi =
             [1, 0.99-filtered (V - threshold), 0.99-filtered A, 0.99-filtered spikes (for HebbMean)]; the
             inhibitory-mean feature is left out. w = its TD(lambda) fixed point (LSTD over all trajectories), i.e.
             the critic at its best; online learning would add noise.
  input    : r_t - b_t, b_t linear in the neuron's presynaptic traces (least squares): the neuron's own noise cannot
             move it, so it is unbiased by construction
  oracle   : r_t - (mean of r_t over the other rollouts of the same window), an upper bound for input baselines
Per synapse: the estimator's mean over all trajectories (bias: compared with the unbiased "none" mean and with the
finite-difference truth T_S) and its per-trajectory SNR (|mean| / sd); the drift's SNR for reference.

Usage: python compare_baselines.py RECORD.npz [--slabs 0] [--neurons 8] [--rollouts 32] [--fd 8] [--depth 8]
"""
import sys, json, argparse
import numpy as np
sys.path.insert(0, "/home/user/repos/ml_genn/examples/eprop")
import drift_toy as dt
from compare_estimators import momentum, ltrace, corr

GAMMA, LAM_C, FD = 0.977, 0.977, 0.99


def filt(x, d):
    out = np.empty_like(x); f = x[:, 0].copy()
    for t in range(x.shape[1]):
        f = d * f + (1 - d) * x[:, t]; out[:, t] = f
    return out


def neuron_terms(rec, j, wr, noise, ks, g=None):
    s, e, v, parts = dt.simulate(rec, j, wr, noise, g=g, parts_k=ks)
    pb, dev, z = parts["psibar"], parts["psi"] - parts["psibar"], parts["z"]
    sensd = ltrace(dev, rec.lam)
    r = -np.concatenate([np.zeros((len(dev), 1)), sensd[:, :-1]], 1)            # (R, T)
    gtr = ltrace(np.moveaxis(-pb[:, :, None] * parts["pre"] * parts["xi"], 1, -1), rec.lam)   # (R, k, T)
    beta = rec.state["Beta"][wr, 0, j]; A = rec.state["A"][wr, 0, j].astype(np.float64)
    As = np.empty_like(z)
    for t in range(z.shape[1]):
        A = rec.rho * A + z[:, t]; As[:, t] = A
    thr = rec.vth + beta[:, None] * As
    phi = np.stack([np.ones_like(v), filt(v - thr, FD), filt(As, FD), filt(z, FD)], -1)     # (R, T, F)
    drift = momentum(ltrace(np.moveaxis(s[:, :, ks], 1, -1), rec.lam), rec.beta1)            # (R, k)
    return r, gtr, phi, drift, sensd


def lstd(phi, r):
    """TD(lambda) fixed point of V = phi . w for reward r (the critic's targets): sum e (phi_{t-1} - g phi_t) w = sum e r."""
    R, T, F = phi.shape
    Am, bv, e = np.zeros((F, F)), np.zeros(F), np.zeros((R, F))
    for t in range(1, T):
        e = LAM_C * e + phi[:, t - 1]
        Am += e.T @ (phi[:, t - 1] - GAMMA * phi[:, t]); bv += e.T @ r[:, t]
    return np.linalg.solve(Am + 1e-6 * np.eye(F), bv)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("record")
    ap.add_argument("--slabs", default="0"); ap.add_argument("--neurons", type=int, default=8)
    ap.add_argument("--rollouts", type=int, default=32); ap.add_argument("--fd", type=int, default=8)
    ap.add_argument("--depth", type=int, default=8); ap.add_argument("--max-aff", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=None); a = ap.parse_args()
    rec = dt.Recording(a.record); rng = np.random.default_rng(a.seed)
    n_tgt = rec.state["V"].shape[2]; z_of = np.arange(n_tgt) % a.depth
    slabs = [int(x) for x in a.slabs.split(",")]
    cand = np.flatnonzero(np.isin(z_of, slabs))
    act = rec.state["Psi"][:, :, cand].mean((0, 1)); cand = cand[act > np.percentile(act, 25)]
    js = rng.choice(cand, min(a.neurons, len(cand)), replace=False)
    rms = np.sqrt(np.mean(np.concatenate([gr["g"] for gr in rec.groups if gr["name"] == rec.target_group]) ** 2))
    h = 0.2 * rms
    names = ("none", "constant", "ncritic", "input", "oracle")
    M = {k: [] for k in names}; S = {k: [] for k in names}; TS, snr_drift, r2 = [], [], {"ncritic": [], "input": []}
    for j in js:
        j = int(j)
        src, pre, g0, sig, ids = rec.afferents(j)
        deliver, trace = rec.inputs(j)
        rate = deliver.mean((0, 1))
        ks = [int(k) for k in np.argsort(-rate)[:a.max_aff] if rate[k] > 0 and sig[k] > 0]
        if not ks:
            continue
        W = rec.W; wr = np.repeat(np.arange(W), a.rollouts)
        r, gtr, phi, drift, _ = neuron_terms(rec, j, wr, rng.standard_normal((len(wr), rec.T1)), ks)
        T = r.shape[1]
        # advantages
        adv = {"none": r, "constant": r - r.mean()}
        w = lstd(phi, r)
        V = phi @ w
        adv["ncritic"] = r + GAMMA * V - np.concatenate([V[:, :1], V[:, :-1]], 1)
        X = np.concatenate([np.ones((len(wr), T, 1)), trace[wr, :T]], -1)               # presynaptic traces, all afferents
        Xf, rf = X.reshape(-1, X.shape[-1]), r.reshape(-1)
        beta = np.linalg.lstsq(Xf.T @ Xf + 1e-3 * np.eye(Xf.shape[1]), Xf.T @ rf, rcond=None)[0]
        b_in = X @ beta
        adv["input"] = r - b_in
        rr = r.reshape(W, a.rollouts, T)
        loo = (rr.sum(1, keepdims=True) - rr) / (a.rollouts - 1)
        adv["oracle"] = (rr - loo).reshape(-1, T)
        var_r = r.var()
        r2["ncritic"].append(1 - np.var(adv["ncritic"]) / var_r); r2["input"].append(1 - np.var(r - b_in) / var_r)
        for k in names:
            est = momentum(adv[k][:, None, :] * gtr, rec.beta1)                       # (R, n_k)
            M[k].append(est.mean(0)); S[k].append(np.abs(est.mean(0)) / (est.std(0) + 1e-30))
        snr_drift.append(np.abs(drift.mean(0)) / (drift.std(0) + 1e-30))
        # truth: d/dW of the momentum average of E[SensD]
        wf = np.repeat(np.arange(W), a.fd); nz = rng.standard_normal((len(wf), rec.T1)); t_k = []
        for k in ks:
            o = []
            for sgn in (1, -1):
                g = np.tile(g0, (len(wf), 1)); g[:, k] += sgn * h
                *_, sd = neuron_terms(rec, j, wf, nz, [k], g=g)
                o.append(momentum(sd.mean(0)[None], rec.beta1)[0])
            t_k.append((o[0] - o[1]) / (2 * h))
        TS.append(np.array(t_k))
        print(f"  neuron {j}: {len(ks)} synapses, critic R2 {r2['ncritic'][-1]:.2f}, input R2 {r2['input'][-1]:.2f}",
              flush=True)
    TS = np.concatenate(TS); base = np.concatenate(M["none"])
    res = {"slabs": slabs, "n_syn": int(len(TS)), "drift SNR (median)": float(np.median(np.concatenate(snr_drift))),
           "critic variance reduction (median)": float(np.median(r2["ncritic"])), "input variance reduction (median)": float(np.median(r2["input"]))}
    for k in names:
        m = np.concatenate(M[k])
        slope = float(m @ TS / (TS @ TS + 1e-30))
        res[k] = {"SNR (median)": float(np.median(np.concatenate(S[k]))), "corr with truth": corr(m, TS),
                  "slope vs truth": slope}
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
