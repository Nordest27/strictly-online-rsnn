"""Do the one-trace ("leg") and the local-reward forms estimate the same derivatives, and how well?

Open-loop single-neuron replay of the recorded 5-layer network (drift_toy.py; rec_L1 / rec_L5 recordings of the
drift check): the target neuron gets its recorded inputs and fresh noise in every rollout. A fixed external
modulator delta_t (piecewise constant over blocks of 25 steps, N(0, 1), the same in every rollout; open loop, so the
neuron's noise cannot change it and the estimators' gradient parts are absent) stands in for the TD error.

For a local quantity q_t (contrast: the squared deviation of the spike trace from its mean, (z - zbar)^2; drift:
psi - psibar) the objective is J = sum_t delta_t Q_t, Q_t = lambda Q_{t-1} + q_t. Its weight derivative for one
synapse is measured by finite differences (weight +-h, common random numbers). Two estimators from the same
rollouts, summed over the window:
  - leg (one trace of the product, as the drift and contrast_drift): sum_t delta_t T_t,
    T_t = lambda T_{t-1} + q_t pre_t xi_t;
  - reward (the local-reward form, as local.contrast on the lambda-trace route): sum_t (delta_t Q_t - b_t) S_t,
    S_t = lambda S_{t-1} + psibar_t pre_t xi_t, b_t the mean of delta_t Q_t over rollouts (an ideal baseline);
xi the normalised noise trace, pre the presynaptic trace.
Reported per quantity and layer: the correlation across synapses of each estimator's mean with the derivative
(Pearson and Spearman, and with the derivative of the presynaptically weighted objective sum_t delta_t sum_k
lambda^(t-k) pre_k q_k, which the leg form targets by construction), and the per-synapse signal-to-noise of one
window's estimate, |mean| / s.d. over windows x rollouts (the number of windows needed to resolve the sign ~ 1/snr^2).

Usage: python compare.py [--rollouts 16] [--windows 128] [--fd 8] [--neurons 6] [--per 5]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "/home/user/repos/ml_genn/examples/eprop")
import drift_toy as dt  # noqa: E402

REC = "/home/user/repos/personal/analysis/factor_norm_cpu/rec_{L}/outputs/rec_{L}(0)_drift_record.npz"


def lam_trace(x, lam):
    out = np.empty_like(x); acc = np.zeros(x.shape[:1] + x.shape[2:])
    for t in range(x.shape[1]):
        acc = lam * acc + x[:, t]
        out[:, t] = acc
    return out


def quantities(P, alpha, zbar):
    zt = lam_trace(P["z"], alpha)                       # HebbZ: +1 per spike, x alpha per step
    return {"contrast": (zt - zbar) ** 2, "drift": P["psi"] - P["psibar"]}


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main(a):
    rng = np.random.default_rng(a.seed)
    for L in ("L1", "L5"):
        rec = dt.Recording(REC.format(L=L))
        tg = next(g for g in rec.groups if g["name"] == rec.target_group)
        W = np.arange(min(a.windows, rec.W))
        nW = len(W)
        delta = rng.standard_normal((rec.W, (rec.T1 - 1 + 24) // 25)).repeat(25, axis=1)[:, :rec.T1 - 1]
        dW = delta[W]                                                     # (nW, T)
        # target neurons: the most active ones that are not saturated; synapses: the afferents of the target group
        # that deliver most often
        posts = np.unique(tg["post"])
        z_rate = rec.state["RefracTime"][:, 1:, :].astype(bool).mean((0, 1)) if "RefracTime" in rec.state else None
        cand = [j for j in posts if 0.005 < (z_rate[j] if z_rate is not None else 0.05) < 0.3]
        rng.shuffle(cand)
        res = {q: {"leg": [], "reward": [], "fd": [], "fd_pre": [], "snr_leg": [], "snr_reward": [],
                   "leg_w": [], "reward_w": [], "fd_w": []} for q in ("contrast", "drift")}
        for j in cand[:a.neurons]:
            _, _, g0, _, ids = rec.afferents(j)
            deliver, _ = rec.inputs(j)
            own = [i for i, (gname, _) in enumerate(ids) if gname == rec.target_group]
            counts = deliver[W][:, :, own].sum((0, 1))
            ks = [own[i] for i in np.argsort(-counts)[:a.per]]
            h = a.h * float(np.sqrt(np.mean(g0[own] ** 2)))
            acc = {(qn, f, i): [] for qn in ("contrast", "drift") for f in ("leg_w", "reward_w", "fd_w")
                   for i in range(len(ks))}
            zbar = None
            for c0 in range(0, nW, a.chunk):            # windows in chunks (memory: the toy keeps every afferent)
                Wc = W[c0:c0 + a.chunk]; nc = len(Wc); dc = dW[c0:c0 + a.chunk]
                wr = np.repeat(Wc, a.rollouts)
                _, _, _, P = dt.simulate(rec, j, wr, rng.standard_normal((len(wr), rec.T1)), parts_k=ks)
                if zbar is None:
                    zbar = float(lam_trace(P["z"], rec.alpha).mean())
                Q = quantities(P, rec.alpha, zbar)
                d = np.repeat(dc, a.rollouts, axis=0)
                for qn, q in Q.items():
                    r = d * lam_trace(q, rec.lam)
                    b = r.reshape(nc, a.rollouts, -1).mean(1).repeat(a.rollouts, axis=0)
                    for i in range(len(ks)):
                        xi, pre = P["xi"][:, :, i], P["pre"][:, :, i]
                        acc[(qn, "leg_w", i)].append((d * lam_trace(q * pre * xi, rec.lam)).sum(1).reshape(nc, a.rollouts))
                        acc[(qn, "reward_w", i)].append(((r - b) * lam_trace(P["psibar"] * pre * xi, rec.lam)).sum(1)
                                                         .reshape(nc, a.rollouts))
                del P, Q
                wf = np.repeat(Wc, a.fd)
                noise = rng.standard_normal((len(wf), rec.T1))
                df = np.repeat(dc, a.fd, axis=0)
                for i, k in enumerate(ks):
                    J = {}
                    for sgn in (+1, -1):
                        g = np.tile(g0, (len(wf), 1)); g[:, k] += sgn * h
                        _, _, _, Pf = dt.simulate(rec, j, wf, noise, g=g, parts_k=[k])
                        for qn, q in quantities(Pf, rec.alpha, zbar).items():
                            J[(qn, sgn)] = (df * lam_trace(q, rec.lam)).sum(1).reshape(nc, a.fd)
                    for qn in ("contrast", "drift"):
                        acc[(qn, "fd_w", i)].append((J[(qn, 1)] - J[(qn, -1)]) / (2 * h))
            for (qn, f, i), parts in acc.items():
                res[qn][f].append(np.concatenate(parts, 0))
            print(f"  {L} neuron {j}: {len(ks)} synapses done", flush=True)
        print(f"== {L}: {len(res['drift']['fd_w'])} synapses, {nW} windows x {a.rollouts} rollouts, fd {a.fd}")
        def per_window(X):   # (synapses, windows, samples) -> mean per (synapse, window), and the two halves
            return X.mean(2).ravel(), X[:, :, 0::2].mean(2).ravel(), X[:, :, 1::2].mean(2).ravel()

        def sb(a_, b_):      # split-half reliability, Spearman-Brown corrected (0 if the halves do not agree)
            r = np.corrcoef(a_, b_)[0, 1]
            return float(2 * r / (1 + r)) if r > 0 else 0.0
        for qn, r_ in res.items():
            R = {k: np.array(v) for k, v in r_.items()}
            fd, fd0, fd1 = per_window(R["fd_w"]); rel_fd = sb(fd0, fd1)
            line = f"  {qn:8s} per (synapse, window), {fd.size} pairs: derivative reliability {rel_fd:.2f}"
            for est in ("leg", "reward"):
                e, e0, e1 = per_window(R[est + "_w"]); rel = sb(e0, e1)
                c0 = np.corrcoef(e, fd)[0, 1]; ceil = np.sqrt(rel * rel_fd)
                corr = f"{c0 / ceil:+.2f}" if ceil > 0.1 else "n/a"
                line += f" | {est}: reliability {rel:.2f}, corr {c0:+.2f}, ceiling {ceil:.2f}, corrected {corr}"
            print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=int, default=16)
    ap.add_argument("--windows", type=int, default=128)
    ap.add_argument("--fd", type=int, default=8)
    ap.add_argument("--neurons", type=int, default=6)
    ap.add_argument("--per", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=8, help="windows per simulation batch")
    ap.add_argument("--h", type=float, default=0.2, help="finite-difference step, fraction of the RMS weight")
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
