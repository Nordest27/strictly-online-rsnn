"""Decorrelation reward: instantaneous vs lambda-trace credit (open-loop single-neuron replay, as compare.py).

The reward of step t (hidden_rule.py, covariance form): r_t = -(z_t - zbar) (I_t - Ibar) / Ibar, z_t the neuron's
spike in step t, I_t the inhibitory input delivered in step t (the recorded inhibitory afferents' |weight| x spikes;
open loop, so the neuron's noise cannot change it), zbar and Ibar their means. The objective is J = sum_t r_t over
the 200-step window; its weight derivative per (synapse, window) by finite differences (common random numbers).
Two estimators from the same rollouts, each pairing r_t with the noise score of the next synapse stage (it contains
the noise delivered in step t, which the spike of step t depends on):
  - instantaneous: sum_t (r_t - b_t) psibar pre xi                          (route GRADIENT_INSTANT)
  - lambda-trace:  sum_t (r_t - b_t) S,  S = lambda-trace of psibar pre xi  (route GRADIENT_TRACE)
b_t: the mean of r_t over the window's rollouts. Reported as in compare.py: per (synapse, window), the reliability
of each (split halves of the rollouts, Spearman-Brown), its correlation with the derivative, the measurement-noise
ceiling and the corrected correlation.

Usage: python decorr_routes.py [--rollouts 128] [--windows 128] [--fd 64] [--neurons 8] [--per 5] [--chunk 4]
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "/home/user/repos/ml_genn/examples/eprop")
import drift_toy as dt  # noqa: E402
from compare import REC, lam_trace  # noqa: E402


def decorr_reward(z, inh, zbar, ibar):
    return -(z - zbar) * (inh - ibar) / (ibar + 1e-9)


def main(a):
    rng = np.random.default_rng(a.seed)
    for L in ("L1", "L5"):
        rec = dt.Recording(REC.format(L=L))
        tg = next(g for g in rec.groups if g["name"] == rec.target_group)
        inh_groups = {g["name"] for g in rec.groups if g["g"].mean() < 0}
        W = np.arange(min(a.windows, rec.W)); nW = len(W)
        posts = np.unique(tg["post"])
        z_rate = rec.state["RefracTime"][:, 1:, :].astype(bool).mean((0, 1))
        cand = [j for j in posts if 0.005 < z_rate[j] < 0.3]
        rng.shuffle(cand)
        res = {k: [] for k in ("inst", "trace", "fd")}
        for j in cand[:a.neurons]:
            _, _, g0, _, ids = rec.afferents(j)
            deliver, _ = rec.inputs(j)
            inh_idx = [i for i, (gname, _) in enumerate(ids) if gname in inh_groups]
            inh = (deliver[:, 1:, :][:, :, inh_idx] * np.abs(g0[inh_idx])).sum(2)     # I_t for t = 1..T, per window
            ibar = float(inh[W].mean())
            own = [i for i, (gname, _) in enumerate(ids) if gname == rec.target_group]
            counts = deliver[W][:, :, own].sum((0, 1))
            ks = [own[i] for i in np.argsort(-counts)[:a.per]]
            h = a.h * float(np.sqrt(np.mean(g0[own] ** 2)))
            acc = {(f, i): [] for f in res for i in range(len(ks))}
            zbar = None
            for c0 in range(0, nW, a.chunk):
                Wc = W[c0:c0 + a.chunk]; nc = len(Wc)
                wr = np.repeat(Wc, a.rollouts)
                _, _, _, P = dt.simulate(rec, j, wr, rng.standard_normal((len(wr), rec.T1)), parts_k=ks)
                if zbar is None:
                    zbar = float(P["z"].mean())
                I = np.repeat(inh[Wc], a.rollouts, axis=0)
                r = decorr_reward(P["z"], I, zbar, ibar)[:, :-1]          # r_t, paired with the next synapse stage
                b = r.reshape(nc, a.rollouts, -1).mean(1).repeat(a.rollouts, axis=0)
                for i in range(len(ks)):
                    sc = (P["psibar"] * P["pre"][:, :, i] * P["xi"][:, :, i])[:, 1:]
                    acc[("inst", i)].append(((r - b) * sc).sum(1).reshape(nc, a.rollouts))
                    acc[("trace", i)].append(((r - b) * lam_trace(sc, rec.lam)).sum(1).reshape(nc, a.rollouts))
                del P
                wf = np.repeat(Wc, a.fd)
                noise = rng.standard_normal((len(wf), rec.T1))
                If = np.repeat(inh[Wc], a.fd, axis=0)
                for i, k in enumerate(ks):
                    J = {}
                    for sgn in (+1, -1):
                        g = np.tile(g0, (len(wf), 1)); g[:, k] += sgn * h
                        _, _, _, Pf = dt.simulate(rec, j, wf, noise, g=g, parts_k=[k])
                        J[sgn] = decorr_reward(Pf["z"], If, zbar, ibar)[:, :-1].sum(1).reshape(nc, a.fd)
                    acc[("fd", i)].append((J[1] - J[-1]) / (2 * h))
            for (f, i), parts in acc.items():
                res[f].append(np.concatenate(parts, 0))
            print(f"  {L} neuron {j}: {len(ks)} synapses, {len(inh_idx)} inhibitory afferents", flush=True)
        R = {k: np.array(v) for k, v in res.items()}

        def per_window(X):
            return X.mean(2).ravel(), X[:, :, 0::2].mean(2).ravel(), X[:, :, 1::2].mean(2).ravel()

        def sb(x, y):
            r_ = np.corrcoef(x, y)[0, 1]
            return float(2 * r_ / (1 + r_)) if r_ > 0 else 0.0
        fd, fd0, fd1 = per_window(R["fd"]); rel_fd = sb(fd0, fd1)
        line = f"== {L} decorr, {fd.size} (synapse, window) pairs: derivative reliability {rel_fd:.2f}"
        for est in ("inst", "trace"):
            e, e0, e1 = per_window(R[est]); rel = sb(e0, e1)
            c = np.corrcoef(e, fd)[0, 1]; ceil = np.sqrt(rel * rel_fd)
            line += (f" | {est}: reliability {rel:.2f}, corr {c:+.2f}, ceiling {ceil:.2f}, corrected "
                     + (f"{c / ceil:+.2f}" if ceil > 0.1 else "n/a"))
        print(line, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=int, default=128)
    ap.add_argument("--windows", type=int, default=128)
    ap.add_argument("--fd", type=int, default=64)
    ap.add_argument("--neurons", type=int, default=8)
    ap.add_argument("--per", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--h", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=3)
    main(ap.parse_args())
