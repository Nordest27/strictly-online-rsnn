"""Nonlinear critics of the truly local (TD-free) rewards from state connections, in the open-loop replay.

Same rewards and credit as local_rewards.py (vreg, rate, decorr; shared route GRADIENT_INSTANT: momentum average of
(r - b) * g_score). The critic sees presynaptic traces of: the neuron's afferents, its K nearest neighbours (own
population, 3D) and K random neurons of the whole network ("global"). Readouts (cross-fitted between even and odd
windows; inputs are identical across a window's rollouts, so they are fitted to the rollout mean):
  linear afferent / linear combined
  random features H: ReLU(standardised traces @ R + c), R ~ N(0, 1/F), c ~ U(-1, 1), then a ridge readout
     (a stand-in for what a learned nonlinear dendrite could represent)
  oracle: mean of r over the other rollouts of the same window and step
Reported: the estimator's variance vs no baseline (lower = fewer samples), reward variance explained, bias (SE units).

Usage: python nonlinear_critic.py RECDIR PREFIX [--slabs 0] [--neurons 6] [--rollouts 16] [--K 32] [--H 256 1024]
"""
import sys, json, argparse
import numpy as np
sys.path.insert(0, "/home/user/repos/ml_genn/examples/eprop")
import drift_toy as dt
from compare_estimators import momentum
from local_rewards import terms


def ridge(X, y, lam=1e-3):
    A = X.T @ X
    return np.linalg.solve(A + lam * np.trace(A) / len(A) * np.eye(len(A)) + 1e-9 * np.eye(len(A)), X.T @ y)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("recdir"); ap.add_argument("prefix")
    ap.add_argument("--slabs", default="0"); ap.add_argument("--neurons", type=int, default=6)
    ap.add_argument("--rollouts", type=int, default=16); ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--H", type=int, nargs="+", default=[256, 1024]); ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--max-aff", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None); a = ap.parse_args()
    out = f"{a.recdir}/outputs/{a.prefix}(0)"
    times = dict(tuple(map(int, l.split())) for l in open(f"{out}_drift_record_times.txt"))
    recs = [dt.Recording(f"{out}_drift_record_{k}.npz") for k in sorted(times)]
    rng = np.random.default_rng(a.seed)
    r0 = recs[0]; n_tgt = r0.state["V"].shape[2]; z_of = np.arange(n_tgt) % a.depth
    slabs = [int(x) for x in a.slabs.split(",")]
    cand = np.flatnonzero(np.isin(z_of, slabs))
    act = np.mean([rc.state["Psi"][:, :, cand].mean((0, 1)) for rc in recs], 0)
    cand = cand[act > np.percentile(act, 25)]
    js = rng.choice(cand, min(a.neurons, len(cand)), replace=False)
    sizes = [z.shape[2] for z in r0.spikes]
    tsrc = next(gr["source"] for gr in r0.groups if gr["name"] == r0.target_group)
    side = int(round(np.sqrt(n_tgt / a.depth)))
    idx = np.arange(n_tgt); px, py, pz = idx // (side * a.depth), (idx // a.depth) % side, idx % a.depth
    variants = ["none", "constant", "linear afferent", "linear combined"] + [f"nonlinear H{h}" for h in a.H] + ["oracle"]
    acc = {tn: {"VR": {v: [] for v in variants}, "BIAS": {v: [] for v in variants}, "R2": {v: [] for v in variants}}
           for tn in ("vreg", "rate", "decorr")}
    W, Rr = r0.W, a.rollouts
    for j in js:
        j = int(j)
        src, pre, g0, sig, ids = r0.afferents(j)
        rate_in = np.mean([rc.inputs(j)[0].mean((0, 1)) for rc in recs], 0)
        ks = [int(k) for k in np.argsort(-rate_in)[:a.max_aff] if rate_in[k] > 0 and sig[k] > 0]
        inh_k = np.flatnonzero(g0 < 0)
        if not ks:
            continue
        dx = np.abs(px - px[j]); dx = np.minimum(dx, side - dx); dy = np.abs(py - py[j]); dy = np.minimum(dy, side - dy)
        d2 = dx ** 2 + dy ** 2 + (pz - pz[j]) ** 2 + rng.random(n_tgt) * 1e-3; d2[j] = np.inf
        nbr = [(tsrc, int(n)) for n in np.argsort(d2)[:a.K]]
        pool = [(si, n) for si, sz in enumerate(sizes) for n in range(sz) if not (si == tsrc and n == j)]
        glob = [pool[i] for i in rng.choice(len(pool), a.K, replace=False)]
        RS = {tn: [] for tn in acc}; G, XA, XC, WID = [], [], [], []
        for rc in recs:
            wr = np.repeat(np.arange(W), Rr)
            rr, gsc = terms(rc, j, wr, rng.standard_normal((len(wr), rc.T1)), ks, inh_k)
            T = gsc.shape[-1]
            for tn in acc:
                RS[tn].append(rr[tn])
            _, trace = rc.inputs(j)
            xa = trace[:, :T]
            xn = np.stack([rc.zf[si][:, :T, n] for si, n in nbr], -1)
            xg = np.stack([rc.zf[si][:, :T, n] for si, n in glob], -1)
            G.append(gsc); XA.append(xa); XC.append(np.concatenate([xa, xn, xg], -1)); WID.append(np.arange(W))
        gsc = np.concatenate(G)                                     # (S*W*Rr, k, T)
        Xa, Xc, wid = np.concatenate(XA), np.concatenate(XC), np.concatenate(WID)   # per (state, window)
        NW, T = Xa.shape[:2]
        fold_w = (wid % 2 == 0)
        mu, sd = Xc.reshape(-1, Xc.shape[-1]).mean(0), Xc.reshape(-1, Xc.shape[-1]).std(0) + 1e-6
        Xs = ((Xc - mu) / sd).astype(np.float32)
        feats = {"linear afferent": np.concatenate([np.ones((NW, T, 1)), Xa], -1),
                 "linear combined": np.concatenate([np.ones((NW, T, 1)), Xc], -1)}
        F = Xs.shape[-1]
        for h in a.H:
            R = (rng.standard_normal((F, h)) / np.sqrt(F)).astype(np.float32)
            c = rng.uniform(-1, 1, h).astype(np.float32)
            feats[f"nonlinear H{h}"] = np.concatenate([np.ones((NW, T, 1), np.float32),
                                                       np.maximum(Xs @ R + c, 0.0)], -1)
        for tn in acc:
            r = np.concatenate(RS[tn])                              # (S*W*Rr, T)
            if r.var() < 1e-20:
                continue
            rw = r.reshape(NW, Rr, T)
            rmean = rw.mean(1)                                      # fit target per (state, window, step)
            base = {"none": np.zeros_like(r), "constant": np.full_like(r, r.mean())}
            for name, X in feats.items():
                b = np.empty((NW, T))
                for tr in (fold_w, ~fold_w):
                    Xtr = X[tr].reshape(-1, X.shape[-1]).astype(np.float64)
                    w = ridge(Xtr, rmean[tr].reshape(-1))
                    b[~tr] = (X[~tr].reshape(-1, X.shape[-1]).astype(np.float64) @ w).reshape(-1, T)
                base[name] = np.repeat(b, Rr, 0)
            base["oracle"] = ((rw.sum(1, keepdims=True) - rw) / (Rr - 1)).reshape(-1, T)
            est = {v: momentum((r - base[v])[:, None, :] * gsc, r0.beta1) for v in variants}
            var0, se0 = est["none"].var(0), est["none"].std(0) / np.sqrt(len(r))
            for v in variants:
                acc[tn]["VR"][v].append(est[v].var(0) / (var0 + 1e-30))
                acc[tn]["BIAS"][v].append(np.abs(est[v].mean(0) - est["none"].mean(0)) / (se0 + 1e-30))
                acc[tn]["R2"][v].append(1 - np.var(r - base[v]) / np.var(r))
        print(f"  neuron {j}: {len(ks)} synapses, {Xc.shape[-1]} critic inputs", flush=True)
    res = {}
    for tn, d in acc.items():
        if not d["VR"]["none"]:
            continue
        res[tn] = {v: {"variance vs none": float(np.median(np.concatenate(d["VR"][v]))),
                       "bias (SE units)": float(np.median(np.concatenate(d["BIAS"][v]))),
                       "reward explained": float(np.median(d["R2"][v]))} for v in variants}
        print(f"== {tn} (slabs {slabs})")
        print(f"   {'baseline':18s} variance vs none   bias (SE)   reward explained")
        for v in variants:
            x = res[tn][v]
            print(f"   {v:18s} {x['variance vs none']:.3f}             {x['bias (SE units)']:.2f}        "
                  f"{x['reward explained']:+.2f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
