"""Per-neuron critics for the reward version of the drift, over many game states (multi-snapshot recording).

Reward of target neuron i in state k: r_t = -delta_k * SensD_{t-1}, delta_k = the network's TD error of the decision
the snapshot falls in (move log, first decision after the snapshot time). Credited through Gtrace (lambda-trace of
-psibar * ZF * xi), Adam momentum at the end of each 200-step window.

Baselines (fitted ones cross-fitted: trained on even windows, applied to odd ones and vice versa), each as
  "rb": advantage r_t - b_t (b fitted to the per-step reward, least squares), and
  "td": advantage r_t + gamma V_t - V_{t-1} (V at its TD(lambda) fixed point, LSTD):
  constant  : the mean reward
  own       : the code's neuron critic features (filtered V - threshold, A, spikes; bias), own state: noise-dependent
  afferent  : the neuron's presynaptic traces ("duplicated weights")
  dedicated : traces of K random other neurons of the whole network + input ("a dedicated connection set")
  shadow    : the noise-free replay of the same neuron (same inputs, no injected noise): its reward -delta SensD
  shadow+ded: shadow and dedicated together
  oracle    : mean of r_t over the other rollouts of the same window (upper bound for input-only baselines)
Truth: d/dW of E_states[ -delta_k * momentum(E[SensD]) ] -> sign-aligned as before (finite differences).
Note: open loop, so other neurons never carry the target's noise here; in the closed loop, neurons downstream of the
target do (after >= 1 synapse), which can bias "dedicated" if they are included.

Usage: python compare_critics.py RECDIR PREFIX [--slabs 0] [--neurons 6] [--rollouts 16] [--fd 8] [--K 32]
"""
import sys, json, csv, glob, argparse
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


def sim_terms(rec, j, wr, noise, ks, g=None):
    s, e, v, parts = dt.simulate(rec, j, wr, noise, g=g, parts_k=ks)
    pb, dev, z = parts["psibar"], parts["psi"] - parts["psibar"], parts["z"]
    sensd = ltrace(dev, rec.lam)
    sprev = np.concatenate([np.zeros((len(dev), 1)), sensd[:, :-1]], 1)
    gtr = ltrace(np.moveaxis(-pb[:, :, None] * parts["pre"] * parts["xi"], 1, -1), rec.lam) if ks else None
    beta = rec.state["Beta"][wr, 0, j]; A = rec.state["A"][wr, 0, j].astype(np.float64)
    As = np.empty_like(z)
    for t in range(z.shape[1]):
        A = rec.rho * A + z[:, t]; As[:, t] = A
    phi = np.stack([filt(v - (rec.vth + beta[:, None] * As), FD), filt(As, FD), filt(z, FD)], -1)
    return sprev, gtr, phi, sensd


def lstd(X, r, ridge=1e-3):
    """V = X w at the TD(lambda) fixed point. X (N, T, F), r (N, T)."""
    N, T, F = X.shape
    Am, bv, e = np.zeros((F, F)), np.zeros(F), np.zeros((N, F))
    for t in range(1, T):
        e = LAM_C * e + X[:, t - 1]
        Am += e.T @ (X[:, t - 1] - GAMMA * X[:, t]); bv += e.T @ r[:, t]
    return np.linalg.solve(Am + ridge * np.trace(Am) / F * np.eye(F) + 1e-9 * np.eye(F), bv)


def ls(X, r, ridge=1e-3):
    Xf, rf = X.reshape(-1, X.shape[-1]), r.reshape(-1)
    A = Xf.T @ Xf
    return np.linalg.solve(A + ridge * np.trace(A) / A.shape[0] * np.eye(A.shape[0]) + 1e-9 * np.eye(A.shape[0]), Xf.T @ rf)


def td_adv(r, V):
    return r + GAMMA * V - np.concatenate([V[:, :1], V[:, :-1]], 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("recdir"); ap.add_argument("prefix")
    ap.add_argument("--slabs", default="0"); ap.add_argument("--neurons", type=int, default=6)
    ap.add_argument("--rollouts", type=int, default=16); ap.add_argument("--fd", type=int, default=8)
    ap.add_argument("--K", type=int, default=32); ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--max-aff", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None); a = ap.parse_args()
    out = f"{a.recdir}/outputs/{a.prefix}(0)"
    times = dict(tuple(map(int, l.split())) for l in open(f"{out}_drift_record_times.txt"))
    moves = [(int(r["timestep"]), float(r["td_after"])) for r in csv.DictReader(open(f"{out}_moves.csv"))]
    recs, deltas = [], []
    for k in sorted(times):
        recs.append(dt.Recording(f"{out}_drift_record_{k}.npz"))
        deltas.append(next((td for ts, td in moves if ts > times[k]), moves[-1][1]))   # last snapshot: the last decision
    deltas = np.array(deltas)
    print(f"{len(recs)} states, windows {recs[0].W}, deltas {np.round(deltas, 2).tolist()}", flush=True)
    rng = np.random.default_rng(a.seed)
    r0 = recs[0]; n_tgt = r0.state["V"].shape[2]; z_of = np.arange(n_tgt) % a.depth
    slabs = [int(x) for x in a.slabs.split(",")]
    cand = np.flatnonzero(np.isin(z_of, slabs))
    act = np.mean([rc.state["Psi"][:, :, cand].mean((0, 1)) for rc in recs], 0)
    cand = cand[act > np.percentile(act, 25)]
    js = rng.choice(cand, min(a.neurons, len(cand)), replace=False)
    rms = np.sqrt(np.mean(np.concatenate([gr["g"] for gr in r0.groups if gr["name"] == r0.target_group]) ** 2))
    h = 0.2 * rms
    sizes = [z.shape[2] for z in r0.spikes]
    variants = ["none", "constant"] + [f"{b}_{f}" for b in ("own", "afferent", "dedicated", "neighbours", "shadow", "shadow+ded", "shadow+nbr")
                                       for f in ("rb", "td")] + [f"{b}xdelta_rb" for b in ("afferent", "dedicated", "neighbours")] + ["oracle"]
    M = {v: [] for v in variants}; S = {v: [] for v in variants}; TS = []; R2 = {v: [] for v in variants}
    SW = {v: [] for v in variants}
    for j in js:
        j = int(j)
        src, pre, g0, sig, ids = r0.afferents(j)
        rate = np.mean([rc.inputs(j)[0].mean((0, 1)) for rc in recs], 0)
        ks = [int(k) for k in np.argsort(-rate)[:a.max_aff] if rate[k] > 0 and sig[k] > 0]
        if not ks:
            continue
        # the dedicated set: K random neurons of all source populations (not the target itself)
        tsrc = next(gr["source"] for gr in r0.groups if gr["name"] == r0.target_group)   # the target's own population
        pool = [(si, n) for si, sz in enumerate(sizes) for n in range(sz) if not (si == tsrc and n == j)]
        ded = [pool[i] for i in rng.choice(len(pool), a.K, replace=False)]
        # the local set: the K nearest neurons of the target's own (E) population in the volume (toroidal x, y)
        side = int(round(np.sqrt(n_tgt / a.depth)))
        idx = np.arange(n_tgt); px, py, pz = idx // (side * a.depth), (idx // a.depth) % side, idx % a.depth
        dx = np.abs(px - px[j]); dx = np.minimum(dx, side - dx); dy = np.abs(py - py[j]); dy = np.minimum(dy, side - dy)
        d2 = dx ** 2 + dy ** 2 + (pz - pz[j]) ** 2 + rng.random(n_tgt) * 1e-3; d2[j] = np.inf
        nbr = [(tsrc, int(n)) for n in np.argsort(d2)[:a.K]]
        W, Rr = r0.W, a.rollouts
        rs, gtrs, phis, sh, Xa, Xd, Xn, sid, wid = [], [], [], [], [], [], [], [], []
        for k, rc in enumerate(recs):
            wr = np.repeat(np.arange(W), Rr)
            sp, gt, ph, _ = sim_terms(rc, j, wr, rng.standard_normal((len(wr), rc.T1)), ks)
            rs.append(-deltas[k] * sp); gtrs.append(gt); phis.append(ph)
            sp0, *_ = sim_terms(rc, j, np.arange(W), np.zeros((W, rc.T1)), ks[:1])   # shadow: no own noise
            sh.append(np.repeat(-deltas[k] * sp0, Rr, 0))
            T = sp.shape[1]
            _, trace = rc.inputs(j)
            Xa.append(np.repeat(trace[:, :T], Rr, 0).astype(np.float32))
            Xd.append(np.repeat(np.stack([rc.zf[si][:, :T, n] for si, n in ded], -1), Rr, 0).astype(np.float32))
            Xn.append(np.repeat(np.stack([rc.zf[si][:, :T, n] for si, n in nbr], -1), Rr, 0).astype(np.float32))
            sid.append(np.full(len(wr), k)); wid.append(wr)
        r = np.concatenate(rs); gtr = np.concatenate(gtrs); phi = np.concatenate(phis); sh = np.concatenate(sh)
        Xa = np.concatenate(Xa); Xd = np.concatenate(Xd); Xn = np.concatenate(Xn); sid = np.concatenate(sid); wid = np.concatenate(wid)
        N, T = r.shape
        one = np.ones((N, T, 1))
        feats_tmp_a, feats_tmp_d, feats_tmp_n = (np.concatenate([one, Xa], -1), np.concatenate([one, Xd], -1),
                                                 np.concatenate([one, Xn], -1))
        feats = {"own": np.concatenate([one, phi], -1), "afferent": np.concatenate([one, Xa], -1),
                 "dedicated": np.concatenate([one, Xd], -1), "shadow": np.concatenate([one, sh[..., None]], -1),
                 "shadow+ded": np.concatenate([one, sh[..., None], Xd], -1),
                 "neighbours": np.concatenate([one, Xn], -1), "shadow+nbr": np.concatenate([one, sh[..., None], Xn], -1)}
        adv = {"none": r, "constant": r - r.mean()}
        fold = (wid % 2 == 0)
        # delta-gated critic synapses: u . x predicts the neuron's own SensD (a reward-free local regression),
        # the baseline is that prediction times the broadcast TD error: b = -delta * (u . x)
        dlt = deltas[sid][:, None]
        sens_target = -r / np.where(dlt == 0, 1e-9, dlt)                       # = SensD_{t-1}
        for name, X in (("afferent", feats_tmp_a), ("dedicated", feats_tmp_d), ("neighbours", feats_tmp_n)):
            A_ = np.empty_like(r)
            for tr_mask in (fold, ~fold):
                te = ~tr_mask
                u = ls(X[tr_mask], sens_target[tr_mask]); A_[te] = r[te] + dlt[te] * (X[te] @ u)
            adv[f"{name}xdelta_rb"] = A_
        for name, X in feats.items():
            for form in ("rb", "td"):
                A_ = np.empty_like(r)
                for tr_mask in (fold, ~fold):                 # cross-fitting
                    te = ~tr_mask
                    if form == "rb":
                        w = ls(X[tr_mask], r[tr_mask]); A_[te] = r[te] - X[te] @ w
                    else:
                        w = lstd(X[tr_mask], r[tr_mask]); A_[te] = td_adv(r[te], X[te] @ w)
                adv[f"{name}_{form}"] = A_
        key = (sid * W + wid).reshape(-1, Rr)
        assert np.all(key == key[:, :1]), "trajectories are not grouped by (state, window)"
        rr = r.reshape(len(recs) * W, Rr, T)
        loo = (rr.sum(1, keepdims=True) - rr) / (Rr - 1)
        adv["oracle"] = (rr - loo).reshape(N, T)
        for v in variants:
            est = momentum(adv[v][:, None, :] * gtr, r0.beta1)
            M[v].append(est.mean(0)); S[v].append(np.abs(est.mean(0)) / (est.std(0) + 1e-30))
            R2[v].append(1 - adv[v].var() / r.var())
            # within-state SNR: per game state, |mean| / sd over that state's trajectories, then the median over
            # states (the sign flips of the true update between states are signal, not noise)
            SW[v].append(np.median([np.abs(est[sid == k].mean(0)) / (est[sid == k].std(0) + 1e-30)
                                    for k in range(len(recs))], 0))
        # truth
        t_k = np.zeros(len(ks))
        for k, rc in enumerate(recs):
            wf = np.repeat(np.arange(W), a.fd); nz = rng.standard_normal((len(wf), rc.T1))
            for i, kk in enumerate(ks):
                o = []
                for sgn in (1, -1):
                    g = np.tile(g0, (len(wf), 1)); g[:, kk] += sgn * h
                    *_, sd = sim_terms(rc, j, wf, nz, [kk], g=g)
                    o.append(momentum(sd.mean(0)[None], rc.beta1)[0])
                t_k[i] += deltas[k] * (o[0] - o[1]) / (2 * h) / len(recs)
        TS.append(t_k)
        print(f"  neuron {j}: {len(ks)} synapses; variance removed: " +
              ", ".join(f"{v} {R2[v][-1]:+.2f}" for v in variants if v not in ("none",)), flush=True)
    TS = np.concatenate(TS)
    res = {"slabs": slabs, "n_syn": int(len(TS)), "deltas": deltas.tolist()}
    for v in variants:
        m = np.concatenate(M[v])
        res[v] = {"SNR": float(np.median(np.concatenate(S[v]))), "SNR within state": float(np.median(np.concatenate(SW[v]))),
                  "accuracy": corr(m, TS), "variance removed": float(np.median(R2[v]))}
    w = max(len(v) for v in variants)
    print(f"{'baseline':{w}s}  SNR pooled  SNR in-state  accuracy  variance removed")
    for v in variants:
        print(f"{v:{w}s}  {res[v]['SNR']:.3f}       {res[v]['SNR within state']:.3f}         "
              f"{res[v]['accuracy']:+.2f}     {res[v]['variance removed']:+.2f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
