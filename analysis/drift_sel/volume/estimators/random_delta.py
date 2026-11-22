"""Are the local, TD-gated rewards predictable from the state when the TD error carries no information?

The TD error is replaced by a random, slowly varying signal (Ornstein-Uhlenbeck, zero mean, unit sd, correlation time
--tau steps), drawn independently for every trajectory: independent of the state, the network and the neuron's
noise. Then every TD-gated reward r = delta * Q_i can only be predicted through its neuron-side factor Q_i, and
baselines of the form delta * Qhat (Qhat from noise-free features) are unbiased.

Q_i, one at a time: sensitivity  Q = -SensD (lambda-trace of psi - psibar)
                    contrast     Q = ContrD (lambda-trace of (HebbZ - HebbMean)^2, HebbZ the spike trace)
Credit as the rule: momentum average of (r - b) * Gtrace (lambda-trace of -psibar * ZF * xi), per 200-step window.

Baselines b (Qhat fitted by least squares of Q on the features, cross-fitted between halves of the windows):
  none, constant (mean of r), delta*mean(Q),
  delta * Qhat from: own (the current neuron critic's own-state features: noise-dependent, for comparison),
                     afferent (presynaptic traces), dedicated (K random neurons), neighbours (K nearest),
                     shadow (the noise-free replay's Q), shadow+nbr
  oracle: delta * (mean of Q over the other rollouts of the same window and step)
Reported: fraction of Q's variance each predictor explains; the estimator's variance relative to "none" (same mean
for unbiased baselines; lower = fewer samples needed); and a bias check (the shift of the mean estimate from
"none", in units of its standard error, median over synapses).

Usage: python random_delta.py RECDIR PREFIX [--slabs 0] [--neurons 6] [--rollouts 16] [--K 32] [--tau 300]
"""
import sys, json, argparse
import numpy as np
sys.path.insert(0, "/home/user/repos/ml_genn/examples/eprop")
import drift_toy as dt
from compare_estimators import momentum, ltrace
from compare_critics import filt, ls, FD

HEBB_DECAY = 0.9999


def terms(rec, j, wr, noise, ks):
    s, e, v, parts = dt.simulate(rec, j, wr, noise, parts_k=ks)
    pb, dev, z = parts["psibar"], parts["psi"] - parts["psibar"], parts["z"]
    sensd = ltrace(dev, rec.lam)
    # spike trace and its slow mean (started at the trace's average rate), contrast trace
    hz = np.zeros_like(z); h = np.zeros(len(z))
    for t in range(z.shape[1]):
        h = rec.alpha * h + z[:, t]; hz[:, t] = h
    hmean = np.full(len(z), z.mean() / (1 - rec.alpha))
    hm = np.empty_like(z)
    for t in range(z.shape[1]):
        hmean = HEBB_DECAY * hmean + (1 - HEBB_DECAY) * hz[:, t]; hm[:, t] = hmean
    contrd = ltrace((hz - hm) ** 2, rec.lam)
    lag = lambda x: np.concatenate([np.zeros((len(x), 1)), x[:, :-1]], 1)      # what the synapse stage of t sees
    gtr = ltrace(np.moveaxis(-pb[:, :, None] * parts["pre"] * parts["xi"], 1, -1), rec.lam)
    beta = rec.state["Beta"][wr, 0, j]; A = rec.state["A"][wr, 0, j].astype(np.float64)
    As = np.empty_like(z)
    for t in range(z.shape[1]):
        A = rec.rho * A + z[:, t]; As[:, t] = A
    phi = np.stack([filt(v - (rec.vth + beta[:, None] * As), FD), filt(As, FD), filt(z, FD)], -1)
    return {"sens": -lag(sensd), "contr": lag(contrd)}, gtr, phi


def ou(rng, n, T, tau):
    rho = np.exp(-1.0 / tau); x = np.empty((n, T)); x[:, 0] = rng.standard_normal(n)
    eps = rng.standard_normal((n, T)) * np.sqrt(1 - rho ** 2)
    for t in range(1, T):
        x[:, t] = rho * x[:, t - 1] + eps[:, t]
    return x


def main():
    import csv
    ap = argparse.ArgumentParser(); ap.add_argument("recdir"); ap.add_argument("prefix")
    ap.add_argument("--slabs", default="0"); ap.add_argument("--neurons", type=int, default=6)
    ap.add_argument("--rollouts", type=int, default=16); ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--tau", type=float, default=300.0); ap.add_argument("--depth", type=int, default=8)
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
    preds = ["own", "afferent", "dedicated", "neighbours", "shadow", "shadow+nbr"]
    variants = ["none", "constant", "delta*meanQ"] + [f"delta*{p}" for p in preds] + ["oracle"]
    est_extra = ["delta*persynapse"]
    res = {}
    for qname in ("sens", "contr"):
        VR = {v: [] for v in variants}; BIAS = {v: [] for v in variants}; R2 = {p: [] for p in preds + ["oracle"]}
        for j in js:
            j = int(j)
            src, pre, g0, sig, ids = r0.afferents(j)
            rate = np.mean([rc.inputs(j)[0].mean((0, 1)) for rc in recs], 0)
            ks = [int(k) for k in np.argsort(-rate)[:a.max_aff] if rate[k] > 0 and sig[k] > 0]
            if not ks:
                continue
            pool = [(si, n) for si, sz in enumerate(sizes) for n in range(sz) if not (si == tsrc and n == j)]
            ded = [pool[i] for i in rng.choice(len(pool), a.K, replace=False)]
            dx = np.abs(px - px[j]); dx = np.minimum(dx, side - dx); dy = np.abs(py - py[j]); dy = np.minimum(dy, side - dy)
            d2 = dx ** 2 + dy ** 2 + (pz - pz[j]) ** 2 + rng.random(n_tgt) * 1e-3; d2[j] = np.inf
            nbr = [(tsrc, int(n)) for n in np.argsort(d2)[:a.K]]
            W, Rr = r0.W, a.rollouts
            Q, G, PH, SH, XA, XD, XN, WID = [], [], [], [], [], [], [], []
            for rc in recs:
                wr = np.repeat(np.arange(W), Rr)
                q, gt, ph = terms(rc, j, wr, rng.standard_normal((len(wr), rc.T1)), ks)
                q0, _, _ = terms(rc, j, np.arange(W), np.zeros((W, rc.T1)), ks[:1])
                T = q[qname].shape[1]
                _, trace = rc.inputs(j)
                Q.append(q[qname]); G.append(gt); PH.append(ph); SH.append(np.repeat(q0[qname], Rr, 0))
                XA.append(np.repeat(trace[:, :T], Rr, 0))
                XD.append(np.repeat(np.stack([rc.zf[si][:, :T, n] for si, n in ded], -1), Rr, 0))
                XN.append(np.repeat(np.stack([rc.zf[si][:, :T, n] for si, n in nbr], -1), Rr, 0))
                WID.append(wr)
            q = np.concatenate(Q); gtr = np.concatenate(G); phi = np.concatenate(PH); sh = np.concatenate(SH)
            Xa, Xd, Xn, wid = np.concatenate(XA), np.concatenate(XD), np.concatenate(XN), np.concatenate(WID)
            N, T = q.shape
            delta = ou(rng, N, T, a.tau)
            r = delta * q
            one = np.ones((N, T, 1))
            feats = {"own": np.concatenate([one, phi], -1), "afferent": np.concatenate([one, Xa], -1),
                     "dedicated": np.concatenate([one, Xd], -1), "neighbours": np.concatenate([one, Xn], -1),
                     "shadow": np.concatenate([one, sh[..., None]], -1),
                     "shadow+nbr": np.concatenate([one, sh[..., None], Xn], -1)}
            fold = (wid % 2 == 0)
            adv = {"none": r, "constant": r - r.mean(), "delta*meanQ": r - delta * q.mean()}
            for p, X in feats.items():
                qhat = np.empty_like(q)
                for tr in (fold, ~fold):
                    te = ~tr
                    qhat[te] = X[te] @ ls(X[tr], q[tr])
                adv[f"delta*{p}"] = r - delta * qhat
                R2[p].append(1 - np.var(q - qhat) / np.var(q))
            qq = q.reshape(len(recs) * W, Rr, T)
            loo = ((qq.sum(1, keepdims=True) - qq) / (Rr - 1)).reshape(N, T)
            adv["oracle"] = r - delta * loo
            R2["oracle"].append(1 - np.var(q - loo) / np.var(q))
            est = {v: momentum(adv[v][:, None, :] * gtr, r0.beta1) for v in variants}     # (N, n_k)
            # per-synapse critics: synapse k predicts Q from its own presynaptic trace (a_k + u_k ZF_k) and uses
            # delta * that as its own baseline (a per-weight baseline: unbiased while delta is noise-free)
            ps = np.empty_like(est["none"])
            for i in range(len(ks)):
                Xk = np.concatenate([one, Xa[:, :, ks[i]:ks[i] + 1]], -1)
                qk = np.empty_like(q)
                for tr in (fold, ~fold):
                    te = ~tr
                    qk[te] = Xk[te] @ ls(Xk[tr], q[tr])
                ps[:, i] = momentum(((r - delta * qk)[:, None, :] * gtr[:, i:i + 1]), r0.beta1)[:, 0]
            est["delta*persynapse"] = ps
            var0 = est["none"].var(0)
            se0 = est["none"].std(0) / np.sqrt(N)
            for v in variants + est_extra:
                VR.setdefault(v, []); BIAS.setdefault(v, [])
                VR[v].append(est[v].var(0) / (var0 + 1e-30))
                BIAS[v].append(np.abs(est[v].mean(0) - est["none"].mean(0)) / (se0 + 1e-30))
            print(f"  [{qname}] neuron {j}: {len(ks)} synapses, Q explained: "
                  + ", ".join(f"{p} {R2[p][-1]:.2f}" for p in R2), flush=True)
        res[qname] = {v: {"variance vs none": float(np.median(np.concatenate(VR[v]))),
                          "bias (SE units)": float(np.median(np.concatenate(BIAS[v])))} for v in variants + est_extra}
        res[qname]["Q explained"] = {p: float(np.median(R2[p])) for p in R2}
        print(f"== {qname}: slabs {slabs}, tau {a.tau}")
        print(f"   Q explained: " + ", ".join(f"{p} {v:.2f}" for p, v in res[qname]["Q explained"].items()))
        print(f"   {'baseline':16s} variance vs none   bias (SE units)")
        for v in variants + est_extra:
            print(f"   {v:16s} {res[qname][v]['variance vs none']:.3f}             {res[qname][v]['bias (SE units)']:.2f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
