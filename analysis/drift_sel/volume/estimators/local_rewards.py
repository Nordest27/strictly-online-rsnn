"""Baselines for the truly local (TD-free) rewards on the shared route: how much a state critic removes.

Terms (each alone, unit weight), as in hidden_rule.py, from the open-loop replay of the 16-state recording:
  vreg   : -(max(0, V - thr) + max(0, -V - thr)),  thr = Vthresh + Beta A   (V after the step)
  rate   : FAvgTrace - |FAvg - FTarget|  (FAvg += (1 - a) per spike, decay a = 0.998; FTarget 0.01 per step;
           FAvgTrace a lambda-filter of |FAvg - FTarget|)
  decorr : -(z - zbar)(I - Ibar) / Ibar, I = |inhibitory input this step|, means with tau 1000 steps
Credited as the shared route GRADIENT_INSTANT: momentum average of (r - b) * g_score, g_score = -psibar * ZF * xi.
Baselines: none, constant, per-synapse critic (a + u * ZF_k), joint afferent critic (all presynaptic traces),
dedicated (K random neurons), oracle (mean of r over the other rollouts of the same window and step).
None of them uses delta or the neuron's own state, so all are unbiased (open loop).

Usage: python local_rewards.py RECDIR PREFIX [--slabs 0] [--neurons 6] [--rollouts 16] [--K 32]
"""
import sys, json, argparse
import numpy as np
sys.path.insert(0, "/home/user/repos/ml_genn/examples/eprop")
import drift_toy as dt
from compare_estimators import momentum
from compare_critics import ls

A_FAV, F_TARGET, DEC = 0.998, 0.01, float(np.exp(-1.0 / 1000.0))


def run_mean(x, d, x0):
    out = np.empty_like(x); m = np.broadcast_to(x0, (len(x),)).astype(np.float64).copy()
    for t in range(x.shape[1]):
        out[:, t] = m                      # the mean before this step's sample (as the code reads it)
        m = d * m + (1 - d) * x[:, t]
    return out


def terms(rec, j, wr, noise, ks, inh_k):
    s, e, v, parts = dt.simulate(rec, j, wr, noise, parts_k=ks)
    z = parts["z"]
    beta = rec.state["Beta"][wr, 0, j]; A = rec.state["A"][wr, 0, j].astype(np.float64)
    As = np.empty_like(z)
    for t in range(z.shape[1]):
        A = rec.rho * A + z[:, t]; As[:, t] = A
    thr = rec.vth + beta[:, None] * As
    vreg = -(np.maximum(0, v - thr) + np.maximum(0, -v - thr))
    fav = np.empty_like(z); f = np.full(len(z), z.mean())
    for t in range(z.shape[1]):
        f = A_FAV * f + (1 - A_FAV) * z[:, t]; fav[:, t] = f
    dev = np.abs(fav - F_TARGET)
    tr = np.empty_like(z); x = dev[:, 0].copy()
    for t in range(z.shape[1]):
        x = rec.lam * x + (1 - rec.lam) * dev[:, t]; tr[:, t] = x
    rate = tr - dev
    # inhibitory input this step: the inhibitory afferents' delivered weights (no noise)
    src, pre, g0, sig, _ = rec.afferents(j)
    deliver, _ = rec.inputs(j)
    T = z.shape[1]
    inh = np.abs((deliver[wr, :T][..., inh_k] * g0[inh_k]).sum(-1)) if len(inh_k) else np.zeros_like(z)
    zbar = run_mean(z, DEC, z.mean()); ibar = run_mean(inh, DEC, inh.mean())
    decorr = -(z - zbar) * (inh - ibar) / (ibar + 1e-6)
    gsc = np.moveaxis(-parts["psibar"][:, :, None] * parts["pre"] * parts["xi"], 1, -1)     # (N, k, T) instant
    return {"vreg": vreg, "rate": rate, "decorr": decorr}, gsc


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("recdir"); ap.add_argument("prefix")
    ap.add_argument("--slabs", default="0"); ap.add_argument("--neurons", type=int, default=6)
    ap.add_argument("--rollouts", type=int, default=16); ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--depth", type=int, default=8); ap.add_argument("--max-aff", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=None); a = ap.parse_args()
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
    variants = ["none", "constant", "persynapse", "afferent", "dedicated", "oracle"]
    acc = {tn: {"VR": {v: [] for v in variants}, "BIAS": {v: [] for v in variants}, "R2": {v: [] for v in variants}}
           for tn in ("vreg", "rate", "decorr")}
    for j in js:
        j = int(j)
        src, pre, g0, sig, ids = r0.afferents(j)
        rate_in = np.mean([rc.inputs(j)[0].mean((0, 1)) for rc in recs], 0)
        ks = [int(k) for k in np.argsort(-rate_in)[:a.max_aff] if rate_in[k] > 0 and sig[k] > 0]
        inh_k = np.flatnonzero(g0 < 0)
        if not ks:
            continue
        pool = [(si, n) for si, sz in enumerate(sizes) for n in range(sz) if not (si == tsrc and n == j)]
        ded = [pool[i] for i in rng.choice(len(pool), a.K, replace=False)]
        W, Rr = r0.W, a.rollouts
        RS = {tn: [] for tn in acc}; G, XA, XD, WID = [], [], [], []
        for rc in recs:
            wr = np.repeat(np.arange(W), Rr)
            rr, gsc = terms(rc, j, wr, rng.standard_normal((len(wr), rc.T1)), ks, inh_k)
            T = gsc.shape[-1]
            for tn in acc:
                RS[tn].append(rr[tn])
            _, trace = rc.inputs(j)
            G.append(gsc); XA.append(np.repeat(trace[:, :T], Rr, 0))
            XD.append(np.repeat(np.stack([rc.zf[si][:, :T, n] for si, n in ded], -1), Rr, 0)); WID.append(wr)
        gsc = np.concatenate(G); Xa, Xd, wid = np.concatenate(XA), np.concatenate(XD), np.concatenate(WID)
        N, T = Xa.shape[:2]
        one = np.ones((N, T, 1)); fold = (wid % 2 == 0)
        Fa, Fd = np.concatenate([one, Xa], -1), np.concatenate([one, Xd], -1)
        for tn in acc:
            r = np.concatenate(RS[tn])
            if r.var() < 1e-20:
                continue
            base = {"none": np.zeros_like(r), "constant": np.full_like(r, r.mean())}
            for name, X in (("afferent", Fa), ("dedicated", Fd)):
                b = np.empty_like(r)
                for tr in (fold, ~fold):
                    b[~tr] = X[~tr] @ ls(X[tr], r[tr])
                base[name] = b
            rr_ = r.reshape(len(recs) * W, Rr, T)
            base["oracle"] = ((rr_.sum(1, keepdims=True) - rr_) / (Rr - 1)).reshape(N, T)
            est = {v: momentum((r - base[v])[:, None, :] * gsc, r0.beta1) for v in base}
            ps = np.empty_like(est["none"])
            for i in range(len(ks)):
                Xk = np.concatenate([one, Xa[:, :, ks[i]:ks[i] + 1]], -1)
                b = np.empty_like(r)
                for tr in (fold, ~fold):
                    b[~tr] = Xk[~tr] @ ls(Xk[tr], r[tr])
                ps[:, i] = momentum(((r - b)[:, None, :] * gsc[:, i:i + 1]), r0.beta1)[:, 0]
            est["persynapse"] = ps; base["persynapse"] = None
            var0, se0 = est["none"].var(0), est["none"].std(0) / np.sqrt(N)
            for v in variants:
                acc[tn]["VR"][v].append(est[v].var(0) / (var0 + 1e-30))
                acc[tn]["BIAS"][v].append(np.abs(est[v].mean(0) - est["none"].mean(0)) / (se0 + 1e-30))
                if base[v] is not None:
                    acc[tn]["R2"][v].append(1 - np.var(r - base[v]) / np.var(r))
        print(f"  neuron {j}: {len(ks)} synapses, {len(inh_k)} inhibitory afferents", flush=True)
    res = {}
    for tn, d in acc.items():
        if not d["VR"]["none"]:
            continue
        res[tn] = {v: {"variance vs none": float(np.median(np.concatenate(d["VR"][v]))),
                       "bias (SE units)": float(np.median(np.concatenate(d["BIAS"][v]))),
                       "reward explained": float(np.median(d["R2"][v])) if d["R2"][v] else None} for v in variants}
        print(f"== {tn} (slabs {slabs})")
        print(f"   {'baseline':11s} variance vs none   bias (SE)   reward explained")
        for v in variants:
            x = res[tn][v]
            print(f"   {v:11s} {x['variance vs none']:.3f}             {x['bias (SE units)']:.2f}        "
                  + ("-" if x["reward explained"] is None else f"{x['reward explained']:+.2f}"))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
