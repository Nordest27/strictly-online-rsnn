"""Drift term vs its reward version (local.sensitivity) as estimators, in the trained v031 volume, by slab.

Open-loop replay of recorded windows (drift_toy.py). Per sampled synapse (all afferents of sampled target neurons in
the front / middle / back slabs), with the TD error set to 1 (both versions are multiplied by the same delta):
  drift  : rule average (lambda-trace, then Adam momentum beta1) of (psi - psibar) * ZF * xi       [p_score]
  reward : momentum average of -(SensD - b) * Gtrace, Gtrace = lambda-trace of -psibar * ZF * xi    [g_score route]
           SensD = the neuron's lambda-trace of (psi - psibar), b = its overall mean (a long running mean)
Truths (finite differences of the weight, common random numbers):
  T_e : d/dW of the rule average of E[e], e = (psi - psibar) * ZF   (what the drift was shown to estimate)
  T_S : d/dW of the momentum average of E[SensD]                       (the neuron's sensitivity trace)
Reported per slab group: correlation across synapses of each estimator (mean over windows and rollouts) with each
truth; split-half reliability of the estimators; per-trajectory SNR (|mean| / sd over single trajectories).

Usage: python compare_estimators.py RECORD.npz [--neurons 8] [--rollouts 32] [--fd 8] [--depth 8]
"""
import sys, json, argparse
import numpy as np
sys.path.insert(0, "/home/user/repos/ml_genn/examples/eprop")
import drift_toy as dt


def momentum(x, beta1):
    """(..., T) -> momentum average at the end of the window."""
    T = x.shape[-1]
    w = (1 - beta1) * beta1 ** np.arange(T - 1, -1, -1)
    return x @ w


def ltrace(x, lam):
    """(..., T) -> lambda-trace per step."""
    out = np.empty_like(x); tr = np.zeros(x.shape[:-1])
    for t in range(x.shape[-1]):
        tr = lam * tr + x[..., t]; out[..., t] = tr
    return out


def per_step(rec, j, wr, noise, ks, g=None):
    """Per trajectory and afferent k: drift per-step score, reward per-step term parts; e and SensD for truths."""
    s, e, _, parts = dt.simulate(rec, j, wr, noise, g=g, parts_k=ks)
    xi, pre, psi, pb = parts["xi"], parts["pre"], parts["psi"], parts["psibar"]      # (R, T, k) / (R, T)
    dev = psi - pb
    sensd = ltrace(dev, rec.lam)                                     # (R, T): after the neuron update of step t
    sens_prev = np.concatenate([np.zeros((len(dev), 1)), sensd[:, :-1]], 1)   # what the synapse stage of t sees
    gsc = -pb[:, :, None] * pre * xi                                  # g_score per step (R, T, k)
    return {"p": s[:, :, ks], "e": e[:, :, ks], "sens_prev": sens_prev, "sensd": sensd, "gsc": gsc}


def analyse(rec, j, ks, rollouts, fd, h, rng):
    W = rec.W; wr = np.repeat(np.arange(W), rollouts)
    d = per_step(rec, j, wr, rng.standard_normal((len(wr), rec.T1)), ks)
    lam, b1 = rec.lam, rec.beta1
    drift_tr = momentum(ltrace(np.moveaxis(d["p"], 1, -1), lam), b1)               # (R, k)
    gtr = ltrace(np.moveaxis(d["gsc"], 1, -1), lam)                                 # (R, k, T)
    b = d["sens_prev"].mean()
    rew_tr = momentum(-(d["sens_prev"] - b)[:, None, :] * gtr, b1)                  # (R, k)
    # truths by finite differences
    _, _, g0, _, _ = rec.afferents(j)
    wf = np.repeat(np.arange(W), fd); nz = rng.standard_normal((len(wf), rec.T1))
    Te, TS = [], []
    for k in ks:
        out = []
        for sgn in (+1, -1):
            g = np.tile(g0, (len(wf), 1)); g[:, k] += sgn * h
            dd = per_step(rec, j, wf, nz, [k], g=g)
            out.append((momentum(ltrace(dd["e"][:, :, 0].mean(0)[None], lam), b1)[0],
                        momentum(dd["sensd"].mean(0)[None], b1)[0]))
        Te.append((out[0][0] - out[1][0]) / (2 * h)); TS.append((out[0][1] - out[1][1]) / (2 * h))
    return drift_tr, rew_tr, np.array(Te), np.array(TS)


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b) + 1e-30))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("record")
    ap.add_argument("--neurons", type=int, default=8); ap.add_argument("--rollouts", type=int, default=32)
    ap.add_argument("--fd", type=int, default=8); ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--max-aff", type=int, default=24); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None); a = ap.parse_args()
    rec = dt.Recording(a.record)
    rng = np.random.default_rng(a.seed)
    D = a.depth
    n_tgt = rec.state["V"].shape[2]
    print(f"recording: {rec.W} windows x {rec.T1 - 1} steps, {n_tgt} targets, lambda {rec.lam}, beta1 {rec.beta1}, "
          f"replay error {dt.validate(rec, 0, 4):.1e}", flush=True)
    rms = np.sqrt(np.mean(np.concatenate([gr["g"] for gr in rec.groups if gr["name"] == rec.target_group]) ** 2))
    h = 0.2 * rms
    groups = {"front (slab 0)": [0], "middle (slabs 2-4)": [2, 3, 4], "back (slabs 5-7)": [5, 6, 7]}
    z = np.arange(n_tgt) % D
    res = {}
    for gname, slabs in groups.items():
        cand = np.flatnonzero(np.isin(z, slabs))
        if len(cand) == 0:
            continue
        # neurons that are not silent in the recording (otherwise every estimator is zero)
        act = rec.state["Psi"][:, :, cand].mean((0, 1))
        cand = cand[act > np.percentile(act, 25)]
        js = rng.choice(cand, min(a.neurons, len(cand)), replace=False)
        rows = {k: [] for k in ("drift", "reward", "Te", "TS", "drift_a", "drift_b", "rew_a", "rew_b",
                                "snr_drift", "snr_rew", "pre_rate")}
        for j in js:
            src, pre, g0, sig, ids = rec.afferents(int(j))
            deliver, _ = rec.inputs(int(j))
            rate = deliver.mean((0, 1))
            ks = [int(k) for k in np.argsort(-rate)[:a.max_aff] if rate[k] > 0 and sig[k] > 0]
            if not ks:
                continue
            dr, rw, Te, TS = analyse(rec, int(j), ks, a.rollouts, a.fd, h, rng)
            W = rec.W
            drw, rww = dr.reshape(W, a.rollouts, -1), rw.reshape(W, a.rollouts, -1)
            rows["drift"].append(dr.mean(0)); rows["reward"].append(rw.mean(0))
            rows["drift_a"].append(drw[0::2].mean((0, 1))); rows["drift_b"].append(drw[1::2].mean((0, 1)))
            rows["rew_a"].append(rww[0::2].mean((0, 1))); rows["rew_b"].append(rww[1::2].mean((0, 1)))
            rows["snr_drift"].append(np.abs(dr.mean(0)) / (dr.std(0) + 1e-30))
            rows["snr_rew"].append(np.abs(rw.mean(0)) / (rw.std(0) + 1e-30))
            rows["Te"].append(Te); rows["TS"].append(TS); rows["pre_rate"].append(rate[ks])
            print(f"  {gname}: neuron {j} ({len(ks)} synapses) done", flush=True)
        R = {k: np.concatenate(v) for k, v in rows.items()}
        res[gname] = {
            "n_syn": int(len(R["drift"])),
            "drift~Te": corr(R["drift"], R["Te"]), "drift~TS": corr(R["drift"], R["TS"]),
            "reward~Te": corr(R["reward"], R["Te"]), "reward~TS": corr(R["reward"], R["TS"]),
            "Te~TS": corr(R["Te"], R["TS"]),
            "drift split-half": corr(R["drift_a"], R["drift_b"]), "reward split-half": corr(R["rew_a"], R["rew_b"]),
            "SNR/trajectory drift (median)": float(np.median(R["snr_drift"])),
            "SNR/trajectory reward (median)": float(np.median(R["snr_rew"])),
            "pre spikes/step (median)": float(np.median(R["pre_rate"])),
        }
        print(gname, json.dumps(res[gname], indent=1), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
