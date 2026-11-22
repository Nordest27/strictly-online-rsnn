"""Soma noise with the extra presynaptic factor (the "soma fix"), on the drift toy (open-loop replay of recorded
neurons from the 5-layer CPU Snake network, layers 1 and 5).

For each sampled synapse and recorded window, rule-averaged (lambda trace, then beta1 momentum) as in drift_check:
  syn        : synapse (input-gated) noise, the rule's score  g pi xi_syn                vs de/dW (synapse noise)
  soma       : soma noise added to the membrane, score       g pi H_soma                 vs de/dW (soma noise)
  soma_fix   : soma noise, score with one more presynaptic factor  g pi^2 H_soma          vs de/dW (soma noise)
  soma_dpsi  : the plain soma score against d(gate)/dW alone (de/dW / pi, per step), its expected target
g = psi - psibar (the drift gate), pi the presynaptic trace, H_soma the soma noise trace (the same filter as the
synaptic input). de/dW by finite differences of the mean eligibility (common random numbers). Reported: Pearson
correlation over (synapse, window) pairs, and the split-half reliability of each side (alternate windows).
Usage: python soma_fix.py OUT.json [--synapses 24] [--rollouts 48] [--fd 16] [--windows 64]
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "/home/user/repos/ml_genn/examples/eprop")
import drift_toy as dt  # noqa: E402

REC = "/home/user/repos/personal/analysis/factor_norm_cpu/rec_{L}/outputs/rec_{L}(0)_drift_record.npz"


def soma_trace(noise, a, T):
    """The soma noise trace at the synapse stage of steps 1..T (noise[:, t-1] delivered in step t)."""
    H, out = np.zeros(noise.shape[0]), []
    for t in range(1, T + 1):
        out.append(H.copy())
        H = a * H + noise[:, t - 1]
    return np.stack(out, 1)


def one_synapse(rec, syn, windows, R, F, h, rng):
    tg = next(gr for gr in rec.groups if gr["name"] == rec.target_group)
    j = int(tg["post"][syn])
    _, _, g0, sig, ids = rec.afferents(j)
    k = ids.index((rec.target_group, syn))
    # soma noise matched to this neuron's total synaptic noise per step (one shared sample x sum of SynSig x delivery)
    deliver, _ = rec.inputs(j)
    soma_sigma = float(np.sqrt(np.mean(((deliver[windows] * sig).sum(-1)) ** 2)) + 1e-9)
    T = rec.T1 - 1
    nW = len(windows)
    wr = np.repeat(windows, R)
    out = {}
    # synapse noise: the rule's own score
    s, e, _ = dt.simulate(rec, j, wr, rng.standard_normal((len(wr), rec.T1)))
    out["syn_score"] = s[:, :, k].reshape(nW, R, T).mean(1)
    # soma noise: build the scores from the parts
    noise = rng.standard_normal((len(wr), rec.T1))
    _, _, _, parts = dt.simulate(rec, j, wr, noise, soma_sigma=soma_sigma, parts_k=[k])
    g = parts["psi"] - parts["psibar"]                                        # (R, T)
    pre = parts["pre"][..., 0]                                                # (R, T)
    H = soma_trace(noise, rec.alpha, T) * soma_sigma
    kappa = 1 - rec.alpha ** 2
    plain = kappa * g * pre * H / soma_sigma ** 2
    out["soma_score"] = plain.reshape(nW, R, T).mean(1)
    out["soma_fix_score"] = (plain * pre).reshape(nW, R, T).mean(1)
    # finite-difference de/dW under each noise
    wf = np.repeat(windows, F)
    for name, ss in (("syn", None), ("soma", soma_sigma)):
        nz = rng.standard_normal((len(wf), rec.T1))
        gp = np.tile(g0, (len(wf), 1)); gp[:, k] += h
        gm = np.tile(g0, (len(wf), 1)); gm[:, k] -= h
        _, ep, _ = dt.simulate(rec, j, wf, nz, g=gp, soma_sigma=ss)
        _, em, _ = dt.simulate(rec, j, wf, nz, g=gm, soma_sigma=ss)
        D = ((ep[:, :, k] - em[:, :, k]) / (2 * h)).reshape(nW, F, T).mean(1)
        out[f"{name}_deriv"] = D
    pre_w = pre.reshape(nW, R, T).mean(1)
    out["soma_dgate"] = out["soma_deriv"] / np.maximum(pre_w, 1e-3)              # d(gate)/dW ~ de/dW / pi
    out["_soma_sigma"] = soma_sigma
    return out


def filt(X, rec):
    return np.array([dt.rule_filter(x[:, None], rec.lam, rec.beta1)[0] for x in X])


def corr(a, b):
    a, b = np.ravel(a), np.ravel(b)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out")
    ap.add_argument("--synapses", type=int, default=24); ap.add_argument("--rollouts", type=int, default=48)
    ap.add_argument("--fd", type=int, default=16); ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--h", type=float, default=0.02); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    result = {}
    for L in ("L1", "L5"):
        rec = dt.Recording(REC.format(L=L))
        tg = next(gr for gr in rec.groups if gr["name"] == rec.target_group)
        windows = np.sort(rng.choice(rec.W, a.windows, replace=False))
        # synapses whose input fires in the chosen windows (silent ones carry no drift), sampled at random
        src_spk = {}
        cand = []
        for syn in range(len(tg["post"])):
            sp = rec.spikes[tg["source"]][windows][:, :, tg["pre"][syn]].sum()
            if sp >= a.windows:                       # at least ~1 spike per window on average
                cand.append(syn)
        syns = rng.choice(cand, min(a.synapses, len(cand)), replace=False)
        rows, sigmas = {}, []
        for i, syn in enumerate(syns):
            o = one_synapse(rec, int(syn), windows, a.rollouts, a.fd, a.h, rng)
            sigmas.append(o.pop("_soma_sigma"))
            for key, v in o.items():
                rows.setdefault(key, []).append(filt(v, rec))
            print(f"{L} synapse {i + 1}/{len(syns)}", flush=True)
        X = {k: np.array(v) for k, v in rows.items()}                    # (n_syn, nW)
        res = {
            "syn vs de/dW": corr(X["syn_score"], X["syn_deriv"]),
            "soma plain vs de/dW": corr(X["soma_score"], X["soma_deriv"]),
            "soma fix vs de/dW": corr(X["soma_fix_score"], X["soma_deriv"]),
            "soma plain vs dgate/dW": corr(X["soma_score"], X["soma_dgate"]),
        }
        for name in ("syn_score", "soma_score", "soma_fix_score", "syn_deriv", "soma_deriv"):
            res[f"reliability {name}"] = corr(X[name][:, 0::2].mean(1), X[name][:, 1::2].mean(1)) if X[name].shape[1] > 3 else None
        res["soma_sigma_mean"] = float(np.mean(sigmas)); res["n_synapses"] = int(len(syns)); res["windows"] = int(a.windows)
        result[L] = res
        print(L, json.dumps(res, indent=1), flush=True)
    json.dump(result, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
