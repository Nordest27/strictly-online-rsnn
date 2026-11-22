"""Synapse noise vs soma noise (level matched per neuron to its mean input spikes per step) as estimators, in the
open-loop replay: drift score (gate psi - psibar) vs the FD derivative of the centred eligibility, and G (gate -psibar,
reward = the neuron's spikes, lambda-trace) vs the FD gradient of the expected spike count; SNR per trajectory.
(Derived from gate_compare.py.)

Drift gate centres and G score weights with the noise-free (shadow) sensitivity, in the open-loop replay.

Shadow = the same neuron replayed with the same inputs and no injected noise: psi_s per step, psibar_s its running
mean (decay 0.99, started from the recorded PsiBar). Per synapse (rule average: lambda-trace, then momentum):
  drift scores, gate x ZF x xi:  A psi - psibar (current) | B psi - psi_s | C psi - psibar_s
  truths (finite differences, common random numbers):
     T_cen = d/dW of the rule average of E[(psi - psibar) ZF]   (centred eligibility, psibar from the run)
     T_raw = d/dW of the rule average of E[psi ZF]               (raw eligibility)
  G scores, gate x ZF x xi with gate -psibar (current) | -psi_s | -psibar_s, credited for the reward r_t = z_t
  (the neuron's spikes) through the lambda-trace (as GRADIENT_TRACE), momentum at the end of the window;
     truth T_G = d/dW of E[sum_t z_t] over the window
Reported: correlation across synapses of each estimator's mean with each truth, and the median per-trajectory SNR.

Usage: python gate_compare.py RECORD.npz [--n-syn 20] [--windows 64] [--rollouts 32]
"""
import sys, json, argparse
import numpy as np
sys.path.insert(0, "/home/user/repos/ml_genn/examples/eprop")
import drift_toy as dt
from compare_estimators import momentum, ltrace, corr


def run(rec, j, wr, noise, k, g=None, soma=None):
    s, e, v, parts = dt.simulate(rec, j, wr, noise, g=g, parts_k=[k], soma_sigma=soma)
    return parts["psi"], parts["psibar"], parts["pre"][:, :, 0], parts["xi"][:, :, 0], parts["z"]


def shadow(rec, j, W, k, g=None):
    psi, pb, pre, xi, z = run(rec, j, np.arange(W), np.zeros((W, rec.T1)), k, g)
    pbs = np.empty_like(psi); m = rec.state["PsiBar"][np.arange(W), 0, j].astype(np.float64)
    for t in range(psi.shape[1]):
        pbs[:, t] = m; m = 0.99 * m + 0.01 * psi[:, t]          # the code: psibar of the previous update at stage t
    return psi, pbs


def rule(x, rec):
    return momentum(ltrace(x, rec.lam), rec.beta1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("record"); ap.add_argument("--n-syn", type=int, default=20)
    ap.add_argument("--windows", type=int, default=64); ap.add_argument("--rollouts", type=int, default=32)
    ap.add_argument("--fd", type=int, default=16); ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=None); a = ap.parse_args()
    rec = dt.Recording(a.record); rng = np.random.default_rng(a.seed)
    tg = next(gr for gr in rec.groups if gr["name"] == rec.target_group)
    n_tgt = rec.state["V"].shape[2]; z_of = np.arange(n_tgt) % a.depth
    act = rec.state["Psi"].mean((0, 1)); pre_rate = rec.spikes[tg["source"]].mean((0, 1))
    ok = np.flatnonzero((z_of[tg["post"]] <= 2) & (act[tg["post"]] > np.percentile(act, 50))
                        & (pre_rate[tg["pre"]] > 0.002))
    syns = rng.choice(ok, min(a.n_syn, len(ok)), replace=False)
    h = 0.2 * float(np.sqrt(np.mean(tg["g"] ** 2)))
    W = min(a.windows, rec.W)
    modes = ["synapse", "soma"]
    names = [f"{q} {m}" for m in modes for q in ("drift", "G")]
    M = {n: [] for n in names}; S = {n: [] for n in names}
    T = {f"{t} {m}": [] for m in modes for t in ("T_cen", "T_G")}
    for syn in syns:
        j = int(tg["post"][syn])
        _, _, g0, sig, ids = rec.afferents(j)
        k = ids.index((rec.target_group, int(syn)))
        deliver, _ = rec.inputs(j)
        count = float(deliver.mean((0, 1)).sum())
        for m in modes:
            soma = None if m == "synapse" else float(sig[k]) * max(count, 1e-3)
            wr = np.repeat(np.arange(W), a.rollouts)
            psi, pb, pre, xi, z = run(rec, j, wr, rng.standard_normal((len(wr), rec.T1)), k, soma=soma)
            base = pre * xi
            est = {f"drift {m}": rule((psi - pb) * base, rec),
                   f"G {m}": momentum(z * ltrace(-pb * base, rec.lam), rec.beta1)}
            for n, x in est.items():
                M[n].append(x.mean()); S[n].append(abs(x.mean()) / (x.std() + 1e-30))
            wf = np.repeat(np.arange(W), a.fd); nz = rng.standard_normal((len(wf), rec.T1)); vals = []
            for sgn in (1, -1):
                g = np.tile(g0, (len(wf), 1)); g[:, k] += sgn * h
                psi_, pb_, pre_, _, z_ = run(rec, j, wf, nz, k, g, soma=soma)
                vals.append((rule(((psi_ - pb_) * pre_).mean(0)[None], rec)[0], z_.sum(1).mean()))
            T[f"T_cen {m}"].append((vals[0][0] - vals[1][0]) / (2 * h))
            T[f"T_G {m}"].append((vals[0][1] - vals[1][1]) / (2 * h))
        print(f"  synapse {syn} (inputs/step {count:.2f}) done", flush=True)
    T = {k: np.array(v) for k, v in T.items()}
    res = {}
    print(f"{'estimator':16s} {'accuracy':>9s} {'SNR':>8s}")
    for m in modes:
        for q, t in (("drift", "T_cen"), ("G", "T_G")):
            n = f"{q} {m}"
            res[n] = {"accuracy": corr(np.array(M[n]), T[f"{t} {m}"]), "SNR": float(np.median(S[n]))}
            print(f"{n:16s} {res[n]['accuracy']:9.2f} {res[n]['SNR']:8.4f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
