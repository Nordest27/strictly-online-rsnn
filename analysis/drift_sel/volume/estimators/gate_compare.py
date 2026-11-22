"""Drift gate centres and G score weights with the noise-free (shadow) sensitivity, in the open-loop replay.

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


def run(rec, j, wr, noise, k, g=None):
    s, e, v, parts = dt.simulate(rec, j, wr, noise, g=g, parts_k=[k])
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
    names_d = ["drift psi-psibar", "drift psi-psi_s", "drift psi-psibar_s", "drift cv"]
    names_g = ["G -psibar", "G -psi_s", "G -psibar_s"]
    M = {n: [] for n in names_d + names_g}; S = {n: [] for n in names_d + names_g}
    T = {"T_cen": [], "T_raw": [], "T_G": []}
    for syn in syns:
        j = int(tg["post"][syn])
        _, _, g0, _, ids = rec.afferents(j)
        k = ids.index((rec.target_group, int(syn)))
        wr = np.repeat(np.arange(W), a.rollouts)
        psi, pb, pre, xi, z = run(rec, j, wr, rng.standard_normal((len(wr), rec.T1)), k)
        ps_w, pbs_w = shadow(rec, j, W, k)
        ps, pbs = np.repeat(ps_w, a.rollouts, 0), np.repeat(pbs_w, a.rollouts, 0)
        base = pre * xi
        est = {"drift psi-psibar": rule((psi - pb) * base, rec), "drift psi-psi_s": rule((psi - ps) * base, rec),
               "drift psi-psibar_s": rule((psi - pbs) * base, rec),
               "drift cv": rule(((psi - pb) - (ps - pbs)) * base, rec)}
        for n, gate in (("G -psibar", -pb), ("G -psi_s", -ps), ("G -psibar_s", -pbs)):
            gtr = ltrace(gate * base, rec.lam)
            est[n] = momentum(z * gtr, rec.beta1)
        for n, x in est.items():
            M[n].append(x.mean()); S[n].append(abs(x.mean()) / (x.std() + 1e-30))
        # truths by finite differences
        wf = np.repeat(np.arange(W), a.fd); nz = rng.standard_normal((len(wf), rec.T1)); vals = []
        for sgn in (1, -1):
            g = np.tile(g0, (len(wf), 1)); g[:, k] += sgn * h
            psi_, pb_, pre_, _, z_ = run(rec, j, wf, nz, k, g)
            vals.append((rule(((psi_ - pb_) * pre_).mean(0)[None], rec)[0], rule((psi_ * pre_).mean(0)[None], rec)[0],
                         z_.sum(1).mean()))
        for i, key in enumerate(("T_cen", "T_raw", "T_G")):
            T[key].append((vals[0][i] - vals[1][i]) / (2 * h))
        print(f"  synapse {syn} done", flush=True)
    T = {k: np.array(v) for k, v in T.items()}
    res = {"truth corr": {"T_cen~T_raw": corr(T["T_cen"], T["T_raw"])}}
    print(f"truths: T_cen vs T_raw correlation {res['truth corr']['T_cen~T_raw']:.2f}")
    print(f"{'estimator':20s} {'~T_cen':>7s} {'~T_raw':>7s} {'~T_G':>7s} {'SNR':>7s}")
    for n in names_d + names_g:
        m = np.array(M[n])
        res[n] = {"T_cen": corr(m, T["T_cen"]), "T_raw": corr(m, T["T_raw"]), "T_G": corr(m, T["T_G"]),
                  "SNR": float(np.median(S[n]))}
        r = res[n]
        print(f"{n:20s} {r['T_cen']:7.2f} {r['T_raw']:7.2f} {r['T_G']:7.2f} {r['SNR']:7.3f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
