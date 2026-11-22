"""noise_site "soma" in the open-loop replay: (1) does the drift score still estimate the weight derivative of the
eligibility (rule-averaged correlation with finite differences, as the drift check), synapse vs soma noise on the
same synapses; (2) inputs firing per step onto each neuron, by slab (the gain that matches the total noise).

Usage: python soma_noise_check.py RECORD.npz [--n-syn 40] [--depth 8]
"""
import sys, json, argparse
import numpy as np
sys.path.insert(0, "/home/user/repos/ml_genn/examples/eprop")
import drift_toy as dt


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b) + 1e-30))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("record"); ap.add_argument("--n-syn", type=int, default=40)
    ap.add_argument("--depth", type=int, default=8); ap.add_argument("--rollouts", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--windows", type=int, default=64); a = ap.parse_args()
    rec = dt.Recording(a.record); rng = np.random.default_rng(a.seed)
    tg = next(gr for gr in rec.groups if gr["name"] == rec.target_group)
    n_tgt = rec.state["V"].shape[2]; z_of = np.arange(n_tgt) % a.depth
    # (2) inputs firing per step onto every target neuron
    count = np.zeros(n_tgt)
    for j in range(n_tgt):
        deliver, _ = rec.inputs(j)
        count[j] = deliver.mean((0, 1)).sum()
    by_slab = {int(z): float(np.mean(count[z_of == z])) for z in range(a.depth)}
    sig_all = np.concatenate([gr["syn_sig"] for gr in rec.groups if gr["syn_sig"] is not None])
    print("SynSig: min %.4g max %.4g (uniform: %s)" % (sig_all.min(), sig_all.max(), np.allclose(sig_all, sig_all[0])))
    print("inputs firing per step, mean by slab:", {k: round(v, 2) for k, v in by_slab.items()},
          "overall mean %.2f" % count.mean())
    # (1) drift check, synapse vs soma noise: synapses onto neurons of the first slabs whose input fires
    act = rec.state["Psi"].mean((0, 1))
    pre_rate = rec.spikes[tg["source"]].mean((0, 1))
    ok = np.flatnonzero((z_of[tg["post"]] <= 2) & (act[tg["post"]] > np.percentile(act, 50))
                        & (pre_rate[tg["pre"]] > 0.002))
    syn = rng.choice(ok, min(a.n_syn, len(ok)), replace=False)
    rms = float(np.sqrt(np.mean(tg["g"] ** 2))); h = 0.2 * rms
    sigma = float(sig_all[0])
    out = {}
    for name, ss in (("synapse", None), ("soma", sigma * count.mean())):
        r = dt.drift(rec, syn, h, rollouts=a.rollouts, fd_rollouts=16, seed=a.seed, soma_sigma=ss, windows=np.arange(min(a.windows, rec.W)))
        s, d = r["score_f"], r["deriv_f"]
        out[name] = {"score~deriv": corr(s, d), "score split-half": corr(r["score_f_a"], r["score_f_b"]),
                     "deriv split-half": corr(r["deriv_f_a"], r["deriv_f_b"]),
                     "scale (score/deriv)": float(s @ d / (d @ d + 1e-30))}
        print(name, json.dumps(out[name]), flush=True)


if __name__ == "__main__":
    main()
