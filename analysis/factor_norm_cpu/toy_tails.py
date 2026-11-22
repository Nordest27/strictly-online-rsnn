"""Per-synapse signal-to-noise of the rule-averaged drift with the code's triangle gate vs tailed gates, on the same
sampled synapses (24 random neurons per layer, every recurrent synapse that transmits in the window), and the
drift-vs-derivative check for each gate on the network check's 20 synapses.
Usage: python toy_tails.py L gate [gate ...]"""
import glob, sys
import numpy as np
sys.path.insert(0, "/home/user/repos/ml_genn/examples/eprop")
import drift_toy as dt

L, gates = sys.argv[1], sys.argv[2:]
thr = 3 / np.sqrt(6144)
rec = dt.Recording(glob.glob(f"rec_L{L}/outputs/*drift_record.npz")[0])
net = np.load(glob.glob(f"dcw200_L{L}/outputs/*drift_check.npz")[0])
tg = next(gr for gr in rec.groups if gr["name"] == rec.target_group)
neurons = np.random.default_rng(7).choice(np.unique(tg["post"]), size=24, replace=False)
res = {}
for gate in ["drift"] + [g for g in gates if g != "drift"]:
    rng = np.random.default_rng(11)          # the same noise for every gate
    snr = []
    for j in neurons:
        src, pre, g0, sig, ids = rec.afferents(j)
        mine = [k for k, (name, _) in enumerate(ids) if name == rec.target_group]
        wr = np.repeat(np.arange(rec.W), 12)
        s, e, _ = dt.simulate(rec, j, wr, rng.standard_normal((len(wr), rec.T1)), gate=gate)
        f = dt.rule_filter(s[:, :, mine], rec.lam, rec.beta1)
        deliver, _ = rec.inputs(j)
        tx = deliver[:, 1:, mine].any((0, 1))
        snr += list(np.where(tx, np.abs(f.mean(0)) / (f.std(0) + 1e-30), np.nan))
    res[gate] = np.array(snr)
    out = dt.drift(rec, net["synapses"], float(net["h"]), rollouts=24, fd_rollouts=8, gate=gate)
    c = np.corrcoef(out["deriv_f"], out["score_f"])[0, 1]
    valid = ~np.isnan(res[gate])
    bad = res["drift"][valid] < thr
    print(f"L{L} {gate:7s}: SNR median {np.nanmedian(res[gate]):.3f} | below bound {np.mean(res[gate][valid] < thr):.0%}"
          f" | of the triangle's ill-conditioned synapses now measurable: {np.mean(res[gate][valid][bad] >= thr):.0%}"
          f" | drift vs derivative of its own eligibility: corr {c:.2f}", flush=True)
