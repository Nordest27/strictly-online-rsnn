"""Hidden thresholds: can a neuron learn where its gate should be centred from its own spikes and noise?
Each neuron gets a random true threshold (the gate is not told). Phase 1 learns the centre (Stein-weighted mean of
the noise-free voltage, or the mean noise-free voltage at spikes) and a binned measured gate. Phase 2 compares the
rule-averaged drift per synapse for gates centred on the nominal / learned / true threshold (same noise)."""
import glob, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, "/home/user/repos/ml_genn/examples/eprop")
import drift_toy as dt

L = sys.argv[1]
rec = dt.Recording(glob.glob(f"rec_L{L}/outputs/*drift_record.npz")[0])
tg = next(gr for gr in rec.groups if gr["name"] == rec.target_group)
rng = np.random.default_rng(9)
neurons = rng.choice(np.unique(tg["post"]), size=12, replace=False)
gates = ["true", "nominal", "stein", "spike", "bins"]
D = {gt: [] for gt in gates}; err = {"nominal": [], "stein": [], "spike": []}; nspk = []
for j in neurons:
    th_true = float(np.clip(rec.vth * np.exp(0.2 * rng.standard_normal()), 0.4, 1.0))
    src, pre, g0, sig, ids = rec.afferents(j)
    mine = [k for k, (name, _) in enumerate(ids) if name == rec.target_group]
    deliver, _ = rec.inputs(j)
    tx = deliver[:, 1:, mine].any((0, 1))
    # phase 1: learn the centre from the neuron's own observables (its noise-free voltage, spikes and noise)
    wr = np.repeat(np.arange(rec.W), 8)
    _, _, _, (x, wgt, z) = dt.simulate(rec, j, wr, rng.standard_normal((len(wr), rec.T1)), th_true=th_true, track=True)
    sel = wgt != 0
    th_stein = float((x[sel] * wgt[sel]).sum() / wgt[sel].sum()) if sel.any() else rec.vth
    th_spike = float(x[z > 0].mean()) if (z > 0).any() else rec.vth
    nspk.append(int((z > 0).sum()))
    edges = np.arange(-0.8, 0.85, 0.05)
    xr = x - th_stein
    idx = np.clip(np.searchsorted(edges, xr) - 1, 0, len(edges) - 2)
    vals = np.array([wgt[idx == b].mean() if (idx == b).any() else 0.0 for b in range(len(edges) - 1)])
    vals = np.maximum(vals, 0.0)
    for name, th in (("nominal", rec.vth), ("stein", th_stein), ("spike", th_spike)):
        err[name].append(abs(th - th_true))
    # phase 2: drift per synapse, same noise for every gate
    wr2 = np.repeat(np.arange(rec.W), 12)
    noise = rng.standard_normal((len(wr2), rec.T1))
    for gt in gates:
        kw = dict(th_true=th_true)
        kw.update({"true": dict(th_gate=th_true), "nominal": dict(th_gate=rec.vth), "stein": dict(th_gate=th_stein),
                   "spike": dict(th_gate=th_spike), "bins": dict(th_gate=th_stein, gate_table=(edges, vals))}[gt])
        s, _, _ = dt.simulate(rec, j, wr2, noise, **kw)
        f = dt.rule_filter(s[:, :, mine], rec.lam, rec.beta1).mean(0)
        D[gt] += list(f[tx])
print(f"L{L}: 12 neurons, spikes used for learning per neuron (median) {int(np.median(nspk))}")
for name in ("nominal", "stein", "spike"):
    print(f"   |centre - true threshold|: {name:8s} median {np.median(err[name]):.3f}, max {np.max(err[name]):.3f}")
M = {gt: np.array(v) for gt, v in D.items()}
for gt in ("nominal", "stein", "spike", "bins"):
    print(f"   drift with gate '{gt:7s}' vs true-threshold gate: corr {np.corrcoef(M[gt], M['true'])[0,1]:.2f}, "
          f"Spearman {spearmanr(M[gt], M['true'])[0]:.2f}, sign agreement {np.mean(np.sign(M[gt]) == np.sign(M['true'])):.2f}  (n={len(M['true'])})")
