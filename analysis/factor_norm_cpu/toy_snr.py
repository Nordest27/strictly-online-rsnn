"""Single-trajectory signal-to-noise of the rule-averaged drift, per synapse, and the local quantities that could
explain it. One sample = one window x one own-noise rollout, filtered with the rule's trace and momentum."""
import glob, sys
import numpy as np
sys.path.insert(0, "/home/user/repos/ml_genn/examples/eprop")
import drift_toy as dt

for L in sys.argv[1:]:
    rec = dt.Recording(glob.glob(f"rec_L{L}/outputs/*drift_record.npz")[0])
    net = np.load(glob.glob(f"dcw200_L{L}/outputs/*drift_check.npz")[0])
    tg = next(gr for gr in rec.groups if gr["name"] == rec.target_group)
    rng = np.random.default_rng(1)
    rows = []
    for syn in net["synapses"]:
        j = int(tg["post"][syn])
        src, pre, g0, sig, ids = rec.afferents(j)
        k = ids.index((rec.target_group, syn))
        wr = np.repeat(np.arange(rec.W), 16)
        s, e, v = dt.simulate(rec, j, wr, rng.standard_normal((len(wr), rec.T1)))
        f = dt.rule_filter(s[:, :, k:k + 1], rec.lam, rec.beta1)[:, 0]          # one value per trajectory
        deliver, trace = rec.inputs(j)
        gate = None
        rows.append(dict(snr=abs(f.mean()) / f.std(), n_aff=len(g0),
                         pre_rate=deliver[:, :, k].mean() * 1000, aff_rate=deliver.mean() * 1000 * len(g0),
                         psi=rec.state["Psi"][:, :, j].mean(), v_sd=rec.state["V"][:, :, j].std(),
                         share=1.0 / max(deliver[:, :, :].sum(-1)[deliver[:, :, k]].mean(), 1e-9)))
    med = {key: np.median([r[key] for r in rows]) for key in rows[0]}
    print(f"L{L} medians: single-trajectory |mean|/sd {med['snr']:.3f} | afferents {med['n_aff']:.0f} | "
          f"this synapse's presynaptic rate {med['pre_rate']:.0f} Hz | total afferent spikes/s {med['aff_rate']:.0f} | "
          f"mean psi {med['psi']:.3f} | V sd {med['v_sd']:.3f} | share of the noise (1/active afferents when it transmits) {med['share']:.2f}")
