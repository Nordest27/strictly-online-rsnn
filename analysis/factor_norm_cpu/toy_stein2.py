"""Model-free drift via second-order Stein (gate = spike x the neuron's own noise trace / variance) vs the psi-gated
drift, on recorded network neurons: rule-averaged mean per synapse, agreement between gates, per-trajectory SNR."""
import glob, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, "/home/user/repos/ml_genn/examples/eprop")
import drift_toy as dt

L = sys.argv[1]
rec = dt.Recording(glob.glob(f"rec_L{L}/outputs/*drift_record.npz")[0])
tg = next(gr for gr in rec.groups if gr["name"] == rec.target_group)
rng = np.random.default_rng(5)
neurons = rng.choice(np.unique(tg["post"]), size=12, replace=False)
gates = ["drift", "stein2", "stein2x"]
means = {gt: [] for gt in gates}; snrs = {gt: [] for gt in gates}; halves = {gt: [] for gt in gates}
for j in neurons:
    src, pre, g0, sig, ids = rec.afferents(j)
    mine = [k for k, (name, _) in enumerate(ids) if name == rec.target_group]
    deliver, _ = rec.inputs(j)
    tx = deliver[:, 1:, mine].any((0, 1))
    reps = 16
    wr = np.repeat(np.arange(rec.W), reps)
    noise = rng.standard_normal((len(wr), rec.T1)); noise2 = rng.standard_normal((len(wr), rec.T1))
    for gt in gates:
        kw = dict(noise2=noise2, sigma2=2.0 * float(np.mean(sig))) if gt == "stein2x" else {}
        s, _, _ = dt.simulate(rec, j, wr, noise, gate=gt, **kw)
        f = dt.rule_filter(s[:, :, mine], rec.lam, rec.beta1)                  # (trajectories, synapses)
        m = f.mean(0)
        means[gt] += list(m[tx]); snrs[gt] += list((np.abs(m) / (f.std(0) + 1e-30))[tx])
        halves[gt] += list(np.stack([f[0::2].mean(0), f[1::2].mean(0)], 1)[tx])
for gt in gates:
    h = np.array(halves[gt]); r = np.corrcoef(h[:, 0], h[:, 1])[0, 1]
    print(f"L{L} {gt:8s}: per-trajectory SNR median {np.median(snrs[gt]):.4f} | split-half agreement of the means {r:.2f}")
M = {gt: np.array(v) for gt, v in means.items()}
for a, b in [("drift", "stein2"), ("drift", "stein2x"), ("stein2", "stein2x")]:
    print(f"L{L} {a} vs {b}: corr {np.corrcoef(M[a], M[b])[0,1]:.2f}, Spearman {spearmanr(M[a], M[b])[0]:.2f}, "
          f"sign agreement {np.mean(np.sign(M[a]) == np.sign(M[b])):.2f}  (n={len(M[a])})")
