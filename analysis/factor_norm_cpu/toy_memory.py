"""Memory neurons (rate-matched): how well does e-prop's eligibility track the true dE[z_t]/dW (finite differences,
common noise)? Single decaying trace (unknown model) vs the eligibility vector derived from the neuron's Jacobian
(known model). Usage: python toy_memory.py L resonator PERIOD GAIN | python toy_memory.py L plateau GAIN"""
import glob, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, "/home/user/repos/ml_genn/examples/eprop")
import drift_toy as dt

L, model = sys.argv[1], sys.argv[2]
if model == "resonator":
    kw = {"resonator": {"period": float(sys.argv[3]), "gain": float(sys.argv[4])}}
else:
    kw = {"plateau": {"gp": 0.08, "thp": 0.3, "s": 0.05, "tau": 30.0, "gain": float(sys.argv[3])}}
rec = dt.Recording(glob.glob(f"rec_L{L}/outputs/*drift_record.npz")[0])
tg = next(gr for gr in rec.groups if gr["name"] == rec.target_group)
rng = np.random.default_rng(13)
neurons = rng.choice(np.unique(tg["post"]), size=8, replace=False)
E1, E2, DZ, rates = [], [], [], []
for j in neurons:
    src, pre, g0, sig, ids = rec.afferents(j)
    mine = [k for k, (nm, _) in enumerate(ids) if nm == rec.target_group]
    deliver, _ = rec.inputs(j)
    tx = [k for k in mine if deliver[:, 1:, k].any()]
    if not tx:
        continue
    wr = np.repeat(np.arange(rec.W), 8)
    noise = rng.standard_normal((len(wr), rec.T1))
    out = dt.simulate(rec, j, wr, noise, gate="psi", track=True, **kw)
    _, _, z, e1, e2 = out[3][:5]
    rates.append(z.mean() * 1000)
    h = 0.2 * float(np.sqrt(np.mean(g0 ** 2)))
    for k in rng.choice(tx, size=min(6, len(tx)), replace=False):
        gp = np.tile(g0, (len(wr), 1)); gp[:, k] += h
        gm = np.tile(g0, (len(wr), 1)); gm[:, k] -= h
        zp = dt.simulate(rec, j, wr, noise, g=gp, gate="psi", track=True, **kw)[3][2]
        zm = dt.simulate(rec, j, wr, noise, g=gm, gate="psi", track=True, **kw)[3][2]
        d = ((zp - zm) / (2 * h)).reshape(rec.W, 8, -1).mean(1)
        E1.append(e1[:, :, k].reshape(rec.W, 8, -1).mean(1).sum(1)); E2.append(e2[:, :, k].reshape(rec.W, 8, -1).mean(1).sum(1))
        DZ.append(d.sum(1))
E1, E2, DZ = map(np.concatenate, (E1, E2, DZ))
print(f"L{L} {model} {kw}: rate {np.mean(rates):.0f} Hz")
for name, E in (("single trace (unknown model)", E1), ("derived trace (known model)", E2)):
    print(f"   e-prop {name:38s} vs true dE[z]/dW (window sums): corr {np.corrcoef(E, DZ)[0,1]:.2f}, "
          f"Spearman {spearmanr(E, DZ)[0]:.2f}, sign agreement {np.mean(np.sign(E) == np.sign(DZ)):.2f}  (n={len(DZ)})")
