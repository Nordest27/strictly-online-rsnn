"""Exponential (EIF) upswing vs the standard neuron, on the same recorded inputs:
(1) the true spike sensitivity (Stein samples z H / Var(H) against the noise-free voltage): centre, width, asymmetry;
(2) how well e-prop's eligibility (assumed triangle psi at the nominal threshold x linear-decay presynaptic trace)
tracks the true derivative of the spike probability, dE[z_t]/dW (finite differences, common noise)."""
import glob, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, "/home/user/repos/ml_genn/examples/eprop")
import drift_toy as dt

L = sys.argv[1]
EIF = {"delta": float(sys.argv[2]), "vt": float(sys.argv[3]), "vpeak": float(sys.argv[4])}   # rate-matched per layer
rec = dt.Recording(glob.glob(f"rec_L{L}/outputs/*drift_record.npz")[0])
tg = next(gr for gr in rec.groups if gr["name"] == rec.target_group)
rng = np.random.default_rng(13)
neurons = rng.choice(np.unique(tg["post"]), size=8, replace=False)
for name, eif in (("standard", None), ("EIF", EIF)):
    xs, ws, el, dz, el_sum, dz_sum, rates = [], [], [], [], [], [], []
    for j in neurons:
        src, pre, g0, sig, ids = rec.afferents(j)
        mine = [k for k, (nm, _) in enumerate(ids) if nm == rec.target_group]
        deliver, _ = rec.inputs(j)
        tx = [k for k in mine if deliver[:, 1:, k].any()]
        wr = np.repeat(np.arange(rec.W), 8)
        noise = rng.standard_normal((len(wr), rec.T1))
        _, e, _, (x, w, z) = dt.simulate(rec, j, wr, noise, gate="psi", eif=eif, track=True)
        xs.append(x.ravel()); ws.append(w.ravel()); rates.append(z.mean() * 1000)
        h = 0.2 * float(np.sqrt(np.mean(g0 ** 2)))
        for k in rng.choice(tx, size=min(6, len(tx)), replace=False) if tx else []:
            gp = np.tile(g0, (len(wr), 1)); gp[:, k] += h
            gm = np.tile(g0, (len(wr), 1)); gm[:, k] -= h
            *_, (_, _, zp) = dt.simulate(rec, j, wr, noise, g=gp, gate="psi", eif=eif, track=True)
            *_, (_, _, zm) = dt.simulate(rec, j, wr, noise, g=gm, gate="psi", eif=eif, track=True)
            d = ((zp - zm) / (2 * h)).reshape(rec.W, 8, -1).mean(1)        # dE[z_t]/dW per window and step
            ek = e[:, :, k].reshape(rec.W, 8, -1).mean(1)
            # e-prop's e_t is the eligibility for z_t computed at step t+1's synapse stage: align e[:, 1:] with z[:, :-1]
            el.append(ek[:, 1:].ravel()); dz.append(d[:, :-1].ravel())
            el_sum.append(ek[:, 1:].sum(1)); dz_sum.append(d[:, :-1].sum(1))
    x, w = np.concatenate(xs), np.concatenate(ws)
    # sensitivity profile E[w | x]: x = noise-free voltage relative to the real spike threshold (binned)
    edges = np.arange(-0.8, 0.4, 0.025); mid = (edges[:-1] + edges[1:]) / 2
    idx = np.searchsorted(edges, x) - 1
    prof = np.array([w[idx == b].mean() if (idx == b).sum() > 50 else 0.0 for b in range(len(mid))])
    pos = np.maximum(prof, 0); pos = pos / (pos.sum() + 1e-30)
    cen = (pos * mid).sum(); wid = np.sqrt((pos * (mid - cen) ** 2).sum())
    below = pos[mid < -0.1].sum()
    EL, DZ = np.concatenate(el), np.concatenate(dz)
    ES, DS = np.concatenate(el_sum), np.concatenate(dz_sum)
    print(f"L{L} {name:8s}: rate {np.mean(rates):.0f} Hz | true sensitivity: centre {cen:.3f}, width (sd) {wid:.3f}, "
          f"share more than 0.1 below the spike threshold {below:.2f}")
    print(f"           e-prop eligibility vs true dE[z]/dW: per step corr {np.corrcoef(EL, DZ)[0,1]:.2f} "
          f"(Spearman {spearmanr(EL, DZ)[0]:.2f}) | summed over the window corr {np.corrcoef(ES, DS)[0,1]:.2f} "
          f"(Spearman {spearmanr(ES, DS)[0]:.2f}), sign agreement {np.mean(np.sign(ES) == np.sign(DS)):.2f}")
