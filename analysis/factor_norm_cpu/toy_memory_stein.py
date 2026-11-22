"""Model-free estimate of dE[z_t]/dW from the neuron's own noise (Stein: spike x the synapse's noise trace / SynSig^2,
summed over the window) vs the true derivative (finite differences, common noise), next to e-prop's single-trace
eligibility, for the standard neuron, the resonator and the plateau neuron (rate-matched). Windows are processed in
chunks (memory). Reports per synapse-window and per synapse (aggregated over all windows), with split-half
repeatability of each side and noise-corrected correlations.
Usage: python toy_memory_stein.py L standard | resonator PERIOD GAIN | plateau GAIN"""
import glob, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, "/home/user/repos/ml_genn/examples/eprop")
import drift_toy as dt

L, model = sys.argv[1], sys.argv[2]
kw = {}
if model == "resonator":
    kw = {"resonator": {"period": float(sys.argv[3]), "gain": float(sys.argv[4])}}
elif model == "plateau":
    kw = {"plateau": {"gp": 0.08, "thp": 0.3, "s": 0.05, "tau": 30.0, "gain": float(sys.argv[3])}}
rec = dt.Recording(glob.glob(f"rec_L{L}/outputs/*drift_record.npz")[0])
tg = next(gr for gr in rec.groups if gr["name"] == rec.target_group)
rng = np.random.default_rng(21)
neurons = rng.choice(np.unique(tg["post"]), size=6, replace=False)
REPS, R2, CH = 64, 8, 16                       # Stein rollouts / FD pairs per window, windows per chunk
rows = []                                      # per synapse: arrays over windows (Stein halves, e-prop, FD halves)
for j in neurons:
    src, pre, g0, sig, ids = rec.afferents(j)
    mine = [k for k, (nm, _) in enumerate(ids) if nm == rec.target_group]
    deliver, _ = rec.inputs(j)
    tx = [k for k in mine if deliver[:, 1:, k].any()]
    if not tx:
        continue
    ks = list(rng.choice(tx, size=min(5, len(tx)), replace=False))
    h = 0.2 * float(np.sqrt(np.mean(g0 ** 2)))
    acc = {k: {"sa": [], "sb": [], "ep": [], "da": [], "db": []} for k in ks}
    for c0 in range(0, rec.W, CH):
        win = np.arange(c0, min(c0 + CH, rec.W)); nW = len(win)
        wr = np.repeat(win, REPS); noise = rng.standard_normal((len(wr), rec.T1))
        out = dt.simulate(rec, j, wr, noise, gate="psi", track=True, **kw)
        z = out[3][2]
        for k in ks:
            d = deliver[wr, :, k]
            nt = np.zeros(len(wr)); est = np.zeros(len(wr))
            for t in range(1, rec.T1):
                nt = rec.alpha * nt + sig[k] * noise[:, t - 1] * d[:, t - 1]
                est += z[:, t - 1] * nt / (sig[k] ** 2)
            est = est.reshape(nW, REPS)
            acc[k]["sa"].append(est[:, 0::2].mean(1)); acc[k]["sb"].append(est[:, 1::2].mean(1))
            ep = out[3][3][:, :, k].sum(1).reshape(nW, REPS).mean(1) if model != "standard" else \
                 (out[1][:, 1:, k].sum(1).reshape(nW, REPS).mean(1))          # standard: psi x ZFilter (e-prop)
            acc[k]["ep"].append(ep)
        del out, z
        wr2 = np.repeat(win, R2); n2 = rng.standard_normal((len(wr2), rec.T1))
        for k in ks:
            gp = np.tile(g0, (len(wr2), 1)); gp[:, k] += h
            gm = np.tile(g0, (len(wr2), 1)); gm[:, k] -= h
            zp = dt.simulate(rec, j, wr2, n2, g=gp, gate="psi", track=True, **kw)[3][2]
            zm = dt.simulate(rec, j, wr2, n2, g=gm, gate="psi", track=True, **kw)[3][2]
            dd = ((zp - zm) / (2 * h)).sum(1).reshape(nW, R2)
            acc[k]["da"].append(dd[:, 0::2].mean(1)); acc[k]["db"].append(dd[:, 1::2].mean(1))
    for k in ks:
        rows.append({n: np.concatenate(v) for n, v in acc[k].items()})

def rel(a, b):
    r = np.corrcoef(a, b)[0, 1]; return 2 * r / (1 + r)

def report(label, S_a, S_b, E, D_a, D_b):
    S, D = (S_a + S_b) / 2, (D_a + D_b) / 2
    rs, rd = rel(S_a, S_b), rel(D_a, D_b)
    cs, ce = np.corrcoef(S, D)[0, 1], np.corrcoef(E, D)[0, 1]
    corr = lambda c, ra: c / np.sqrt(max(ra, 1e-9) * max(rd, 1e-9)) if ra > 0 and rd > 0 else np.nan
    print(f"   {label}: repeatability ours {rs:.2f}, truth {rd:.2f} | ours vs truth corr {cs:.2f} (Spearman "
          f"{spearmanr(S, D)[0]:.2f}, noise-corrected {corr(cs, rs):.2f}) | e-prop vs truth corr {ce:.2f} (Spearman "
          f"{spearmanr(E, D)[0]:.2f}, noise-corrected {corr(ce, 1.0):.2f})")

print(f"L{L} {model} {kw}: {len(rows)} synapses x {rec.W} windows")
cat = lambda n: np.concatenate([r[n] for r in rows])
report("per synapse-window", cat("sa"), cat("sb"), cat("ep"), cat("da"), cat("db"))
agg = lambda n: np.array([r[n].mean() for r in rows])
report("per synapse (all windows)", agg("sa"), agg("sb"), agg("ep"), agg("da"), agg("db"))
