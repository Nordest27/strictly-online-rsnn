"""Is the trained network decorrelated? Trained network vs the reservoir on identical frames (narrow_5l, 2026-10-30).

Per population (E sheets, channels kept separate), on membrane potentials (V) and on the refractory flags (spikes):
  * neighbour r: mean correlation between neurons adjacent on the sheet (same channel), and the same for random pairs;
  * synchrony: var(population mean) / mean(var(neuron)); 1 = every neuron moves together, ~1/N = independent.

Usage: python -I decorrelation_check.py LABEL=probe.npz:window [...]
"""
import sys
import numpy as np


def load(spec):
    label, _, rest = spec.partition("=")
    path, _, window = rest.rpartition(":")
    z = np.load(path, allow_pickle=False)
    X = z[f"{window}_x"]
    pops, off = {}, 0
    for lab, n, shp in zip(z["labels"], z["sizes"], z["shapes"]):
        pops[str(lab)] = (X[:, off:off + n].astype(np.float32), X[:, off + n:off + 2 * n].astype(np.float32),
                          tuple(int(s) for s in shp))
        off += 2 * n
    return label, pops


def stats(A, shp, rng):
    T = len(A)
    Az = (A - A.mean(0)) / (A.std(0) + 1e-6)
    live = A.std(0) > 1e-6
    h, w, c = shp
    idx = np.arange(h * w * c).reshape(h, w, c)
    a, b = idx[:, :-1, :].ravel(), idx[:, 1:, :].ravel()          # horizontal neighbours, same channel
    a2, b2 = idx[:-1, :, :].ravel(), idx[1:, :, :].ravel()        # vertical neighbours
    a, b = np.r_[a, a2], np.r_[b, b2]
    ok = live[a] & live[b]
    neigh = np.mean(np.sum(Az[:, a[ok]] * Az[:, b[ok]], 0) / T)
    ra, rb = rng.integers(0, h * w * c, 4000), rng.integers(0, h * w * c, 4000)
    ok = live[ra] & live[rb] & (ra != rb)
    rand = np.mean(np.sum(Az[:, ra[ok]] * Az[:, rb[ok]], 0) / T)
    sync = A[:, live].mean(1).var() / A[:, live].var(0).mean()
    return neigh, rand, sync, live.sum()


rng = np.random.default_rng(0)
for spec in sys.argv[1:]:
    label, pops = load(spec)
    print(f"\n== {label}")
    print(f"{'population':14s} {'V: neigh r':>10s} {'rand r':>7s} {'sync':>6s} | {'spk: neigh r':>12s} {'rand r':>7s} {'sync':>6s}")
    for name, (V, S, shp) in pops.items():
        if len(shp) != 3 or not (name.endswith("_E") or "field" in name):
            continue
        v, s = stats(V, shp, rng), stats(S, shp, rng)
        print(f"{name:14s} {v[0]:10.3f} {v[1]:7.3f} {v[2]:6.3f} | {s[0]:12.3f} {s[1]:7.3f} {s[2]:6.3f}")
