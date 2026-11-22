"""How well does the learned credit prediction L track its target Y = delta * H (recorded credit run)?

Per neuron (the recorded L1_E population, all states and windows pooled): the correlation of L with Y step by step and
of their per-window means, against a null (Y taken from another window of the same state, shuffled); the slope of Y on
L (~1 if calibrated) and std(L) / std(Y). Reported as medians over neurons, and per slab (depth 8).

Usage: python credit_quality.py RECDIR PREFIX [--depth 8]
"""
import sys, argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("recdir"); ap.add_argument("prefix"); ap.add_argument("--depth", type=int, default=8)
    a = ap.parse_args()
    out = f"{a.recdir}/outputs/{a.prefix}(0)"
    times = [tuple(map(int, l.split())) for l in open(f"{out}_drift_record_times.txt")]
    L, Y = [], []
    for k, _ in times:
        d = np.load(f"{out}_drift_record_{k}.npz", allow_pickle=True)
        L.append(d["state_CreditL"][:, 1:]); Y.append(d["state_CreditY"][:, 1:])
    L, Y = np.concatenate(L), np.concatenate(Y)                    # (states*windows, T, n)
    nW, T, n = L.shape
    rng = np.random.default_rng(0)
    perm = rng.permutation(nW)
    res = {"step": [], "step_null": [], "window": [], "window_null": [], "slope": [], "sd_ratio": []}
    for j in range(n):
        l, y = L[:, :, j], Y[:, :, j]
        if l.std() < 1e-12 or y.std() < 1e-12:
            continue
        c = lambda u, v: float(np.corrcoef(u.ravel(), v.ravel())[0, 1])
        res["step"].append(c(l, y)); res["step_null"].append(c(l, y[perm]))
        res["window"].append(c(l.mean(1), y.mean(1))); res["window_null"].append(c(l.mean(1), y[perm].mean(1)))
        res["slope"].append(float(np.polyfit(l.ravel(), y.ravel(), 1)[0])); res["sd_ratio"].append(float(l.std() / y.std()))
    for k, v in res.items():
        v = np.array(v)
        print(f"{k:12s} median {np.median(v):+.3f}   IQR {np.percentile(v, 25):+.3f} .. {np.percentile(v, 75):+.3f}   "
              f"fraction > 0: {np.mean(v > 0):.2f}   (n {len(v)})")
    z = np.arange(n) % a.depth
    print("per slab, median step correlation (L ~ Y) vs null:")
    for s in range(a.depth):
        js = [j for j in np.flatnonzero(z == s) if L[:, :, j].std() > 1e-12 and Y[:, :, j].std() > 1e-12]
        if not js:
            continue
        cs = [np.corrcoef(L[:, :, j].ravel(), Y[:, :, j].ravel())[0, 1] for j in js]
        cn = [np.corrcoef(L[:, :, j].ravel(), Y[perm][:, :, j].ravel())[0, 1] for j in js]
        print(f"  slab {s}: {np.median(cs):+.3f}  (null {np.median(cn):+.3f}, n {len(js)})")


if __name__ == "__main__":
    main()
