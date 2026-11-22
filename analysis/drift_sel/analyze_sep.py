"""Separation of candidate types vs. trial-to-trial noise, from outputs/<run>(0)_activity.npz (move_log "activity").

Per population (spike counts per neuron over one move):
  - sep: mean over type pairs of |mu_a - mu_b|^2 / (2 * within-type variance summed over neurons); > 1 means two
    types differ more than two presentations of the same type (a d'^2 per pair, pooled over neurons);
  - decode: nearest-centroid accuracy of the type from a single move (centroids from the other half of the
    presentations, both ways; chance 1/16): how reliably one move identifies the state;
  - active: the fraction of neurons with at least one spike in a move; count: mean spikes per neuron per move.

Usage: python analyze_sep.py <run_dir> [<run_dir> ...]
"""
import os
import sys

import numpy as np


def separation(X, ct):
    types = [t for t in np.unique(ct) if (ct == t).sum() >= 6]
    mu = np.array([X[ct == t].mean(0) for t in types])
    within = np.mean([X[ct == t].var(0).sum() for t in types])
    pairs = [np.sum((mu[i] - mu[j]) ** 2) for i in range(len(types)) for j in range(i + 1, len(types))]
    sep = np.mean(pairs) / (2 * within) if within > 0 else float("nan")
    hits, total = 0, 0
    for fit, test in ((0, 1), (1, 0)):
        cent, idx = [], []
        for t in types:
            ii = np.flatnonzero(ct == t)
            cent.append(X[ii[fit::2]].mean(0))
            idx.append(ii[test::2])
        cent = np.array(cent)
        for k, ii in enumerate(idx):
            d = ((X[ii][:, None, :] - cent[None]) ** 2).sum(-1)
            hits += int((d.argmin(1) == k).sum())
            total += len(ii)
    return sep, hits / total, float((X > 0).mean()), float(X.mean())


def main(runs):
    print(f"{'run':14s} {'population':14s} {'sep':>7s} {'decode':>7s} {'active':>7s} {'count':>6s}")
    for run in runs:
        name = os.path.basename(os.path.normpath(run))
        d = np.load(f"{run}/outputs/{name}(0)_activity.npz")
        ct = d["ctype"]
        for k in d.files:
            if k in ("ctype", "timestep"):
                continue
            s, acc, act, cnt = separation(d[k].astype(np.float64), ct)
            print(f"{name:14s} {k:14s} {s:7.3f} {acc:7.3f} {act:7.3f} {cnt:6.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
