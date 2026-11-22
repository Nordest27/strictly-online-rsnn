"""State distinction by the eligibility derivative: how similar is the expected drift eligibility of two candidate
types, as a function of how many of the 4 features they share?

From aggregate probes (move_log "aggregate": true). Per connection and probe, the noise-free cosine between two
types' expected eligibility, from split-half cross products (h0_a . h1_b / sqrt(h0_a . h1_a  h0_b . h1_b)), averaged
over the type pairs with 0 .. 3 shared feature values (4 = the same type, 1 by construction). Two versions:
raw (the part shared by all states included) and centred (the mean over types removed).
A slope with the shared features = graded generalisation; a constant offset in "raw" = the part common to all states.

Usage: python analyze_overlap.py <run_dir> [connection substring ...]
"""
import glob
import os
import sys

import numpy as np


def table(S, C, centre):
    ok = np.flatnonzero((C >= 4).all(axis=1))
    h = (S[ok] / C[ok][:, :, None]).astype(np.float64)
    if centre:
        h = h - (S[ok].sum(1) / C[ok].sum(1)[:, None]).mean(0)
    P = h[:, 0] @ h[:, 1].T
    P = 0.5 * (P + P.T)
    sig = np.diag(P)
    out = {k: [] for k in range(4)}
    for i in range(len(ok)):
        for j in range(i + 1, len(ok)):
            if sig[i] > 0 and sig[j] > 0:
                shared = 4 - bin(int(ok[i]) ^ int(ok[j])).count("1")
                out[shared].append(P[i, j] / np.sqrt(sig[i] * sig[j]))
    return [np.mean(out[k]) if out[k] else float("nan") for k in range(4)]


def main(run, subs):
    name = os.path.basename(os.path.normpath(run))
    files = sorted(glob.glob(f"{run}/outputs/{name}(0)_moves/agg_*.npz"),
                   key=lambda f: int(os.path.basename(f)[4:-4]))
    print(f"{name}: cosine of the expected drift eligibility between types sharing 0 / 1 / 2 / 3 features")
    for f in files:
        d = np.load(f)
        keys = sorted({k.rsplit("|", 1)[0] for k in d.files})
        for k in keys:
            if subs and not any(s == k for s in subs):
                continue
            S, C = d[f"{k}|S"], d[f"{k}|C"]
            raw, cen = table(S, C, False), table(S, C, True)
            print(f"  t={os.path.basename(f)[4:-4]:>8s} {k:14s} raw " + " ".join(f"{x:+.2f}" for x in raw)
                  + "   centred " + " ".join(f"{x:+.2f}" for x in cen))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
