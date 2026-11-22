"""State specificity of the drift eligibility from aggregate probes (move_log "aggregate": true).

Per connection (and all together), from the per-type sums over odd / even presentations:
  - same / diff: cosine between the two halves' mean eligibility of the same type vs. of different types, after
    removing the mean over types (the state-independent part). same >> diff: a reproducible per-state direction;
  - overlap: the noise-free cosine between different types' expected eligibility (grand mean removed), from
    split-half cross products h0_a . h1_b / sqrt(h0_a . h1_a  h0_b . h1_b) (types with a positive signal only);
  - common: |mean over types|^2 / mean |type mean|^2 (how much of a state's expected drift is shared by all states);
  - snr: |expected type eligibility - grand mean|^2 (split-half product) / single-move noise variance: 1/snr
    presentations of a state are needed before its own direction equals one move's noise.

Usage: python analyze_agg.py <run_dir> [<run_dir> ...] [--no-all]
"""
import glob
import os
import sys

import numpy as np


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def stats(S, C, SQ, min_n=8):
    ok = np.flatnonzero((C >= min_n / 2).all(axis=1))
    halves = S[ok] / C[ok][:, :, None]                  # types x 2 x synapses
    means = S[ok].sum(1) / C[ok].sum(1)[:, None]        # types x synapses
    grand = means.mean(0)
    h = halves - grand
    same = np.array([cos(h[i, 0], h[i, 1]) for i in range(len(ok))])
    diff = [cos(h[i, 0], h[j, 1]) for i in range(len(ok)) for j in range(len(ok)) if i != j]
    # unbiased (split-half) products: E[h0_a . h1_b] = mu_a . mu_b, the noise of the two halves being independent
    P = np.einsum("is,js->ij", h[:, 0], h[:, 1])
    P = 0.5 * (P + P.T)
    sig = np.diag(P).copy()
    good = sig > 0
    over = [P[i, j] / np.sqrt(sig[i] * sig[j]) for i in range(len(ok)) for j in range(i + 1, len(ok))
            if good[i] and good[j]]
    n = C[ok].sum(1)
    noise = SQ[ok] / n - np.sum(means ** 2, axis=1)        # per-move variance (total over synapses)
    snr = float(np.mean(sig / np.maximum(noise, 1e-30)))
    common = float(grand @ grand / np.mean(np.sum(means ** 2, axis=1)))
    return len(ok), int(n.mean()), np.mean(same), np.mean(diff), (np.mean(over) if over else float('nan')), common, snr


def main(runs):
    print(f"{'run':16s} {'connection':22s} {'types':>5s} {'n/type':>6s} {'same':>6s} {'diff':>6s} {'overlap':>7s} "
          f"{'common':>7s} {'snr':>8s}")
    for run in runs:
        name = os.path.basename(os.path.normpath(run))
        for f in sorted(glob.glob(f"{run}/outputs/{name}(0)_moves/agg_*.npz")):
            d = np.load(f)
            keys = sorted({k.rsplit("|", 1)[0] for k in d.files})
            parts = {k: (d[f"{k}|S"], d[f"{k}|C"], d[f"{k}|SQ"]) for k in keys}
            extra = []
            if not NO_ALL:   # the combined matrix: memory-heavy for large networks (--no-all skips it)
                S = np.concatenate([p[0] for p in parts.values()], axis=2)
                extra = [("all", (S, parts[keys[0]][1], sum(p[2] for p in parts.values())))]
            for k, (s, c, sq) in list(parts.items()) + extra:
                if True:
                    r = stats(s, c, sq)
                    print(f"{name:16s} {k:22s} {r[0]:5d} {r[1]:6d} {r[2]:6.3f} {r[3]:6.3f} {r[4]:7.3f} {r[5]:7.3f} "
                          f"{r[6]:8.4f}")


NO_ALL = False

if __name__ == "__main__":
    NO_ALL = "--no-all" in sys.argv
    main([a for a in sys.argv[1:] if a != "--no-all"])
