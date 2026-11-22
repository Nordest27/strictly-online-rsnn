"""ES check: alignment of the rule's drift (P) and gradient (G) parts.

Reads the AccP / AccGp dumps (outputs/<prefix>(0)_parts/<timestep>.npz, one per window) and prints, per connection:
  - the cosine between P and G in each window, and its mean over the run;
  - the cosine between P and G summed over blocks of windows (longer averaging, less noise);
  - the cross cosine: P summed over the odd windows against G over the even ones (and the reverse, averaged).
    Within a window P and G share the factor pre * xi * delta, so their noise is correlated (and anti-correlated
    when strong input coincides with psi > psibar); across windows the noise is independent, so the cross cosine
    measures only the alignment of their averages;
  - |P| / |G|.
G averaged over time estimates the gradient of the noise-smoothed task objective, so cos(P, G) > 0 means the drift
does not work against it.

Usage: python analyze_parts.py <run_dir> [block]
"""
import glob
import os
import sys

import numpy as np


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def main(run, block=10):
    name = os.path.basename(os.path.normpath(run))
    files = sorted(glob.glob(f"{run}/outputs/{name}(0)_parts/*.npz"), key=lambda f: int(os.path.basename(f)[:-4]))
    if not files:
        sys.exit(f"no dumps in {run}")
    dumps = [np.load(f) for f in files]
    keys = sorted({k.rsplit("|", 1)[0] for k in dumps[0].files})
    print(f"{name}: {len(files)} windows, timesteps {os.path.basename(files[0])[:-4]} .. {os.path.basename(files[-1])[:-4]}")
    print(f"{'connection':28s} {'cos/window':>10s} {'cos/block':>10s} {'cos/run':>8s} {'cross':>7s} {'P/P':>6s} {'G/G':>6s} "
          f"{'|P|/|G|':>8s}  cos/window by fifth")
    for k in keys:
        P = np.array([d[f"{k}|P"] for d in dumps], dtype=np.float64)
        G = np.array([d[f"{k}|G"] for d in dumps], dtype=np.float64)
        per = np.array([cos(p, g) for p, g in zip(P, G)])
        nb = len(P) // block
        blocks = [cos(P[i * block:(i + 1) * block].sum(0), G[i * block:(i + 1) * block].sum(0)) for i in range(nb)]
        ratio = np.linalg.norm(P, axis=1).mean() / np.linalg.norm(G, axis=1).mean()
        fifths = [np.nanmean(c) for c in np.array_split(per, 5)]
        odd, even = slice(1, None, 2), slice(0, None, 2)
        cross = 0.5 * (cos(P[odd].sum(0), G[even].sum(0)) + cos(P[even].sum(0), G[odd].sum(0)))
        # split-half self-consistency: how reproducible each part's average direction is at all
        pp, gg = cos(P[odd].sum(0), P[even].sum(0)), cos(G[odd].sum(0), G[even].sum(0))
        print(f"{k:28s} {np.nanmean(per):10.3f} {np.nanmean(blocks) if blocks else float('nan'):10.3f} "
              f"{cos(P.sum(0), G.sum(0)):8.3f} {cross:7.3f} {pp:6.3f} {gg:6.3f} {ratio:8.2f}  "
              + " ".join(f"{x:6.3f}" for x in fifths))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10)
