"""Is the action visible in L1 and the fields, beyond the state?

From each run's activity windows (move_log "activity_at": spike counts per neuron over the move shown before each
decision) joined with outputs/<run>(0)_moves.csv on the timestep (the action taken at that decision):
  - state decoding: nearest-centroid accuracy of the candidate type from one move (chance 1/16; analyze_sep);
  - action decoding within state: for every candidate type with at least `min_n` moves of each action, a
    nearest-centroid classifier of commit vs skip from that type's moves (centroids from one half, tested on the
    other, both ways, balanced by taking the same number of each action); the mean accuracy over those types
    (chance 0.5). Because the state is fixed inside a type, this is action information beyond the state: the part of
    the activity that differs between trials of one state that ended in commit and trials that ended in skip;
  - commit share per window (how decided the policy is) and the number of types that entered the average.

Usage: python analyze_action.py <run> [<run> ...]   (run directories in this folder, or paths relative to it)
"""
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import analyze_sep  # noqa: E402


def action_decoding(X, ct, act, min_n=10, seed=0):
    rng = np.random.default_rng(seed)
    accs = []
    for t in np.unique(ct):
        i0 = np.flatnonzero((ct == t) & (act == 0))
        i1 = np.flatnonzero((ct == t) & (act == 1))
        n = min(len(i0), len(i1))
        if n < min_n:
            continue
        i0, i1 = rng.permutation(i0)[:n], rng.permutation(i1)[:n]
        hits = total = 0
        for fit, test in ((0, 1), (1, 0)):
            c0, c1 = X[i0[fit::2]].mean(0), X[i1[fit::2]].mean(0)
            for idx, label in ((i0[test::2], 0), (i1[test::2], 1)):
                d0 = ((X[idx] - c0) ** 2).sum(1)
                d1 = ((X[idx] - c1) ** 2).sum(1)
                hits += int(((d1 < d0).astype(int) == label).sum())
                total += len(idx)
        accs.append(hits / total)
    return (float(np.mean(accs)) if accs else float("nan")), len(accs)


def main(runs):
    for run in runs:
        d = os.path.join(HERE, run)
        run = os.path.basename(os.path.normpath(run))
        moves = {int(r["timestep"]): int(r["action"]) for r in csv.DictReader(open(f"{d}/outputs/{run}(0)_moves.csv"))}
        files = sorted(glob.glob(f"{d}/outputs/{run}(0)_activity*.npz"),
                       key=lambda f: int(os.path.basename(f).split("_activity")[1][1:-4] or 0))
        for f in files:
            a = np.load(f)
            keep = np.array([int(t) in moves for t in a["timestep"]])
            ct = a["ctype"][keep]
            act = np.array([moves[int(t)] for t in a["timestep"][keep]])
            start = os.path.basename(f).split("_activity")[1][1:-4] or "0"
            print(f"== {run} window from t={start}: {keep.sum()} moves, commit share {np.mean(act == 0):.2f}")
            for k in [x for x in a.files if x not in ("ctype", "timestep")]:
                X = a[k][keep].astype(np.float64)
                _, dec, _, cnt = analyze_sep.separation(X, ct)
                acc, nt = action_decoding(X, ct, act)
                print(f"  {k:13s} state decode {dec:.2f}   action-within-state decode {acc:.2f} ({nt} types)"
                      f"   spikes/move {cnt:.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
