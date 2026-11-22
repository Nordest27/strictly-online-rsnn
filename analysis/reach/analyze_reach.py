"""Reach runs: per block of moves, how well the chosen directions point at the target, per state.

From outputs/<run>(0)_moves.csv (move_log) and <run>(0).csv:
  - reach: fraction of episodes that reached the target; score: mean episode reward;
  - cos: mean cosine between the action and the direction to the target (0 = random, 1 = optimal);
  - per-state alignment: the mean cosine within each of the 16 state types (8 directions x 2 distances), averaged;
  - speed: mean |action| (after clipping); td: mean TD error after a move.

Usage: python analyze_reach.py <run_dir> [<run_dir> ...]
"""
import csv
import os
import sys

import numpy as np


def main(runs, blocks=8):
    for run in runs:
        name = os.path.basename(os.path.normpath(run))
        rows = list(csv.DictReader(open(f"{run}/outputs/{name}(0)_moves.csv")))
        cos = np.array([float(r["cos"]) for r in rows])
        sp = np.array([float(r["speed"]) for r in rows])
        ct = np.array([int(r["ctype"]) for r in rows])
        td = np.array([float(r["td_after"]) for r in rows])
        ep = [r for r in csv.DictReader(open(f"{run}/outputs/{name}(0).csv"))]
        score = np.array([float(r["score"]) for r in ep])
        steps = np.array([float(r["ep_steps"]) for r in ep])
        print(f"== {name}: {len(rows)} moves, {len(ep)} episodes")
        print(f"{'moves':>15s} {'cos':>6s} {'state-cos':>9s} {'speed':>6s} {'td':>7s}")
        B = max(1, len(rows) // blocks)
        for a in range(0, len(rows) - B + 1, B):
            s = slice(a, a + B)
            per = [np.nanmean(cos[s][ct[s] == t]) for t in range(16) if (ct[s] == t).sum() >= 5]
            print(f"{a:7d}-{a + B:<7d} {np.nanmean(cos[s]):6.3f} {np.nanmean(per):9.3f} {np.nanmean(sp[s]):6.3f} "
                  f"{np.nanmean(td[s]):7.3f}")
        E = max(1, len(ep) // blocks)
        print("episodes: score / reach by block: " + " ".join(
            f"{score[a:a + E].mean():.2f}/{np.mean(steps[a:a + E] < steps.max()):.2f}" for a in range(0, len(ep) - E + 1, E)))


if __name__ == "__main__":
    main(sys.argv[1:])
