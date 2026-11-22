"""Does the drift freeze activity per state? Within-state variability vs that state's outcomes.

From move_log activity windows (spike counts per neuron over the move before each decision; one window at the start,
one late) joined with <run>(0)_moves.csv on the timestep. Per candidate type (16 states) and population:
  - variability: mean over neurons of the within-type variance of the count, divided by the mean count (Fano-like,
    so a pure rate change does not count as freezing);
  - outcome of the type in the late window: correct fraction (commit iff good) and mean TD after the move.
Reported: per population, the change in log variability (late / early) per type, and its Spearman correlation over
types with the late outcome. Freeze prediction: negative (types that go well lose variability more).

Usage: python freeze_test.py <run_dir> [...]
"""
import csv, glob, sys
import numpy as np


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main(runs):
    for run in runs:
        name = run.rstrip("/").split("/")[-1]
        moves = {int(r["timestep"]): r for r in csv.DictReader(open(f"{run}/outputs/{name}(0)_moves.csv"))}
        files = sorted(glob.glob(f"{run}/outputs/{name}(0)_activity*.npz"), key=lambda f: (f.count("_activity_"), f))
        early, late = np.load(files[0]), np.load(files[-1])
        pops = [k for k in early.files if k not in ("ctype", "timestep")]
        # outcome per type in the late window
        ok = np.full(16, np.nan); td = np.full(16, np.nan)
        lt = late["ctype"]
        rows = [moves.get(int(t)) for t in late["timestep"]]
        okv = np.array([np.nan if r is None else float((int(float(r["action"])) == 0) == (float(r["good"]) > 0.5)) for r in rows])
        tdv = np.array([np.nan if r is None else float(r["td_after"]) for r in rows])
        for c in range(16):
            m = lt == c
            ok[c], td[c] = np.nanmean(okv[m]), np.nanmean(tdv[m])
        print(f"== {name}: late window matched {np.mean([r is not None for r in rows]):.2f}; per-type correct "
              f"{np.nanmin(ok):.2f}..{np.nanmax(ok):.2f}")
        for p in pops:
            dv = []
            for c in range(16):
                v = []
                for W in (early, late):
                    X = W[p][W["ctype"] == c].astype(float)
                    mu = X.mean(0); act = mu > 0.05
                    v.append(np.mean(X.var(0)[act] / mu[act]) if act.sum() >= 3 and len(X) >= 10 else np.nan)
                dv.append(np.log(v[1] / v[0]))
            dv = np.array(dv); g = np.isfinite(dv) & np.isfinite(ok)
            print(f"  {p:13s} log var change: mean {np.nanmean(dv):+.2f} | corr with late correct {spearman(dv[g], ok[g]):+.2f}, "
                  f"with late TD {spearman(dv[g], td[g]):+.2f} ({g.sum()} types)")


if __name__ == "__main__":
    main(sys.argv[1:])
