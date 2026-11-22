"""Drift-as-selection checks on headless selection runs with "move_log" (per-decision rows + probe dumps).

Behaviour, from outputs/<run>(0)_moves.csv, per block of decisions and per candidate type (16 = 2^4 feature sets):
  - correct: the fraction of decisions with the right action (commit iff good under the rule in force);
  - consistency: max(P(commit), P(skip)) per type, averaged over types (0.5 = random, 1 = frozen choice);
  - margin: the motor pools' difference (pool0 - pool1) per type, its mean |.| and its spread across presentations;
  - freeze-reward: the correlation over types between consistency and correctness (are the frozen types the
    right ones?).
Around each switch of the required features: correctness and consistency of the types whose correct action flipped
vs. those that did not, in the blocks just before and after (unfreezing).

Drift eligibility, from outputs/<run>(0)_moves/<t>.npz (per move, AccPx: the drift trace increments without TD):
  - split-half: per type, the mean eligibility over half its presentations vs. the other half, after removing the
    probe's grand mean (the state-independent part): same-type cosine vs. different-type cosine. Same > different
    means the per-move estimate carries a state-specific direction;
  - common-mode share: |grand mean|^2 / mean |x|^2 per move;
  - applied drift: sum_k delta_k x_k with delta_k the TD after move k (raw and centred over the probe): its cosine
    with the grand mean (how much of what is applied is the state-independent push).

Usage: python analyze_moves.py <run_dir> [block_decisions]
"""
import csv
import glob
import os
import sys

import numpy as np


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def load_moves(run):
    name = os.path.basename(os.path.normpath(run))
    rows = list(csv.DictReader(open(f"{run}/outputs/{name}(0)_moves.csv")))
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    return name, cols


def behaviour(m, block):
    commit = m["action"] == 0
    correct = commit == (m["good"] == 1)
    margin = m["pool0"] - m["pool1"]
    n = len(commit)
    print(f"{'decisions':>17s} {'correct':>8s} {'consist':>8s} {'|margin|':>8s} {'spread':>7s} {'freeze~ok':>9s} "
          f"{'td':>7s}  rate")
    out = []
    for a in range(0, n - block + 1, block):
        s = slice(a, a + block)
        cons, ok, mm, sd = [], [], [], []
        for t in range(16):
            sel = m["ctype"][s] == t
            if sel.sum() < 4:
                continue
            p = commit[s][sel].mean()
            cons.append(max(p, 1 - p)); ok.append(correct[s][sel].mean())
            mm.append(abs(margin[s][sel].mean())); sd.append(margin[s][sel].std())
        r = np.corrcoef(cons, ok)[0, 1] if len(cons) > 2 and np.std(cons) > 0 and np.std(ok) > 0 else float("nan")
        out.append((a, correct[s].mean(), np.mean(cons), np.mean(mm), np.mean(sd), r, m["td_after"][s].mean()))
        print(f"{a:7d}-{a + block:<9d} {out[-1][1]:8.3f} {out[-1][2]:8.3f} {out[-1][3]:8.4f} {out[-1][4]:7.4f} "
              f"{r:9.3f} {out[-1][6]:7.3f}")
    return out


def switches(m, block):
    sw = m["switches"]
    idx = np.flatnonzero(np.diff(sw) > 0) + 1
    if not len(idx):
        return
    commit = m["action"] == 0
    correct = commit == (m["good"] == 1)
    print("switch   decision   flipped types: correct / consistency before -> after   unflipped: before -> after")
    for i in idx:
        before, after = slice(max(0, i - block), i), slice(i, min(len(sw), i + block))
        # flipped types: those whose 'good' label differs between the rule before and after (estimated from labels)
        stats = {}
        for t in range(16):
            gb = m["good"][before][m["ctype"][before] == t]
            ga = m["good"][after][m["ctype"][after] == t]
            if len(gb) < 3 or len(ga) < 3:
                continue
            stats[t] = gb.mean() != ga.mean()
        res = {}
        for flipped in (True, False):
            types = [t for t, f in stats.items() if f == flipped]
            vals = []
            for s in (before, after):
                sel = np.isin(m["ctype"][s], types)
                c = []
                for t in types:
                    q = commit[s][m["ctype"][s] == t]
                    if len(q):
                        c.append(max(q.mean(), 1 - q.mean()))
                vals.append((correct[s][sel].mean() if sel.any() else float("nan"), np.mean(c) if c else float("nan")))
            res[flipped] = vals
        f, u = res[True], res[False]
        print(f"{int(sw[i]):6d} {i:10d}   {f[0][0]:.2f}/{f[0][1]:.2f} -> {f[1][0]:.2f}/{f[1][1]:.2f}"
              f"                          {u[0][0]:.2f}/{u[0][1]:.2f} -> {u[1][0]:.2f}/{u[1][1]:.2f}")


def probes(run, name, m):
    files = sorted(glob.glob(f"{run}/outputs/{name}(0)_moves/*.npz"), key=lambda f: int(os.path.basename(f)[:-4]))
    if not files:
        return
    t_index = {int(t): i for i, t in enumerate(m["timestep"])}
    print(f"{'probe t':>9s} {'moves':>5s} {'conn':24s} {'same':>6s} {'diff':>6s} {'common':>7s} "
          f"{'drift~cm':>8s} {'cdrift~cm':>9s}")
    for f in files:
        d = np.load(f)
        ts = d["timestep"]
        rows = [t_index.get(int(t)) for t in ts]
        keep = [i for i, r in enumerate(rows) if r is not None]
        rows = [rows[i] for i in keep]
        ctype, td = m["ctype"][rows].astype(int), m["td_after"][rows]
        keys = [k for k in d.files if k != "timestep"]
        for k in keys + ["all"]:
            X = (np.concatenate([d[q] for q in keys], axis=1) if k == "all" else d[k])[keep].astype(np.float64)
            g = X.mean(0)
            common = float(g @ g / np.mean(np.sum(X * X, axis=1)))
            Xc = X - g
            halves = {}
            for t in np.unique(ctype):
                ii = np.flatnonzero(ctype == t)
                if len(ii) >= 4:
                    halves[t] = (Xc[ii[0::2]].mean(0), Xc[ii[1::2]].mean(0))
            ts_ = list(halves)
            same = [cos(*halves[t]) for t in ts_]
            diff = [cos(halves[a][0], halves[b][1]) for a in ts_ for b in ts_ if a != b]
            drift = (td[:, None] * X).sum(0)
            cdrift = ((td - td.mean())[:, None] * X).sum(0)
            if k == "all" or k.endswith("policy_field") or k.startswith("Pop0->L1_E"):
                print(f"{int(ts[0]):9d} {len(rows):5d} {k:24s} {np.mean(same) if same else float('nan'):6.3f} "
                      f"{np.mean(diff) if diff else float('nan'):6.3f} {common:7.3f} {cos(drift, g):8.3f} "
                      f"{cos(cdrift, g):9.3f}")


def main(run, block=2000):
    name, m = load_moves(run)
    print(f"== {name}: {len(m['action'])} decisions")
    behaviour(m, block)
    switches(m, block // 2)
    probes(run, name, m)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2000)
