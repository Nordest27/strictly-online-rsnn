"""Contingency runs: commit rate on types 0-7 (lo) and 8-15 (hi) per sixth of the run.
Arm A rewards (pos) / punishes (neg) commit on lo and skip on hi; arm B the reverse."""
import csv, glob, os, sys
import numpy as np
for r in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "*_s*")):
    p = f"{r}/outputs/{r}(0)_moves.csv"
    if not os.path.exists(p):
        continue
    rows = list(csv.DictReader(open(p)))
    if len(rows) < 1200:
        print(r, len(rows), "moves"); continue
    c = np.array([int(x["ctype"]) for x in rows]); a = np.array([int(x["action"]) for x in rows])
    B = len(rows) // 6
    out = [f"{(a[s:s + B][c[s:s + B] < 8] == 0).mean():.2f}/{(a[s:s + B][c[s:s + B] >= 8] == 0).mean():.2f}"
           for s in range(0, len(rows) - B + 1, B)]
    print(f"{r:9s} {len(rows):6d} moves  commit lo/hi: " + " ".join(out))
