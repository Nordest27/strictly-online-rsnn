"""Per-block mean episode score and reward rate (reward per env step) for runs in runs/.

usage: python summarise.py [RUN ...] [--block=250]
Runs named <variant>_s<seed> are also averaged per variant (mean ± s.d. of the reward rate).
"""
import csv, sys, glob, os, re
from collections import defaultdict
import numpy as np

X = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith("--block=")]
B = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--block=")), 250))


def load(n):
    f = glob.glob(f"{X}/runs/{n}/outputs/*.csv")
    rows = list(csv.DictReader(open(f[0]))) if f else []
    rows = [r for r in rows if r.get("ep_steps") not in (None, "")]  # drop a truncated last row
    if not rows:
        return None
    sc = np.array([float(r["score"]) for r in rows])
    st = np.array([float(r["ep_steps"]) for r in rows])
    blocks = [(i, sc[i:i+B].mean(), 30 * sc[i:i+B].sum() / st[i:i+B].sum()) for i in range(0, len(sc), B)]
    return blocks


names = args or sorted(os.listdir(f"{X}/runs"))
groups = defaultdict(list)
print(f"== per run  [score/rate per {B} episodes]")
for n in names:
    b = load(n)
    if b is None:
        print(f"{n:16s} no data"); continue
    print(f"{n:16s} eps={b[-1][0] + 1:5d}+ " + "  ".join(f"{s:+5.2f}/{r:+.3f}" for _, s, r in b))
    groups[re.sub(r"_s\d+$", "", n)].append([r for _, _, r in b])

if any(len(v) > 1 for v in groups.values()):
    print(f"\n== per variant  [reward rate mean±s.d. over seeds, per {B} episodes]")
    for g, runs in groups.items():
        L = min(len(r) for r in runs)
        a = np.array([r[:L] for r in runs])
        print(f"{g:16s} n={len(runs)} " + "  ".join(f"{m:+.3f}±{s:.3f}" for m, s in zip(a.mean(0), a.std(0))))
