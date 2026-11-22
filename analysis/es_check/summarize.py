"""Scores, rates and drift/G alignment of the ES-check runs (headless selection)."""
import csv, glob, os, subprocess, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_parts import cos


def score(d):
    f = f"{d}/outputs/{d}(0).csv"
    if not os.path.exists(f):
        return None
    s = np.array([float(x["score"]) for x in csv.DictReader(open(f))])
    q = max(1, len(s) // 5)
    return s, [s[a:a + q].mean() for a in range(0, 5 * q, q)]


def rates(d):
    f = f"{d}/outputs/{d}(0)_neurons.csv"
    if not os.path.exists(f):
        return {}
    rows = list(csv.DictReader(open(f)))
    if not rows:
        return {}
    r = rows[-1]
    return {k: float(r[f"{k}_hz"]) for k in ("L1_E", "policy_field", "value_field") if f"{k}_hz" in r}


def alignment(d):
    files = sorted(glob.glob(f"{d}/outputs/{d}(0)_parts/*.npz"), key=lambda f: int(os.path.basename(f)[:-4]))
    if len(files) < 4:
        return None
    dumps = [np.load(f) for f in files]
    keys = sorted({k.rsplit("|", 1)[0] for k in dumps[0].files})
    out = []
    for k in keys:
        P = np.array([x[f"{k}|P"] for x in dumps], dtype=np.float64); G = np.array([x[f"{k}|G"] for x in dumps], dtype=np.float64)
        o, e = slice(1, None, 2), slice(0, None, 2)
        cross = 0.5 * (cos(P[o].sum(0), G[e].sum(0)) + cos(P[e].sum(0), G[o].sum(0)))
        out.append((k, cross, cos(P[o].sum(0), P[e].sum(0)), cos(G[o].sum(0), G[e].sum(0))))
    return len(files), out


for d in sys.argv[1:]:
    sc = score(d)
    if sc is None:
        print(f"{d:16s} not started"); continue
    s, fifths = sc
    rt = rates(d)
    print(f"{d:16s} eps {len(s):6d} fifths {' '.join(f'{x:6.2f}' for x in fifths)}  last500 {s[-500:].mean():6.2f}  "
          + "  ".join(f"{k} {v:.0f}Hz" for k, v in rt.items()))
    al = alignment(d)
    if al:
        n, rows = al
        cr = np.array([r[1] for r in rows]); pp = np.array([r[2] for r in rows]); gg = np.array([r[3] for r in rows])
        print(f"{'':16s} {n} windows: cross cos mean {np.nanmean(cr):6.3f} (min {np.nanmin(cr):6.3f}, max {np.nanmax(cr):6.3f})  "
              f"P/P {np.nanmean(pp):5.2f}  G/G {np.nanmean(gg):5.2f}")
