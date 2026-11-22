"""Frozen probes with 0, 1 and 11 random long-range inputs per neuron (p_global, rates calibrated in each).

Per population (spike counts per move, from outputs/<run>(0)_activity.npz):
  - rate (Hz), state decoding from one move and separation (analyze_sep);
  - noise correlation: mean pairwise correlation of the neurons' counts across presentations of the same candidate
    type (type means removed), i.e. how much neurons vary together on repeats of one state (shared fluctuations:
    fewer independent candidates);
  - signal correlation: mean pairwise correlation of the neurons' type means (how alike their tuning is).
Per stage, the per-state drift (analyze_agg.stats): same / diff and snr, for L1 (input and recurrent E synapses) and
for L1_E -> policy_field.

Usage: python analyze_pg.py pg0 pg1 pg11
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import analyze_agg  # noqa: E402
import analyze_sep  # noqa: E402

STEPS_PER_MOVE = 56   # timesteps per decision in these runs (counts / move -> Hz)


def mean_offdiag_corr(x):
    keep = x.std(0) > 0
    x = x[:, keep]
    if x.shape[1] < 2:
        return float("nan")
    c = np.corrcoef(x, rowvar=False)
    return float(c[~np.eye(len(c), dtype=bool)].mean())


def main(runs):
    for run in runs:
        d = os.path.join(HERE, run)
        act = np.load(f"{d}/outputs/{run}(0)_activity.npz")
        ct = act["ctype"]
        types = [t for t in range(16) if (ct == t).sum() >= 10]
        print(f"== {run}")
        for k in ("L1_E", "L1_I", "policy_field", "value_field"):
            if k not in act.files:
                continue
            X = act[k].astype(np.float64)
            sep, dec, active, cnt = analyze_sep.separation(X, ct)
            resid = np.concatenate([X[ct == t] - X[ct == t].mean(0) for t in types])
            means = np.array([X[ct == t].mean(0) for t in types])
            print(f"  {k:13s} {1000 * cnt / STEPS_PER_MOVE:5.1f} Hz  decode {dec:.2f}  sep {sep:.2f}  active {active:.2f}"
                  f"  noise corr {mean_offdiag_corr(resid):+.3f}  signal corr {mean_offdiag_corr(means):+.3f}")
        agg = np.load(glob.glob(f"{d}/outputs/{run}(0)_moves/agg_*.npz")[0])
        for label, keys in (("L1", ["Pop0->L1_E", "L1_E->L1_E"]), ("fields", ["L1_E->policy_field"])):
            keys = [k for k in keys if f"{k}|S" in agg.files]
            S = np.concatenate([agg[f"{k}|S"] for k in keys], axis=2).astype(np.float64)
            SQ = sum(agg[f"{k}|SQ"].astype(np.float64) for k in keys)
            _, n, same, diff, _, common, snr = analyze_agg.stats(S, agg[f"{keys[0]}|C"], SQ)
            print(f"  drift {label:7s} same {same:+.3f} diff {diff:+.3f} common {common:.3f} snr {snr:.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
