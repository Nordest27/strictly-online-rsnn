"""Does the drift raise or lower the drive of the neurons that fire most in a state?

From a frozen probe with move_log "aggregate" (with synapse indices) and "activity": per candidate type t and
postsynaptic neuron i,
  D_i(t) = sum over i's incoming excitatory synapses of the mean drift eligibility in type t (TD-free: what a
           positive TD error after t would add to the weights, up to the rule's sign and learning rate),
  r_i(t) = i's mean spike count per move in type t.
Sign: the code's (DeltaG += delta * trace; Adam: w -= lr * DeltaG), so the drive change after a positive TD error is
-D; the printed values are that change.

Reported per connection:
  - grand: the mean D (sign of the convention) and its split-half reliability across (i, t);
  - pooled r(D, rate) over all (i, t); within-neuron r (each neuron centred over types: in the states where a
    neuron fires more, does its drive gain more or less?); within-type r (each type centred over neurons: in a state,
    do the most active neurons gain more or less than the rest?);
  - D by rate decile (within-type centred rates): the shape of the relation (a hump: largest gains at mid rates).

Usage: python analyze_postmap.py <run_dir> [population]
"""
import glob
import os
import sys

import numpy as np


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def main(run, pop="L1_E"):
    name = os.path.basename(os.path.normpath(run))
    agg = np.load(glob.glob(f"{run}/outputs/{name}(0)_moves/agg_*.npz")[0])
    act = np.load(f"{run}/outputs/{name}(0)_activity.npz")
    ct, R = act["ctype"], act[pop].astype(np.float64)
    types = [t for t in range(16) if (ct == t).sum() >= 10]
    rate = np.array([R[ct == t].mean(0) for t in types])                  # types x neurons
    n_post = rate.shape[1]
    keys = sorted({k.rsplit("|", 1)[0] for k in agg.files if k.endswith(f"->{pop}|S")})
    keys = [k for k in keys if not k.split("->")[0].endswith("_I")]        # excitatory sources
    print(f"{name}: {pop}, {len(types)} types, {n_post} neurons; excitatory inputs {keys}")
    total = np.zeros((2, len(types), n_post))
    for k in keys + ["all excitatory"]:
        if k == "all excitatory":
            D2 = total
        else:
            S, C, post = agg[f"{k}|S"].astype(np.float64), agg[f"{k}|C"], agg[f"{k}|post"]
            D2 = np.zeros((2, len(types), n_post))
            for h in range(2):
                for a, t in enumerate(types):
                    np.add.at(D2[h, a], post, S[t, h] / max(C[t, h], 1))
            total += D2
        D = D2.mean(0)
        rel = corr(D2[0].ravel(), D2[1].ravel())
        g = D.mean()
        s = -1.0   # the code's sign: DeltaG += delta * trace and Adam does w -= lr * DeltaG, so a positive TD error
                   # changes the weights by -lr * delta * eligibility: Dn = the drive change after a positive TD error
        Dn = s * D
        wn = corr((Dn - Dn.mean(0)).ravel(), (rate - rate.mean(0)).ravel())        # within neuron (over types)
        wt = corr((Dn - Dn.mean(1, keepdims=True)).ravel(), (rate - rate.mean(1, keepdims=True)).ravel())
        rc = (rate - rate.mean(1, keepdims=True)).ravel()
        dc = (Dn - Dn.mean(1, keepdims=True)).ravel()
        edges = np.quantile(rc, np.linspace(0, 1, 11))
        bins = [dc[(rc >= edges[q]) & (rc <= edges[q + 1])].mean() / (np.abs(Dn).mean()) for q in range(10)]
        print(f"  {k:22s} grand {g:+.3e}  split-half r {rel:+.3f}  pooled r {corr(Dn.ravel(), rate.ravel()):+.3f}  "
              f"within-neuron r {wn:+.3f}  within-type r {wt:+.3f}")
        print(f"  {'':22s} D (centred, / mean|D|) by within-type rate decile: " + " ".join(f"{b:+.2f}" for b in bins))
        # the most active neurons of each state: does a positive TD raise or lower their drive (absolute sign)?
        top = rate >= np.quantile(rate, 0.9, axis=1, keepdims=True)
        print(f"  {'':22s} top-10% neurons per state: mean D {Dn[top].mean() / np.abs(Dn).mean():+.2f} "
              f"(others {Dn[~top].mean() / np.abs(Dn).mean():+.2f}); fraction with D < 0: {np.mean(Dn[top] < 0):.2f} "
              f"(others {np.mean(Dn[~top] < 0):.2f})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "L1_E")
