"""Toy (open-loop single-neuron replay) vs network drift check on the same synapses.
Usage: python toy_compare.py L  (uses rec_L<L> and dcw200_L<L>)"""
import glob, sys, time
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "/home/user/repos/ml_genn/examples/eprop")
import drift_toy as dt

L = sys.argv[1]
rollouts, fd_rollouts = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (24, 8)
rec = dt.Recording(glob.glob(f"rec_L{L}/outputs/*drift_record.npz")[0])
net = np.load(glob.glob(f"dcw200_L{L}/outputs/*drift_check.npz")[0])
syn, h = net["synapses"], float(net["h"])
tg = next(gr for gr in rec.groups if gr["name"] == rec.target_group)
posts = sorted({int(tg["post"][s]) for s in syn})
print(f"L{L}: {len(syn)} synapses on {len(posts)} neurons, {rec.W} windows x {rec.T1 - 1} steps, h {h:.4g}, "
      f"lambda {rec.lam:.4f}, beta1 {rec.beta1}")
print(f"  replay with recorded noise, max |V error|: {max(dt.validate(rec, j) for j in posts):.2e}")

t0 = time.time()
toy = dt.drift(rec, syn, h, rollouts=rollouts, fd_rollouts=fd_rollouts)
print(f"  toy check: {time.time() - t0:.0f} s ({rollouts} own-noise rollouts and {fd_rollouts} derivative pairs "
      f"per window)")


def rel(a, b):
    r = np.corrcoef(np.ravel(a), np.ravel(b))[0, 1]
    return 2 * r / (1 + r)


def line(name, x, y):
    c = np.corrcoef(x, y)[0, 1]
    return f"{name}: corr {c:.2f} (Spearman {spearmanr(x, y)[0]:.2f})"


print(f"  toy repeatability (rule-averaged): score {rel(toy['score_f_a'], toy['score_f_b']):.2f}, "
      f"derivative {rel(toy['deriv_f_a'], toy['deriv_f_b']):.2f}")
print("  toy, rule-averaged, score vs derivative   " + line("", toy["deriv_f"], toy["score_f"]))
print("  toy, rule-averaged, score vs eligibility  " + line("", toy["elig_f"], toy["score_f"]))
print("  toy vs network, rule-averaged score       " + line("", net["score_f"], toy["score_f"]))
print("  toy vs network, rule-averaged derivative  " + line("", net["deriv_f"], toy["deriv_f"]))
ratio = np.median(np.abs(toy["score_f"]) / (np.abs(net["score_f"]) + 1e-12))
print(f"  median |toy score| / |network score|: {ratio:.2f}")
np.savez(f"toy_compare_L{L}.npz", **toy)
