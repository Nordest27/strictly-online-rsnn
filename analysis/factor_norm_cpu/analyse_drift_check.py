"""Analyse (partial) drift-check results: repeatability of each side from split halves, raw and
noise-corrected correlations, per step and after the rule's own averaging (trace + momentum)."""
import glob, sys
import numpy as np
from scipy.stats import spearmanr


def rel(a, b):
    """Split-half repeatability of the full-sample mean (Spearman-Brown)."""
    r = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    return 2 * r / (1 + r)


for run in sys.argv[1:]:
    files = glob.glob(f"{run}/outputs/*drift_check.npz")
    if not files:
        print(f"{run}: no results yet"); continue
    d = np.load(files[0]); n = int(d["n_done"])
    print(f"== {run}: stage {d['stage']}, {n}/{len(d['synapses'])} synapses with derivatives, gate {d['gate']}, "
          f"lambda {float(d['trace_lambda']):.4f}, beta1 {float(d['beta1'])}")
    rs = rel(d["score_a"], d["score_b"]); rsf = rel(d["score_fa"], d["score_fb"])
    print(f"   repeatability, score: per step {rs:.2f}, rule-averaged {rsf:.2f}")
    if n == 0:
        continue
    S, D, E = d["score"][:, :n], d["deriv"], d["elig"][:, :n]
    Sf, Df, Ef = d["score_f"][:n], d["deriv_f"], d["elig_f"][:n]
    rd = rel(d["deriv_a"], d["deriv_b"]); rdf = rel(d["deriv_fa"], d["deriv_fb"])
    rsn = rel(d["score_a"][:, :n], d["score_b"][:, :n]); rsfn = rel(d["score_fa"][:n], d["score_fb"][:n])
    c = np.corrcoef(D.ravel(), S.ravel())[0, 1]; cf = np.corrcoef(Df, Sf)[0, 1]
    corr = lambda r, a, b: r / np.sqrt(max(a, 1e-9) * max(b, 1e-9)) if a > 0 and b > 0 else np.nan
    print(f"   repeatability, derivative: per step {rd:.2f}, rule-averaged {rdf:.2f}")
    print(f"   per step:      corr {c:.2f} (Spearman {spearmanr(D.ravel(), S.ravel())[0]:.2f}) -> noise-corrected "
          f"{corr(c, rsn, rd):.2f} | vs E[e] corr {np.corrcoef(E.ravel(), S.ravel())[0, 1]:.2f}")
    sl = np.dot(Df - Df.mean(), Sf - Sf.mean()) / np.dot(Df - Df.mean(), Df - Df.mean())
    print(f"   rule-averaged: corr {cf:.2f} (Spearman {spearmanr(Df, Sf)[0]:.2f}, n={n}) -> noise-corrected "
          f"{corr(cf, rsfn, rdf):.2f}, slope {sl:.2f} | vs E[e] corr {np.corrcoef(Ef, Sf)[0, 1]:.2f}")
