"""Ceiling: how much of each receiver's credit could any link-based predictor read from the network's activity?

From a credit recording (CreditY of the recorded population, spikes of every recorded source population): per window
(states x windows), features = the window-mean presynaptic traces of all recorded source neurons (all hidden + input
spikes onto the recorded population); target = each receiver's window-mean credit Y. Cross-validated ridge (2-fold by
alternating windows), correlation of prediction with the held-out target; median over receivers. Null: the same with
the targets shuffled across windows.

Usage: python credit_ceiling.py DIR [DIR ...]
"""
import sys
from pathlib import Path
import numpy as np


def ridge_cv(X, Y, lam):
    n = len(X); fold = np.arange(n) % 2 == 0; P = np.empty_like(Y)
    for tr in (fold, ~fold):
        Xt = X[tr]; mu = Xt.mean(0); sd = Xt.std(0) + 1e-9
        A = (Xt - mu) / sd; ym = Y[tr].mean(0)
        w = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ (Y[tr] - ym))
        P[~tr] = ((X[~tr] - mu) / sd) @ w + ym
    return np.array([np.corrcoef(P[:, i], Y[:, i])[0, 1] if Y[:, i].std() > 0 else np.nan for i in range(Y.shape[1])])


def main():
    for d in sys.argv[1:]:
        d = Path(d); name = d.name; out = d / "outputs" / f"{name}(0)"
        times = [tuple(map(int, l.split())) for l in open(f"{out}_drift_record_times.txt")]
        recs = [np.load(f"{out}_drift_record_{k}.npz", allow_pickle=True) for k, _ in times]
        alpha = float(recs[0]["alpha"])
        Y = np.concatenate([r["state_CreditY"][:, 1:] for r in recs]).mean(1)            # (windows, receivers)
        feats = []
        for k in range(int(recs[0]["n_sources"])):
            z = np.concatenate([r[f"spikes_{k}"][:, 1:] for r in recs]).astype(float)
            zf = np.empty_like(z); prev = np.zeros((z.shape[0], z.shape[2]))
            for t in range(z.shape[1]):
                prev = alpha * prev + z[:, t]; zf[:, t] = prev
            feats.append(zf.mean(1))
        X = np.concatenate(feats, 1); X = X[:, X.std(0) > 1e-9]
        rng = np.random.default_rng(0)
        print(f"######## {name}: {X.shape[0]} windows, {X.shape[1]} source neurons, {Y.shape[1]} receivers")
        for lam in (1.0, 10.0, 100.0):
            c = ridge_cv(X, Y, lam)
            cn = ridge_cv(X, Y[rng.permutation(len(Y))], lam)
            print(f"  ridge {lam:6.1f}: credit predicted from activity, median corr {np.nanmedian(c):+.3f} "
                  f"(IQR {np.nanpercentile(c, 25):+.3f}..{np.nanpercentile(c, 75):+.3f}); shuffled null {np.nanmedian(cn):+.3f}")


if __name__ == "__main__":
    main()
