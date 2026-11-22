"""5-layer readout run (experiment(4)): learning curve, per-layer dynamics, and per-layer decoding early vs late.

    python -I analyse.py DATA_DIR PREFIX OUT_DIR

Decoding: ridge regression (apple dy/dx: R^2) and ridge classification on one-hot targets (danger flags, heading,
next move: accuracy, chance in brackets), 2-fold split by episode parity (consecutive moves of one episode never
straddle train and test). Features: each population's activity (standardised), and the input frame as reference.
Next move = heading change to the next sample of the same episode (straight / left / right).
Effective dimension: participation ratio of the activity covariance (standardised).
"""
import sys, csv, json
import numpy as np


def ridge_fit(X, Y, lam):
    A = X.T @ X
    return np.linalg.solve(A + lam * np.trace(A) / len(A) * np.eye(len(A)), X.T @ Y)


def prep(X):
    X = X.astype(np.float64)
    keep = X.std(0) > 1e-8
    X = X[:, keep]
    return (X - X.mean(0)) / X.std(0)


def cv(X, y, ep, kind, lam=1e-1):
    """2-fold by episode parity; returns R^2 (kind 'reg') or accuracy (kind 'cls')."""
    if X.shape[1] == 0:
        return float("nan")
    X = np.concatenate([X, np.ones((len(X), 1))], 1)
    fold = ep % 2 == 0
    scores = []
    for tr in (fold, ~fold):
        te = ~tr
        if kind == "reg":
            w = ridge_fit(X[tr], y[tr], lam)
            p = X[te] @ w
            scores.append(1 - np.mean((y[te] - p) ** 2) / np.var(y[te]))
        else:
            classes = np.unique(y[tr])
            Y = (y[tr][:, None] == classes[None, :]).astype(float)
            w = ridge_fit(X[tr], Y, lam)
            p = classes[np.argmax(X[te] @ w, 1)]
            scores.append(np.mean(p == y[te]))
    return float(np.mean(scores))


def participation_ratio(X):
    ev = np.clip(np.linalg.eigvalsh(np.cov(X, rowvar=False)), 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def main():
    data, prefix, out = sys.argv[1], sys.argv[2], sys.argv[3]
    res = {}
    # --- learning curve ---
    rows = list(csv.DictReader(open(f"{data}/{prefix}.csv")))
    score = np.array([float(r["score"]) for r in rows])
    td = np.array([float(r["avg_abs_td_error"]) for r in rows])
    n = len(score)
    marks = [100, 250, 500, 1000, 2000, 2500, 5000, 10000, 15000, n]
    res["curve"] = {int(m): {"mean_score_last500": float(score[max(0, m - 500):m].mean()),
                             "max_score_so_far": float(score[:m].max())} for m in marks if m <= n}
    res["td_abs"] = {"first1000": float(td[:1000].mean()), "last1000": float(td[-1000:].mean())}
    print("== learning curve (mean of the last 500 episodes up to each point; max so far)")
    for m, v in res["curve"].items():
        print(f"  ep {m:6d}: mean {v['mean_score_last500']:6.2f}   max {v['max_score_so_far']:5.1f}")
    print(f"  |TD error|: first 1000 eps {res['td_abs']['first1000']:.4f}, last 1000 eps {res['td_abs']['last1000']:.4f}")
    # --- per-layer dynamics ---
    nrows = list(csv.DictReader(open(f"{data}/{prefix}_neurons.csv")))
    pops = ["L1_E", "L1_I", "L2_E", "L2_I", "L3_E", "L3_I", "L4_E", "L4_I", "L5_E", "L5_I", "policy_field", "value_field"]
    print("\n== per-population dynamics (first monitor row -> last)")
    res["dynamics"] = {}
    for p in pops:
        if f"{p}_hz" not in nrows[0]:
            continue
        g = lambda r, k: float(r.get(f"{p}_{k}", "nan") or "nan")
        a, b = nrows[0], nrows[-1]
        res["dynamics"][p] = {k: [g(a, k), g(b, k)] for k in ("hz", "dead", "mean_psi", "vloss", "deadzone")}
        d = res["dynamics"][p]
        print(f"  {p:12s} hz {d['hz'][0]:6.1f} -> {d['hz'][1]:6.1f}   dead {d['dead'][0]:.2f} -> {d['dead'][1]:.2f}   "
              f"psi {d['mean_psi'][0]:.3f} -> {d['mean_psi'][1]:.3f}   vloss {d['vloss'][0]:.3f} -> {d['vloss'][1]:.3f}")
    # --- decoding ---
    pr = np.load(f"{data}/{prefix}_probe.npz", allow_pickle=False)
    labels = [str(x) for x in pr["labels"]]; sizes = pr["sizes"]
    variables = [str(x) for x in pr["variables"]]; extras = [str(x) for x in pr["extra_names"]]
    offs = np.concatenate([[0], np.cumsum(sizes)])
    res["decode"] = {}
    for phase in ("first", "last"):
        X, Y, E, inp = pr[f"{phase}_x"], pr[f"{phase}_y"], pr[f"{phase}_extra"], pr[f"{phase}_inp"]
        ep = E[:, extras.index("episode")].astype(int)
        head = Y[:, variables.index("heading")].astype(int)
        # next move: heading change to the next sample of the same episode (0 straight, 1 right, 3 left)
        same = np.r_[ep[1:] == ep[:-1], False]
        turn = np.full(len(ep), -1); turn[:-1] = (head[1:] - head[:-1]) % 4
        nm = same & (turn != 2)
        targets = {"apple_dy": ("reg", Y[:, 0]), "apple_dx": ("reg", Y[:, 1])}
        for k, name in enumerate(variables[2:6]):
            targets[name] = ("cls", Y[:, 2 + k].astype(int))
        targets["heading"] = ("cls", head)
        feats = {"input": prep(inp)}
        for i, lab in enumerate(labels):
            feats[lab] = prep(X[:, offs[i]:offs[i + 1]])
        tab = {}
        for fname, F in feats.items():
            row = {t: cv(F, y, ep, kind) for t, (kind, y) in targets.items()}
            row["next_move"] = cv(F[nm], turn[nm], ep[nm], "cls")
            row["eff_dim"] = participation_ratio(F) if F.shape[1] <= 1500 else float("nan")
            tab[fname] = row
        chance = {t: (float(np.max(np.bincount(y[y >= 0]) / max(1, (y >= 0).sum()))) if kind == "cls" else 0.0)
                  for t, (kind, y) in targets.items()}
        chance["next_move"] = float(np.max(np.bincount(turn[nm]) / nm.sum()))
        res["decode"][phase] = {"table": tab, "chance": chance}
        cols = list(targets) + ["next_move", "eff_dim"]
        print(f"\n== decoding, {phase} 5000 moves (R^2 for apple, accuracy otherwise; chance in brackets)")
        print("  " + f"{'':13s}" + "".join(f"{c[:11]:>12s}" for c in cols))
        print("  " + f"{'(chance)':13s}" + "".join(f"{chance.get(c, float('nan')):12.2f}" for c in cols))
        for fname, row in tab.items():
            print("  " + f"{fname:13s}" + "".join(f"{row[c]:12.2f}" for c in cols))
    json.dump(res, open(f"{out}/results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
