"""Does each layer carry the agent's DECISION? Decoding the trained agent's action from the hidden state, per layer,
for the trained network and for the reservoir replayed on the same frames (narrow_5l, 2026-10-30).

The action taken on the frame of sample t-1 is the heading change between samples t-1 and t (egocentric: turn left,
straight, turn right). Alignment as layer_probe_tests.py: the state at sample t is paired with the frame of t-1, i.e.
the state at the end of processing that frame, when the decision is read out. Features: each population's V and
refractory flags (as the probe stores them), reduced to the top 100 principal components (fit on the training part);
ridge one-vs-rest, time-ordered 70/30 split; balanced accuracy above chance (mean recall - 1/3).

Usage: python -I action_decoding.py <trained_probe.npz> <window> <replay_probe.npz> <window>
"""
import sys
import numpy as np


def load(path, window):
    z = np.load(path, allow_pickle=False)
    X, y, ep, inp = z[f"{window}_x"], z[f"{window}_y"], z[f"{window}_extra"][:, 4], z[f"{window}_inp"]
    idx = np.where(np.r_[False, ep[1:] == ep[:-1]])[0]
    turn = ((y[idx, 6] - y[idx - 1, 6]) % 4).astype(int)        # 0 straight, 1 clockwise, 3 anticlockwise
    ok = turn != 2                                               # reversing is impossible; drop any artefact
    idx, turn = idx[ok], turn[ok]
    feats = {"input frame": inp[idx - 1].astype(np.float32)}
    off = 0
    for lab, n in zip(z["labels"], z["sizes"]):
        feats[str(lab)] = X[idx, off:off + 2 * n]
        off += 2 * n
    # targets: the action, and for comparison two state variables of the same frame
    targets = {"action": np.searchsorted([0, 1, 3], turn),
               "apple_y": np.sign(y[idx - 1, 0]).astype(int) + 1,
               "danger_ahead": y[idx - 1, 2 + y[idx - 1, 6].astype(int)].astype(int)}
    return feats, targets, idx


def balanced_acc(Xtr, ytr, Xte, yte, ridge=1.0, k=100):
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    P = vt[:k].T
    A, B = A @ P, B @ P
    A1, B1 = np.c_[A, np.ones(len(A))], np.c_[B, np.ones(len(B))]
    classes = np.unique(ytr)
    Y = (ytr[:, None] == classes[None]).astype(float)
    W = np.linalg.solve(A1.T @ A1 + ridge * np.eye(A1.shape[1]), A1.T @ Y)
    pred = classes[np.argmax(B1 @ W, 1)]
    rec = [np.mean(pred[yte == c] == c) for c in classes if np.any(yte == c)]
    return float(np.mean(rec)) - 1.0 / len(classes)


def report(name, feats, targets):
    n = len(next(iter(targets.values())))
    cut = int(0.7 * n)
    print(f"\n== {name}: {n} samples; action shares straight/cw/acw "
          f"{np.bincount(targets['action'], minlength=3) / n}")
    print(f"{'features':14s}" + "".join(f"{t:>14s}" for t in targets))
    for f, X in feats.items():
        X = X.astype(np.float64)
        row = [balanced_acc(X[:cut], t[:cut], X[cut:], t[cut:]) for t in targets.values()]
        print(f"{f:14s}" + "".join(f"{v:+14.3f}" for v in row))


tf, tt, ti = load(sys.argv[1], sys.argv[2])
rf, rt, ri = load(sys.argv[3], sys.argv[4])
report("trained network", tf, tt)
report("reservoir replay (same frames, the trained agent's actions)", rf, rt)
