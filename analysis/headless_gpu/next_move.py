"""Action information in the headless GPU run's probe (outputs/*_probe.npz; first and last recorded windows).

Rows are consecutive moves; heading(t) is the heading after row t's move, prev_heading(t) the one before it. Turns are
egocentric: (heading - prev_heading) mod 4 -> straight (0), right (1), left (3). Two targets per row:
  - turn now:  the move that produced row t (prev_heading(t) -> heading(t));
  - turn next: the following move (heading(t) -> heading(t+1), same episode only).
Decoded with ridge one-vs-rest least squares (as probe_analyse.py), time-ordered 70/30 split, classes balanced in the
test set by subsampling. Features: each population's membrane state, all populations, and the input frame (what the
state alone predicts about the move; the network's input is the egocentric frame). Reported: balanced test accuracy
(chance 1/3).

Usage: python -I next_move.py <probe.npz>
"""
import sys
import numpy as np


def ridge_acc(x, y, ridge=1.0, seed=0):
    n = len(y); cut = int(0.7 * n)
    mu, sd = x[:cut].mean(0), x[:cut].std(0) + 1e-6
    xs = np.hstack([(x - mu) / sd, np.ones((n, 1))])
    cls = np.array([0, 1, 3])
    a = xs[:cut]; oh = (y[:cut, None] == cls[None]).astype(float)
    w = np.linalg.solve(a.T @ a + ridge * a.shape[1] * np.eye(a.shape[1]), a.T @ oh)
    pred = cls[np.argmax(xs[cut:] @ w, 1)]; yt = y[cut:]
    accs = [np.mean(pred[yt == c] == c) for c in cls if np.sum(yt == c) >= 10]
    return float(np.mean(accs)), {int(c): int(np.sum(yt == c)) for c in cls}


def main(path):
    z = np.load(path)
    sizes, labels = z["sizes"], [str(s) for s in z["labels"]]
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    for w in ("first", "last"):
        x, y, e, inp = z[f"{w}_x"], z[f"{w}_y"], z[f"{w}_extra"], z[f"{w}_inp"].astype(float)
        hd, prev, ep = y[:, 6].astype(int), e[:, 3].astype(int), e[:, 4]
        turn_now = np.where(prev >= 0, (hd - prev) % 4, -1)
        same = np.r_[ep[1:] == ep[:-1], False]
        turn_next = np.where(same, (np.r_[hd[1:], 0] - hd) % 4, -1)
        nfeat = x.shape[1]
        feats = {lab: (bounds[i], bounds[i + 1]) for i, lab in enumerate(labels)}
        print(f"== {w} window ({len(y)} moves)")
        for tname, t in (("turn now", turn_now), ("turn next", turn_next)):
            ok = np.isin(t, [0, 1, 3])
            line = f"  {tname:9s} counts {dict(zip(*np.unique(t[ok], return_counts=True)))}:"
            for lab, (a, b) in feats.items():
                # membrane potentials are the first half of each population's block if the state is (V, refrac)
                line += f" {lab} {ridge_acc(x[ok][:, a:b], t[ok])[0]:.2f}"
            line += f" | all {ridge_acc(x[ok], t[ok])[0]:.2f} | input frame {ridge_acc(inp[ok], t[ok])[0]:.2f}"
            print(line)


if __name__ == "__main__":
    main(sys.argv[1])
