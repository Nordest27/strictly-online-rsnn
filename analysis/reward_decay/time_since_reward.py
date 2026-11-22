"""How far back can the network tell that a reward (an apple) or a death happened?

For every probe sample (one per move), the number of moves since the last apple and since the last death (the start
of the current game) is computed from the recorded snake length and game index. For each distance k, a linear probe
separates "the event was exactly k moves ago" from "it was at least FAR moves ago (or not in the window)", scored by
balanced accuracy above chance (0 = no information, 0.5 = perfect). Features: the hidden state (membrane potential +
adaptation, when the probe recorded it) of the excitatory populations, and the input frame as a control for what is visible on the board.

    python time_since_reward.py RUN_DIR [RUN_DIR ...]
"""
import glob
import sys

import numpy as np

sys.path.insert(0, "/home/user/repos/ml_genn/examples/eprop")
from layer_probe_tests import balanced_probe  # noqa: E402

KS, FAR = (1, 2, 3, 4, 6, 8, 12), 20


def moves_since(event_first):
    """event_first[t]: sample t is the first one after an event. Returns moves since the last event (inf before)."""
    out = np.full(len(event_first), np.inf)
    last = None
    for t, e in enumerate(event_first):
        if e:
            last = t
        if last is not None:
            out[t] = t - last + 1
    return out


def features(z, window):
    x = z[f"{window}_x"].astype(float)
    a = z[f"{window}_a"].astype(float) if f"{window}_a" in z.files else None   # adaptation, if recorded
    sizes = [int(np.prod(s)) for s in z["shapes"]]
    labels = list(z["labels"])
    feats, off, aoff = {}, 0, 0
    for lab, n in zip(labels, sizes):
        v = x[:, off:off + n]; off += 2 * n                      # V (n) then the refractory flag (n)
        feats[lab] = v if a is None else np.hstack([v, a[:, aoff:aoff + n]])
        aoff += n
    e_pops = [l for l in labels if l.endswith("_E")]
    feats["all hidden E"] = np.hstack([feats[l] for l in e_pops])
    feats["input frame"] = z[f"{window}_inp"].astype(float)
    return feats, e_pops


def main():
    for d in sys.argv[1:]:
        z = np.load(glob.glob(f"{d}/outputs/*_probe.npz")[0], allow_pickle=True)
        names = list(z["extra_names"])
        extra = z["last_extra"]
        length, ep = extra[:, names.index("length")], extra[:, names.index("episode")]
        new_game = np.r_[False, ep[1:] != ep[:-1]]
        apple = np.r_[False, (length[1:] > length[:-1]) & ~new_game[1:]]
        since = {"apple": moves_since(apple), "death": moves_since(new_game)}
        feats, e_pops = features(z, "last")
        print(f"\n{d.rstrip('/').split('/')[-1]}  (last window: {int(apple.sum())} apples, {int(new_game.sum())} deaths)")
        for event, s in since.items():
            print(f"  moves since {event}: balanced accuracy above chance, exactly k vs >= {FAR}")
            print(f"    {'features':14s}" + "".join(f"{'k=' + str(k):>7s}" for k in KS))
            for name in ("input frame", "all hidden E", e_pops[0], e_pops[-1], "value_field", "policy_field"):
                row = []
                for k in KS:
                    keep = (s == k) | (s >= FAR)
                    if (s == k).sum() < 20:
                        row.append("     -"); continue
                    row.append(f"{balanced_probe(feats[name][keep], (s[keep] == k).astype(int)):+7.2f}")
                print(f"    {name:14s}" + "".join(row))


if __name__ == "__main__":
    main()
