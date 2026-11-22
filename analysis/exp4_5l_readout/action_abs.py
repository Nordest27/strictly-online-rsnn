"""Decode the next ABSOLUTE direction (the readout's action space) per population, vs chance and vs the
'keep going straight' strategy (predict next = current heading, using the true heading)."""
import sys
import numpy as np
sys.path.insert(0, "/home/user/repos/personal/analysis/exp4_5l_readout")
from analyse import prep, cv
d = np.load(sys.argv[1], allow_pickle=False)
labels = [str(x) for x in d["labels"]]; offs = np.concatenate([[0], np.cumsum(d["sizes"])])
for phase in ("first", "last"):
    X, Y, E = d[f"{phase}_x"], d[f"{phase}_y"], d[f"{phase}_extra"]
    ep = E[:, 4].astype(int); head = Y[:, 6].astype(int)
    same = np.r_[ep[1:] == ep[:-1], False]
    nxt = np.full(len(ep), -1); nxt[:-1] = head[1:]
    ok = same
    chance = np.bincount(nxt[ok]).max() / ok.sum()
    straight = np.mean(nxt[ok] == head[ok])
    print(f"== {phase}: next absolute direction (chance {chance:.2f}; 'keep straight' with the true heading {straight:.2f})")
    feats = {"input": prep(d[f"{phase}_inp"])}
    for i, lab in enumerate(labels):
        feats[lab] = prep(X[:, offs[i]:offs[i + 1]])
    for name, F in feats.items():
        print(f"  {name:13s} next {cv(F[ok], nxt[ok], ep[ok], 'cls'):.2f}   current heading {cv(F[ok], head[ok], ep[ok], 'cls'):.2f}")
