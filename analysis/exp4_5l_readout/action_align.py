"""Which move does a probe sample's activity encode? Decode the turn into the sample (prev_heading -> heading) and the
turn out of it (heading -> next sample's heading), per population, late 5000 moves (and first)."""
import sys
import numpy as np
sys.path.insert(0, "/home/user/repos/personal/analysis/exp4_5l_readout")
from analyse import prep, cv
d = np.load(sys.argv[1], allow_pickle=False)
labels = [str(x) for x in d["labels"]]; offs = np.concatenate([[0], np.cumsum(d["sizes"])])
for phase in ("first", "last"):
    X, Y, E = d[f"{phase}_x"], d[f"{phase}_y"], d[f"{phase}_extra"]
    ep = E[:, 4].astype(int); head = Y[:, 6].astype(int); prev = E[:, 3].astype(int)
    t_in = (head - prev) % 4; ok_in = t_in != 2
    same = np.r_[ep[1:] == ep[:-1], False]; t_out = np.full(len(ep), -1); t_out[:-1] = (head[1:] - head[:-1]) % 4
    ok_out = same & (t_out != 2)
    ch_in = np.bincount(t_in[ok_in]).max() / ok_in.sum(); ch_out = np.bincount(t_out[ok_out]).max() / ok_out.sum()
    print(f"== {phase}: turn INTO the sample (chance {ch_in:.2f}) | turn OUT of it (chance {ch_out:.2f})")
    feats = {"input": prep(d[f"{phase}_inp"])}
    for i, lab in enumerate(labels):
        feats[lab] = prep(X[:, offs[i]:offs[i + 1]])
    for name, F in feats.items():
        print(f"  {name:13s} into {cv(F[ok_in], t_in[ok_in], ep[ok_in], 'cls'):.2f}   out {cv(F[ok_out], t_out[ok_out], ep[ok_out], 'cls'):.2f}")
