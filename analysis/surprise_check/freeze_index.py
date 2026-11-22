"""Does the drift freeze responses around salient moments? Sensitivity psi (the e-prop pseudo-derivative, the rule's
gate) of every hidden neuron at the moves just before an apple or a death, against all other moves.

Prediction: TD-modulated drift lowers psi before apples (better than expected) and raises it before deaths (worse);
surprise-modulated drift lowers it before both. The freeze index 1 - psi_event / psi_routine > 0 means frozen.

    python freeze_index.py NAME=RUN_DIR [NAME=RUN_DIR ...]
"""
import glob
import sys

import numpy as np

VTH, BETA, K = 0.61, 0.0174, 3          # threshold, adaptation strength, moves before an event counted as salient


def psi_and_events(z, window):
    x, a, extra = z[f"{window}_x"].astype(float), z[f"{window}_a"].astype(float), z[f"{window}_extra"]
    sizes = [int(np.prod(s)) for s in z["shapes"]]
    v_parts, r_parts, off = [], [], 0
    for n in sizes:                      # per population: V (n) then the refractory flag (n)
        v_parts.append(x[:, off:off + n]); r_parts.append(x[:, off + n:off + 2 * n]); off += 2 * n
    v, refrac = np.concatenate(v_parts, 1), np.concatenate(r_parts, 1)
    psi = np.where(refrac > 0, 0.0, (0.3 / VTH) * np.maximum(0.0, 1.0 - np.abs(v - (VTH + BETA * a)) / VTH))
    names = list(z["extra_names"])
    length, ep = extra[:, names.index("length")], extra[:, names.index("episode")]
    same_next = np.r_[ep[1:] == ep[:-1], False]
    death_at = ~same_next                                # last recorded move of an episode
    death_at[-1] = False                                 # the window's end is not a death
    apple_at = np.r_[(length[1:] > length[:-1]) & same_next[:-1], False]
    def before(mark):                                    # the K moves up to the event, same episode
        m = np.zeros_like(mark)
        for t in np.nonzero(mark)[0]:
            s = t
            while s >= 0 and t - s < K and ep[s] == ep[t]:
                m[s] = True; s -= 1
        return m
    apple, death = before(apple_at), before(death_at)
    return psi, sizes, list(z["labels"]), apple, death, ~(apple | death)


def main():
    print(f"freeze index 1 - psi(event) / psi(routine), events = the {K} moves before an apple / a death")
    print(f"{'run':16s} {'window':6s} {'population':13s} {'psi routine':>11s} {'apple':>7s} {'death':>7s}")
    for arg in sys.argv[1:]:
        name, d = arg.split("=", 1)
        z = np.load(glob.glob(f"{d}/outputs/*_probe.npz")[0], allow_pickle=True)
        for window in ("first", "last"):
            psi, sizes, labels, apple, death, routine = psi_and_events(z, window)
            off = 0
            for lab, n in zip(labels, sizes):
                p = psi[:, off:off + n].mean(1); off += n
                r = p[routine].mean()
                fi = lambda m: 1.0 - p[m].mean() / r if m.any() else float("nan")
                print(f"{name:16s} {window:6s} {lab:13s} {r:11.4f} {fi(apple):+7.3f} {fi(death):+7.3f}")
        print(f"{'':16s} events (last window): {int(apple.sum())} apple moves, {int(death.sum())} death moves")


if __name__ == "__main__":
    main()
