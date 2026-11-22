"""Event-conditioned selectivity of hidden neurons (narrow_5l probe, 2026-10-30).

Question: in the low-rate regime, are a neuron's rare firing events aligned with task-relevant moments, and is the
information per event higher for sparse neurons (selective plasticity)? Compares the first and last 5000 moves.

Event: the neuron's refractory flag at a move (it fired within the last refractory period). The state at sample t
mostly reflects the frame of sample t-1, so events at t are paired with the variables of t-1 (same episode only).

Usage: python -I event_selectivity.py <probe.npz>
"""
import sys
import numpy as np

z = np.load(sys.argv[1], allow_pickle=False)
labels, sizes = list(z["labels"]), list(z["sizes"])
rng = np.random.default_rng(0)


def discrete_vars(y, extra):
    """Task variables as small categorical codes, plus decision moments."""
    out = {"apple_dy": np.sign(y[:, 0]).astype(int) + 1, "apple_dx": np.sign(y[:, 1]).astype(int) + 1,
           "danger_any": (y[:, 2:6].sum(1) > 0).astype(int), "heading": y[:, 6].astype(int)}
    for i, d in enumerate(("left", "up", "right", "down")):
        out[f"danger_{d}"] = y[:, 2 + i].astype(int)
    # the move direction relative to the apple: is the apple straight ahead / behind / to a side
    return out


def mi(event, code):
    """Mutual information (bits) between a binary event and a categorical code."""
    m = 0.0
    pe = event.mean()
    for c in np.unique(code):
        sel = code == c
        pc = sel.mean()
        for e, p_e in ((1, pe), (0, 1 - pe)):
            pj = np.mean(sel & (event == e))
            if pj > 0:
                m += pj * np.log2(pj / (pc * p_e))
    return m


def mi_matrix(E, code):
    """MI for every column of the binary event matrix E (moves x neurons) with one code, vectorised."""
    T = E.shape[0]
    pe = E.mean(0)
    m = np.zeros(E.shape[1])
    for c in np.unique(code):
        sel = code == c
        pc = sel.mean()
        p1 = E[sel].sum(0) / T
        p0 = pc - p1
        with np.errstate(divide="ignore", invalid="ignore"):
            m += np.where(p1 > 0, p1 * np.log2(p1 / (pc * pe)), 0.0)
            m += np.where(p0 > 0, p0 * np.log2(p0 / (pc * (1 - pe))), 0.0)
    return np.nan_to_num(m)


def analyse(which):
    X, y, extra = z[f"{which}_x"], z[f"{which}_y"], z[f"{which}_extra"]
    ep = extra[:, 4]
    ok = np.r_[False, ep[1:] == ep[:-1]]                  # sample t has a predecessor in the same episode
    idx = np.where(ok)[0]
    codes = discrete_vars(y[idx - 1], extra[idx - 1])
    # decision moments: the heading at t differs from t-1 (a turn was taken on the frame of t-1)
    turn = (y[idx, 6] != y[idx - 1, 6]).astype(int)
    codes["turn"] = turn
    names = list(codes)
    T = len(idx)
    print(f"\n===== {which}: {T} paired moves, {len(np.unique(ep))} episodes, turn rate {turn.mean():.2f}, "
          f"danger rate {codes['danger_any'].mean():.2f}")
    off = 0
    res = {}
    for lab, n in zip(labels, sizes):
        refr = X[:, off + n: off + 2 * n]
        off += 2 * n
        if not lab.endswith("_E") and "field" not in lab:
            continue
        E = (refr[idx] > 0).astype(np.float32)
        rate = E.mean(0)
        live = (rate > 0.002) & (rate < 0.998)            # needs a few events in 5000 moves
        Ev = E[:, live]
        info = np.zeros((len(names), Ev.shape[1]))
        null = np.zeros((len(names), Ev.shape[1]))
        for k, nm in enumerate(names):
            c = codes[nm]
            info[k] = mi_matrix(Ev, c)
            # null: circular shift of the code (keeps both autocorrelations), mean of 5 shifts
            null[k] = np.mean([mi_matrix(Ev, np.roll(c, rng.integers(200, T - 200))) for _ in range(5)], 0)
        corr = np.maximum(info - null, 0.0)               # bias-corrected MI
        best = corr.max(0)
        per_event = best / np.maximum(rate[live], 1e-9)   # bits per event (the event's informativeness)
        # enrichment at decision moments: P(event | turn) / P(event), and the same for danger
        en_turn = Ev[turn == 1].mean(0) / np.maximum(Ev.mean(0), 1e-9)
        en_danger = Ev[codes["danger_any"] == 1].mean(0) / np.maximum(Ev.mean(0), 1e-9)
        r = rate[live]
        q = np.quantile(r, [1 / 3, 2 / 3])
        tiers = [r <= q[0], (r > q[0]) & (r <= q[1]), r > q[1]]
        sig = best > 0.01                                 # > 0.01 bits above the shuffle null
        res[lab] = dict(rate=r, best=best, per_event=per_event)
        rho = np.corrcoef(np.argsort(np.argsort(r)), np.argsort(np.argsort(per_event)))[0, 1]
        top = np.bincount(corr.argmax(0)[sig], minlength=len(names))
        print(f"{lab:13s} live {live.mean():.2f} | median rate/move {np.median(r):.3f} | selective(>0.01b) {sig.mean():.2f} "
              f"| MI med {np.median(best):.4f} | bits/event by rate tier low/mid/high "
              f"{' / '.join(f'{np.median(per_event[t]):.3f}' for t in tiers)} (rank corr rate~bits/event {rho:+.2f}) "
              f"| enrich turn {np.median(en_turn):.2f} danger {np.median(en_danger):.2f} "
              f"| top var {names[int(top.argmax())] if sig.any() else '-'}")
    return res


first = analyse("first")
last = analyse("last")
print("\nChange first -> last (median selective MI, fraction selective):")
for lab in first:
    print(f"{lab:13s} MI {np.median(first[lab]['best']):.4f} -> {np.median(last[lab]['best']):.4f} | "
          f"selective {np.mean(first[lab]['best'] > 0.01):.2f} -> {np.mean(last[lab]['best'] > 0.01):.2f} | "
          f"rate {np.median(first[lab]['rate']):.3f} -> {np.median(last[lab]['rate']):.3f}")
