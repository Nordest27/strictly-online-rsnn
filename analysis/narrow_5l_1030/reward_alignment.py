"""Do hidden neurons' rare firing events concentrate around reward moments? (narrow_5l probe, 2026-10-30)

Reward moments, from the probe: an apple is eaten when the snake's length increases between consecutive samples of an
episode; a death is the last sample of an episode (not the last episode of the snapshot, which may be cut). For each
neuron, enrichment at lag k = P(event at reward_move + k) / P(event). Null: the same reward times shifted by a random
offset inside each episode (keeps the episode structure and the neuron's own autocorrelation).

Usage: python -I reward_alignment.py <probe.npz>
"""
import sys
import numpy as np

z = np.load(sys.argv[1], allow_pickle=False)
labels, sizes = list(z["labels"]), list(z["sizes"])
rng = np.random.default_rng(1)
LAGS = range(-4, 2)                      # moves relative to the reward sample (state lags the frame by ~1 sample)


def events_at(E, times, lag):
    t = times + lag
    t = t[(t >= 0) & (t < E.shape[0])]
    return E[t].mean(0) if len(t) else np.full(E.shape[1], np.nan)


for which in ("first", "last"):
    X, extra = z[f"{which}_x"], z[f"{which}_extra"]
    ep, length = extra[:, 4], extra[:, 2]
    same = np.r_[False, ep[1:] == ep[:-1]]
    eat = np.where(same & (np.r_[0, np.diff(length)] > 0))[0]
    ends = np.where(np.r_[ep[1:] != ep[:-1], False])[0]
    ep_start = {e: np.where(ep == e)[0][0] for e in np.unique(ep)}
    ep_len = {e: np.sum(ep == e) for e in np.unique(ep)}

    def shifted(times):
        out = []
        for t in times:
            e = ep[t]
            s, n = ep_start[e], ep_len[e]
            out.append(s + rng.integers(0, n))
        return np.array(out)

    print(f"\n===== {which}: {len(eat)} apples eaten, {len(ends)} deaths, {X.shape[0]} moves")
    off = 0
    for lab, n in zip(labels, sizes):
        refr = X[:, off + n: off + 2 * n]
        off += 2 * n
        if not lab.endswith("_E") and "field" not in lab:
            continue
        E = (refr > 0).astype(np.float32)
        base = E.mean(0)
        live = base > 0.002
        E, base = E[:, live], base[live]
        line = f"{lab:13s}"
        for name, times in (("eat", eat), ("death", ends)):
            if len(times) < 10:
                continue
            enr = np.array([np.nanmedian(events_at(E, times, k) / base) for k in LAGS])
            # null distribution of the median enrichment at the best lag, from shifted reward times
            nulls = []
            for _ in range(50):
                st = shifted(times)
                nulls.append(max(np.nanmedian(events_at(E, st, k) / base) for k in LAGS))
            # fraction of neurons strongly enriched (x2) at the best lag, against the null
            kbest = int(np.argmax(enr))
            frac = np.mean(events_at(E, times, list(LAGS)[kbest]) / base > 2.0)
            frac_null = np.mean([np.mean(events_at(E, shifted(times), list(LAGS)[kbest]) / base > 2.0)
                                 for _ in range(20)])
            line += (f" | {name}: median enrichment by lag {list(LAGS)[0]}..{list(LAGS)[-1]} "
                     f"[{' '.join(f'{v:.2f}' for v in enr)}] null95 {np.percentile(nulls, 95):.2f}"
                     f"; x2-enriched {frac:.2f} (null {frac_null:.2f})")
        print(line)
