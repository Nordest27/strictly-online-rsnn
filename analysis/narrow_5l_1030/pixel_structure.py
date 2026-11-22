"""Do the hidden layers carry a screen-like copy of the input? (narrow_5l probe, 2026-10-30)

For each population with a 2-D sheet (shape H x W x C) and the input frame:
  * pixel tracking: for each neuron, max |corr| of its membrane potential with any single input pixel (frame of the
    previous sample, the alignment of layer_probe_tests.py), median over neurons;
  * topography: distance (in input cells) between the neuron's sheet position (scaled to the input grid) and its best
    pixel, against the same with neuron positions shuffled; "topo" = 1 - mean distance / shuffled mean distance
    (1: every neuron tracks the pixel under it, 0: no map);
  * picture: mean potential (summed over channels) when the apple is at each position relative to the head, minus
    the overall mean, per layer: a screen-like layer shows a blob that moves with the apple.

Usage: python -I pixel_structure.py OUT_PREFIX LABEL=probe.npz:window [LABEL=probe.npz:window ...]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CMAP = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f2f1ec", "#e34948"])   # cool - neutral - warm
CELL = 4                                       # input pixels per board cell (scale)


def load(spec):
    label, _, rest = spec.partition("=")
    path, _, window = rest.rpartition(":")
    z = np.load(path, allow_pickle=False)
    X, y, ep = z[f"{window}_x"], z[f"{window}_y"], z[f"{window}_extra"][:, 4]
    inp = z[f"{window}_inp"].astype(np.float32).reshape(-1, 20, 20, 3) / 255.0   # stored flattened (H, W, C)
    idx = np.where(np.r_[False, ep[1:] == ep[:-1]])[0]
    pops, off = {}, 0
    for lab, n, shp in zip(z["labels"], z["sizes"], z["shapes"]):
        pops[str(lab)] = (X[idx, off:off + n].astype(np.float32), tuple(int(s) for s in shp))
        off += 2 * n                           # V then the refractory flags
    return label, pops, inp[idx - 1], y[idx - 1]


def corr_max(V, P):
    """Per neuron: max |corr| with any pixel and the index of that pixel."""
    Vz = (V - V.mean(0)) / (V.std(0) + 1e-6)
    Pz = (P - P.mean(0)) / (P.std(0) + 1e-6)
    C = np.abs(Vz.T @ Pz) / len(V)
    C[:, P.std(0) < 1e-6] = 0.0
    return C.max(1), C.argmax(1)


def analyse(label, pops, inp, y, rng):
    T = len(inp)
    H, W = 20, 20
    P = inp.reshape(T, -1)
    pix_yx = np.stack(np.unravel_index(np.arange(P.shape[1]), inp.shape[1:]), 1)
    rows = []
    for name, (V, shp) in pops.items():
        if len(shp) != 3:
            continue
        h, w, c = shp
        mx, best = corr_max(V, P)
        pos = np.stack(np.unravel_index(np.arange(V.shape[1]), (h, w, c)), 1)[:, :2] * np.array([H / h, W / w])
        bp = pix_yx[best, :2]
        d = np.linalg.norm(pos - bp, axis=1) / CELL
        dsh = np.mean([np.linalg.norm(pos[rng.permutation(len(pos))] - bp, axis=1).mean() / CELL for _ in range(20)])
        rows.append((name, float(np.median(mx)), float(np.percentile(mx, 90)), 1.0 - d.mean() / dsh))
    print(f"\n== {label}: {T} samples")
    print(f"{'population':14s} {'median max|r|':>14s} {'90th pct':>9s} {'topo':>6s}")
    for name, med, p90, topo in rows:
        print(f"{name:14s} {med:14.3f} {p90:9.3f} {topo:6.2f}")
    return rows


def body_maps(pops, inp, y, names):
    """Mean summed-channel potential when a body segment occupies the cell at each offset from the head, minus the
    mean when it does not (the body channel of the input: green, 1.0; the head is 0.6 and always at the centre)."""
    offsets = [(0, -1), (-1, 0), (0, 1), (1, 0), (-1, -1), (1, 1)]
    body = inp[..., 1]
    maps = {}
    for name in names:
        A = body if name == "input" else pops[name][0].reshape(len(inp), *pops[name][1]).sum(-1)
        maps[name] = []
        for a, b in offsets:
            cy, cx = (2 + a) * CELL + CELL // 2, (2 + b) * CELL + CELL // 2
            on = body[:, cy, cx] > 0.9
            maps[name].append(A[on].mean(0) - A[~on].mean(0) if on.sum() > 20 and (~on).sum() > 20 else None)
    return offsets, maps


def apple_maps(pops, inp, y, names):
    """Mean summed-channel potential per apple offset (dy, dx in -2..2) minus the overall mean, per population."""
    dy, dx = y[:, 0].astype(int), y[:, 1].astype(int)
    offsets = [(-2, 0), (-1, -1), (0, -2), (0, 2), (1, 1), (2, 0)]
    maps = {}
    for name in names:
        if name == "input":
            A = inp[..., 2]                                    # the apple channel
        else:
            V, (h, w, c) = pops[name]
            A = V.reshape(len(V), h, w, c).sum(-1)
        mean = A.mean(0)
        maps[name] = [(A[(dy == a) & (dx == b)].mean(0) - mean) if np.any((dy == a) & (dx == b)) else None
                      for a, b in offsets]
    return offsets, maps


MODE = "apple"


def main():
    global MODE
    args = sys.argv[1:]
    if args[0] in ("--apple", "--body"):
        MODE = args.pop(0)[2:]
    out = args[0]
    sys.argv = [sys.argv[0]] + args
    rng = np.random.default_rng(0)
    runs = [load(s) for s in sys.argv[2:]]
    if MODE == "apple":
        for label, pops, inp, y in runs:
            analyse(label, pops, inp, y, rng)
    names = ["input", "L1_E", "L2_E", "L3_E", "L4_E", "L5_E", "policy_field"]
    ncol = 6
    fig, axes = plt.subplots(len(names), ncol * len(runs) + len(runs) - 1,
                             figsize=(0.95 * (ncol * len(runs) + 1), 0.95 * len(names) + 0.8), squeeze=False,
                             gridspec_kw={"width_ratios": ([1] * ncol + [0.35]) * (len(runs) - 1) + [1] * ncol})
    for r, (label, pops, inp, y) in enumerate(runs):
        offsets, maps = (body_maps if MODE == "body" else apple_maps)(pops, inp, y, names)
        c0 = r * (ncol + 1)
        for i, name in enumerate(names):
            vals = np.concatenate([np.abs(m).ravel() for m in maps[name] if m is not None])
            lim = np.percentile(vals, 99) + 1e-9          # robust scale: a few extreme cells do not wash out the row
            for j, m in enumerate(maps[name]):
                ax = axes[i, c0 + j]
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if m is not None:
                    ax.imshow(m, cmap=CMAP, vmin=-lim, vmax=lim, interpolation="nearest")
                if i == 0:
                    ax.set_title(f"{MODE} {offsets[j][0]:+d},{offsets[j][1]:+d}", fontsize=6, color="#52514e")
                if j == 0:
                    ax.set_ylabel(name, fontsize=7, color="#0b0b0b", rotation=0, ha="right", va="center")
        axes[0, c0].text(0, 1.55, label, transform=axes[0, c0].transAxes, fontsize=8, fontweight="bold",
                         color="#0b0b0b")
    for r in range(len(runs) - 1):
        for ax in axes[:, r * (ncol + 1) + ncol]:
            ax.set_visible(False)
    fig.suptitle(("Mean activity when a body segment is at an offset (dy, dx) from the head, minus when it is not"
                  if MODE == "body" else "Mean activity when the apple is at an offset (dy, dx) from the head, minus "
                  "the overall mean") + "; each row on its own scale (99th percentile); red higher, blue lower",
                 fontsize=7, color="#52514e")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{out}.png", dpi=200)
    print(f"\nwrote {out}.png")


if __name__ == "__main__":
    main()
