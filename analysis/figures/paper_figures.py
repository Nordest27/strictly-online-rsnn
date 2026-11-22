"""Paper figures from Snake / Pac-Man run outputs.

Every run is a score CSV ``<prefix>(<rep>).csv`` (one row per episode) plus, if the run had ``monitor: true``, the
neuron monitor ``<prefix>(<rep>)_neurons.csv`` (one row per monitor interval, per-population statistics). Activity
(firing rate, voltage loss, silent neurons) is always read from the monitor: the score CSV's ``frequency`` and
``voltage`` columns came from a single population in runs made before the logging fix, so they are never used here.

Everything is put on a timestep axis (cumulative ``ep_steps``), the natural axis of a continuing task:
  * episode reward  = mean score of the episodes that end in a bin;
  * reward rate     = total score of those episodes / steps they took, per 1000 steps;
  * hidden rate     = mean of the excitatory layers' rates (L1_E ... Lk_E, equal-size layers);
  * voltage loss    = mean of the excitatory layers' voltage-regularisation loss.
Seeds are averaged on a common grid of bins; the band is the min-max over seeds (no band for a single run).

Usage:
  python paper_figures.py comparison --group "Ours=runs/ours_s*" --group "e-prop (random)=runs/rand_s*" \
      --out snake_comparison
  python paper_figures.py depth --run "Ours, 5 layers=runs/ours5_s0" --out snake_depth

A run path is a run directory (its outputs/*(0).csv is used), a score CSV, or a glob of either. Files saved from the
browser as "name(5) (1).csv" are paired with "name(5)_neurons (1).csv". Writes <out>.pdf and <out>.png.
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- style: reference palette (light), validated categorical order; colour follows the rule, never the order given
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CANONICAL = {"ours": 0, "proposed": 0, "e-prop (random)": 1, "random": 1, "e-prop (adaptive)": 2, "adaptive": 2,
             "e-prop (symmetric)": 3, "symmetric": 3}
RESERVOIR = MUTED                      # the fixed-weight baseline is a reference, not a competing rule
LAYER_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]   # ordinal blue, L1 light -> deep dark

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.4, "lines.solid_capstyle": "round",
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300,
})


# ---- loading
def _score_files(path):
    out = []
    for p in sorted(glob.glob(path)) or [path]:
        if os.path.isdir(p):
            out += [f for f in sorted(glob.glob(os.path.join(p, "outputs", "*.csv")))
                    if re.search(r"\(\d+\)\.csv$", f)]
        elif p.endswith(".csv") and not re.search(r"_(neurons|policy)( \(\d+\))?\.csv$", p):
            out.append(p)
    if not out:
        raise FileNotFoundError(f"no score CSV for {path!r}")
    return out


def _neurons_file(score_csv):
    m = re.match(r"^(.*?)( \(\d+\))?\.csv$", score_csv)
    f = f"{m.group(1)}_neurons{m.group(2) or ''}.csv"
    return f if os.path.exists(f) else None


def _read_csv(path):
    with open(path) as fh:
        header = fh.readline().strip().split(",")
    if os.path.getsize(path) <= len(",".join(header)) + 2:
        return {h: np.zeros(0) for h in header}
    data = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=float, invalid_raise=False)
    data = np.atleast_2d(data)
    return {h: data[:, i] for i, h in enumerate(header)}


class Run:
    def __init__(self, score_csv):
        self.name = os.path.basename(score_csv)
        s = _read_csv(score_csv)
        self.t_end = np.cumsum(s["ep_steps"])            # timestep at which each episode ends
        self.score, self.steps = s["score"], s["ep_steps"]
        nf = _neurons_file(score_csv)
        self.neurons = _read_csv(nf) if nf else None
        if self.neurons is not None and len(self.neurons["timestep"]) == 0:
            self.neurons = None

    @property
    def T(self):
        return float(self.t_end[-1]) if len(self.t_end) else 0.0

    def binned(self, edges):
        """Episode reward and reward rate (per 1000 steps) in each timestep bin."""
        idx = np.digitize(self.t_end, edges) - 1
        reward, rate = np.full(len(edges) - 1, np.nan), np.full(len(edges) - 1, np.nan)
        for b in range(len(edges) - 1):
            sel = idx == b
            if sel.any() and edges[b + 1] <= self.T:            # only bins the run covers completely
                reward[b] = self.score[sel].mean()
                rate[b] = 1000.0 * self.score[sel].sum() / self.steps[sel].sum()
        return reward, rate

    def layers(self, pop="E"):
        if self.neurons is None:
            return []
        return sorted({int(m.group(1)) for k in self.neurons
                       if (m := re.match(rf"^L(\d+)_{pop}_hz$", k))})

    def monitor(self, column, centres, pop="E", layer=None):
        """A monitor statistic on the bin centres: one layer, or the mean over the population's layers."""
        if self.neurons is None:
            return np.full(len(centres), np.nan)
        t = self.neurons["timestep"]
        layers = [layer] if layer is not None else self.layers(pop)
        y = np.mean([self.neurons[f"L{k}_{pop}_{column}"] for k in layers], axis=0)
        out = np.interp(centres, t, y)
        out[(centres < t[0]) | (centres > t[-1])] = np.nan
        return out


def _runs(spec):
    label, _, path = spec.partition("=")
    return label.strip(), [Run(f) for f in _score_files(path.strip())]


def _colour(label, used):
    key = label.lower()
    if "reservoir" in key:
        return RESERVOIR
    if key in CANONICAL:
        return SLOTS[CANONICAL[key]]
    free = [i for i in range(len(SLOTS)) if i not in used]
    return SLOTS[free[0]]


def _xfmt(ax, T):
    scale, unit = (1e6, "M") if T >= 1e6 else (1e3, "k")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / scale:g}{unit}"))
    ax.set_xlim(0, T)


def _band(ax, x, ys, colour, label, ls="-"):
    ys = np.array(ys, dtype=float)
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(ys, 0)
        if len(ys) > 1:
            ax.fill_between(x, np.nanmin(ys, 0), np.nanmax(ys, 0), color=colour, alpha=0.15, lw=0)
    ax.plot(x, mean, color=colour, label=label, ls=ls)
    return mean


def _direct_labels(ax, ends, min_gap_pt=8.0):
    """Label each series at its last point; labels are pushed apart vertically so they never overlap."""
    ends = sorted((y, x, lab, c) for x, y, lab, c in ends if np.isfinite(y))
    if not ends:
        return
    fig = ax.figure
    fig.canvas.draw()
    to_pt = lambda y: ax.transData.transform((0, y))[1] * 72.0 / fig.dpi
    pos = [to_pt(y) for y, *_ in ends]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_gap_pt)
    for (y, x, lab, c), p in zip(ends, pos):
        ax.plot([x], [y], "o", ms=2.5, color=c, clip_on=False)
        ax.annotate(lab, (x, y), xytext=(4, p - to_pt(y)), textcoords="offset points", va="center",
                    fontsize=6.5, color=INK2, annotation_clip=False)


# ---- figures
def comparison(args):
    groups = [_runs(g) for g in args.group]
    T = args.steps or min(r.T for _, runs in groups for r in runs)     # every run covers the whole axis
    edges = np.linspace(0, T, args.bins + 1)
    x = 0.5 * (edges[1:] + edges[:-1])
    panels = [("Episode reward", None), ("Reward rate (per 1000 steps)", None),
              ("Hidden E rate (Hz)", "hz"), ("Voltage loss", "vloss")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9), constrained_layout=True)
    used, ends = [], []
    for label, runs in groups:
        c = _colour(label, used)
        if c in SLOTS:
            used.append(SLOTS.index(c))
        rew, rate = zip(*(r.binned(edges) for r in runs))
        series = [rew, rate,
                  [r.monitor("hz", x) for r in runs],
                  [r.monitor("vloss", x) for r in runs]]
        for ax, ys in zip(axes, series):
            m = _band(ax, x, ys, c, f"{label} (n={len(runs)})")
            if ax is axes[1]:
                ok = np.where(np.isfinite(m))[0]
                if len(ok):
                    ends.append((x[ok[-1]], m[ok[-1]], label, c))
        missing = sum(r.neurons is None for r in runs)
        if missing:
            print(f"warning: {label}: {missing}/{len(runs)} runs have no monitor rows (activity panels omit them)")
    for ax, (title, _) in zip(axes, panels):
        ax.set_title(title, loc="left", color=INK)
        _xfmt(ax, T)
        ax.set_xlabel("Timestep")
    if len(groups) <= 4:                  # direct labels on the headline panel; the legend carries identity anyway
        _direct_labels(axes[1], ends)
    axes[3].set_ylim(bottom=0)
    axes[2].set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(len(groups), 4))
    _save(fig, args.out)


def depth(args):
    runs = [(lab, rs[0]) for lab, rs in (_runs(r) for r in args.run)]
    fig, axes = plt.subplots(2, len(runs), figsize=(min(7.0, 2.4 * len(runs) + 0.6), 3.2), sharex="col",
                             sharey="row", constrained_layout=True, squeeze=False)
    for j, (label, run) in enumerate(runs):
        T = args.steps or run.T                 # each column on its own run's length
        x = np.linspace(0, T, args.bins)
        layers = run.layers()
        if not layers:
            print(f"warning: {label}: no monitor rows")
        for k in layers:
            c = LAYER_RAMP[min(k - 1, len(LAYER_RAMP) - 1)]
            axes[0, j].plot(x, run.monitor("hz", x, layer=k), color=c, label=f"L{k}")
            axes[1, j].plot(x, run.monitor("dead_silent", x, layer=k), color=c, label=f"L{k}")
        axes[0, j].set_title(label, loc="left", color=INK)
        for ax in axes[:, j]:
            _xfmt(ax, T)
        axes[1, j].set_xlabel("Timestep")
    axes[0, 0].set_ylabel("E rate (Hz)")
    axes[1, 0].set_ylabel("Silent E fraction")
    axes[0, 0].set_ylim(bottom=0)
    axes[1, 0].set_ylim(0, None)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center", title="Layer", title_fontsize=7)
    _save(fig, args.out)


def _save(fig, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    print(f"wrote {out}.pdf / .png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("comparison", help="reward, reward rate, hidden rate, voltage loss per rule")
    c.add_argument("--group", action="append", required=True, help='"Label=path-or-glob" (one per rule)')
    d = sub.add_parser("depth", help="per-layer rate and silent fraction, one column per run")
    d.add_argument("--run", action="append", required=True, help='"Label=path"')
    for p in (c, d):
        p.add_argument("--out", required=True)
        p.add_argument("--bins", type=int, default=60)
        p.add_argument("--steps", type=float, default=None, help="x-axis length (default: longest run)")
    a = ap.parse_args()
    {"comparison": comparison, "depth": depth}[a.cmd](a)
