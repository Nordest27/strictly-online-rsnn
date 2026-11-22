"""Layer activity over real frames: the trained network and the reservoir replayed on the identical frames, move by
move, for one episode (narrow_5l, 2026-10-30).

Each panel is a population's membrane potential per sheet position (z-scored per neuron over the whole recording,
averaged over the 3 channels): red above the neuron's own mean, blue below. The state at sample t is shown with the
frame of sample t-1, the frame it mostly reflects (the alignment of layer_probe_tests.py). The replay's samples are
the recording's minus each episode's first one (presented at reset), so they are matched by position.

Usage: python -I frames_movie.py TRAINED.npz WINDOW REPLAY.npz WINDOW EPISODE OUT_PREFIX
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

CMAP = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f2f1ec", "#e34948"])
LAYERS = ["L1_E", "L2_E", "L3_E", "L4_E", "L5_E", "policy_field"]
ZLIM = 2.5


def maps(path, window):
    z = np.load(path, allow_pickle=False)
    X = z[f"{window}_x"]
    out, off = {}, 0
    for lab, n, shp in zip(z["labels"], z["sizes"], z["shapes"]):
        lab = str(lab)
        if lab in LAYERS:
            V = X[:, off:off + n].astype(np.float32)
            V = (V - V.mean(0)) / (V.std(0) + 1e-6)
            out[lab] = V.reshape(len(V), *[int(s) for s in shp]).mean(-1)
        off += 2 * n
    return z, out


def main():
    tpath, twin, rpath, rwin, episode, out = sys.argv[1:7]
    episode = int(episode)
    tz, tm = maps(tpath, twin)
    _, rm = maps(rpath, rwin)
    ep = tz[f"{twin}_extra"][:, 4]
    keep = np.r_[True, ep[1:] == ep[:-1]]
    keep[0] = False
    rindex = np.cumsum(keep) - 1                      # replay sample index of each kept recording sample
    frames = tz[f"{twin}_inp"].reshape(-1, 20, 20, 3)[..., ::-1]     # stored BGR -> RGB
    steps = [i for i in np.where(ep == episode)[0] if keep[i] and keep[i - 1] if i > 0]
    print(f"episode {episode}: {len(steps)} moves shown; snake length "
          f"{int(tz[f'{twin}_extra'][steps[0], 2])} -> {int(tz[f'{twin}_extra'][steps[-1], 2])}")

    def draw(i, axes):
        axes[0, 0].imshow(frames[i - 1], interpolation="nearest")
        axes[1, 0].imshow(frames[i - 1], interpolation="nearest")
        for k, lab in enumerate(LAYERS):
            axes[0, k + 1].imshow(tm[lab][i], cmap=CMAP, vmin=-ZLIM, vmax=ZLIM, interpolation="nearest")
            axes[1, k + 1].imshow(rm[lab][rindex[i]], cmap=CMAP, vmin=-ZLIM, vmax=ZLIM, interpolation="nearest")
        for ax in axes.ravel():
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        for k, lab in enumerate(["frame"] + LAYERS):
            axes[0, k].set_title(lab, fontsize=7, color="#52514e")
        axes[0, 0].set_ylabel("trained", fontsize=8, color="#0b0b0b")
        axes[1, 0].set_ylabel("reservoir", fontsize=8, color="#0b0b0b")

    # GIF: one image per move
    images = []
    for n, i in enumerate(steps):
        fig, axes = plt.subplots(2, len(LAYERS) + 1, figsize=(8.4, 2.9))
        draw(i, axes)
        fig.suptitle(f"move {n + 1}/{len(steps)} - z-scored potential per neuron (red above its mean, blue below)",
                     fontsize=7, color="#52514e")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    images[0].save(f"{out}.gif", save_all=True, append_images=images[1:], duration=350, loop=0)

    # strip: 8 consecutive moves from the middle of the episode, one row pair per move
    mid = steps[len(steps) // 2 - 4: len(steps) // 2 + 4]
    fig, axes = plt.subplots(2 * len(mid), len(LAYERS) + 1, figsize=(8.4, 1.25 * 2 * len(mid)))
    for r, i in enumerate(mid):
        draw(i, axes[2 * r: 2 * r + 2])
        axes[2 * r, 0].set_ylabel(f"move {steps.index(i) + 1}\ntrained", fontsize=7)
        axes[2 * r + 1, 0].set_ylabel("reservoir", fontsize=7)
        if r:
            for ax in axes[2 * r]:
                ax.set_title("")
    fig.tight_layout()
    fig.savefig(f"{out}_strip.png", dpi=150)
    print(f"wrote {out}.gif ({len(images)} frames) and {out}_strip.png")


if __name__ == "__main__":
    main()
