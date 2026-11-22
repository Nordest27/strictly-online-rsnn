"""Experiment 1 analysis: what did the passive (trained, unused) learning-signal links learn?

Per recording (passive/<name>): (1) the link signal L vs the receivers' credit Y (credit_quality.py: per-window
correlation vs a shuffled null); (2) the learned link weights against their noise floor: under the alignment rule
u += eta (ZF_j Y_i - decay u), a link carrying no information random-walks to sd ~ eta * sd(ZF_j Y_i) / sqrt(2 decay)
(ZF_j from the recorded spikes, Y_i the receiver's recorded credit); (3) |u| / floor and the sign consistency by sender
(volume slab or field population) and receiver slab.

Usage: python passive_analysis.py DIR [DIR ...] [--eta 1e-5 --decay 1e-4 --depth 8]
"""
import sys, argparse, subprocess
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+")
    ap.add_argument("--eta", type=float, default=1e-5); ap.add_argument("--decay", type=float, default=1e-4)
    ap.add_argument("--depth", type=int, default=8); a = ap.parse_args()
    here = Path(__file__).resolve().parent
    for d in a.dirs:
        d = Path(d); name = d.name
        print(f"######## {name}")
        subprocess.run([sys.executable, "-I", str(here / "credit_quality.py"), str(d), name], check=False)
        out = d / "outputs" / f"{name}(0)"
        times = [tuple(map(int, l.split())) for l in open(f"{out}_drift_record_times.txt")]
        recs = [np.load(f"{out}_drift_record_{k}.npz", allow_pickle=True) for k, _ in times]
        last = recs[-1]
        Y = np.concatenate([r["state_CreditY"][:, 1:] for r in recs])          # (W, T, n_tgt)
        alpha = float(last["alpha"])
        srcnames = list(last["source_names"]) if "source_names" in last.files else []
        link_src = [str(x) for x in last["link_sources"]]
        n_tgt = Y.shape[2]
        rows = []
        link_pops = [str(x) for x in last["link_source_pops"]]
        spike_pops = [str(x) for x in last["spike_source_pops"]]
        zf_cache = {}
        for i, sname in enumerate(link_src):
            pre, post, g = last[f"link_{i}_pre"], last[f"link_{i}_post"], last[f"link_{i}_g"]
            pop = link_pops[i]
            if pop in spike_pops:                # the sender's presynaptic trace from its recorded spikes
                k = spike_pops.index(pop)
                if k not in zf_cache:
                    z = np.concatenate([r[f"spikes_{k}"][:, 1:] for r in recs]).astype(float)
                    zf_k = np.empty_like(z); prev = np.zeros((z.shape[0], z.shape[2]))
                    for t in range(z.shape[1]):
                        prev = alpha * prev + z[:, t]; zf_k[:, t] = prev
                    zf_cache[k] = zf_k
                zfk = zf_cache[k]
                prod_sd = np.array([np.std(zfk[:, :, j] * Y[:, :, i_]) for j, i_ in zip(pre, post)])
                floor = a.eta * prod_sd / np.sqrt(2 * a.decay)
            else:                                # sender not recorded (e.g. a field): no floor
                floor = np.full(len(g), np.nan)
            rows.append((f"{sname} ({pop})", pre, post, g, floor))
        print("learned link weights vs their noise floor (median |u| / floor; sign consistency = |mean u| / mean |u|):")
        for sname, pre, post, g, floor in rows:
            ok = np.isfinite(floor) & (floor > 0)
            ratio = np.median(np.abs(g[ok]) / floor[ok]) if ok.any() else float("nan")
            cons = abs(g.mean()) / (np.abs(g).mean() + 1e-30)
            print(f"  from {sname:24s} n {len(g):6d}  median |u|/floor {ratio:6.2f}  mean u {g.mean():+.2e}  "
                  f"consistency {cons:.2f}")
            if ok.any():
                tz = post % a.depth
                by = [f"{s}:{np.median(np.abs(g[ok & (tz == s)]) / floor[ok & (tz == s)]):.2f}"
                      for s in range(a.depth) if (ok & (tz == s)).any()]
                print("      by receiver slab: " + "  ".join(by))


if __name__ == "__main__":
    main()
