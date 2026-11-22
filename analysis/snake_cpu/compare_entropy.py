"""Compare the entropy runs (small Snake network, CPU, proposed rule, seed 1, 1.5M timesteps).

Per block of simulation timesteps: game reward per move, policy entropy (uniform = 1.386), mean max probability,
running |TD error| (the critic's error; a continuous entropy reward could inflate it) and hidden firing rate.
Usage: python3 compare_entropy.py [block]
"""
import csv
import glob
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = {
    "none": HERE / "entropy_ab2/none",
    "bonus const 0.3": HERE / "entropy_ab2/const",
    "bonus |td| 3.0": HERE / "entropy_ab2/abstd",
    "maxent 0.01": HERE / "maxent_ab/r0.01",
    "maxent 0.03": HERE / "maxent_ab/r0.03",
}


def rows(pattern):
    files = glob.glob(str(pattern))
    return list(csv.DictReader(open(files[0]))) if files else []


def blocks(main, pol, block):
    """Per block: reward/move, entropy, max prob, |td|, Hz (main CSV and policy CSV share episodes)."""
    by_ep = {int(r["episode"]): r for r in pol}
    out, acc, t, edge = [], None, 0.0, block
    for r in main:
        if acc is None:
            acc = {"s": 0.0, "n": 0.0, "h": [], "m": [], "td": [], "hz": []}
        acc["s"] += float(r["score"]); acc["n"] += float(r["ep_steps"]); t += float(r["ep_steps"])
        p = by_ep.get(int(r["episode"]))
        if p is not None:
            acc["h"].append(float(p["entropy"])); acc["m"].append(float(p["max_prob"]))
            acc["td"].append(float(p["abs_td"]))
        acc["hz"].append(1000 * float(r["frequency"]))
        if t >= edge:
            mean = lambda x: sum(x) / len(x) if x else float("nan")
            out.append((30 * acc["s"] / acc["n"], mean(acc["h"]), mean(acc["m"]), mean(acc["td"]), mean(acc["hz"])))
            acc, edge = None, edge + block
    return out


def main(block=2.5e5):
    for name, d in RUNS.items():
        b = blocks(rows(d / "outputs/*).csv"), rows(d / "outputs/*_policy.csv"), block)
        if not b:
            print(f"{name:16s} no data yet"); continue
        print(f"{name:16s} " + " | ".join(f"{r:+.3f} H{h:.2f} p{m:.2f} td{td:.3f} {hz:.0f}Hz" for r, h, m, td, hz in b))


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 2.5e5)
