"""Scale check: does the perturbation gradient estimate the same gradient as exact (symmetric) e-prop?

For each dump (diag_dump_<timestep>.npz) and each connection into a field (where symmetric e-prop is exact):
  cos(E, G1)   cosine across synapses between AccE (sum of delta * e-prop trace) and AccG1 (ungated gradient part)
  cos(E, G)    the same for the gated gradient part (gate -PsiBar)
  shuffled     cos(E, G1) with the synapse pairing shuffled (what pure noise gives)
  slope G1/E   least-squares slope of AccG1 on AccE (prediction with node noise: ~1)
  psibar_eff   median of AccG / AccG1 over synapses (the effective gate)
  rate corr    weight noise only: correlation over synapses between the presynaptic rate and AccG1/AccE
               projected on E (the predicted extra factor grows with the presynaptic rate)
Usage: python3 analyse.py [WEIGHT_SHARED|NODE ...]
"""
import glob
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main(noises):
    rng = np.random.default_rng(0)
    for noise in noises:
        dumps = sorted(glob.glob(str(HERE / noise / "diag_dump_*.npz")), key=lambda f: int(f.split("_")[-1][:-4]))
        if not dumps:
            print(f"{noise}: no dumps yet"); continue
        print(f"\n{noise}")
        print(f"  {'timestep':>9s} {'connection':12s} {'cos(E,G1)':>9s} {'cos(E,G)':>9s} {'shuffled':>9s} "
              f"{'slope G1/E':>10s} {'psibar_eff':>10s} {'rate corr':>9s}")
        for f in dumps:
            d = np.load(f)
            ts = int(f.split("_")[-1][:-4])
            for key in sorted({k.split("|")[0] for k in d.files}):
                e, g, g1 = (d[f"{key}|{v}"].astype(np.float64) for v in ("AccE", "AccG", "AccG1"))
                shuf = np.mean([cos(e, rng.permutation(g1)) for _ in range(20)])
                slope = float(e @ g1 / (e @ e + 1e-30))
                with np.errstate(divide="ignore", invalid="ignore"):
                    eff = np.nanmedian(np.where(np.abs(g1) > 0, g / g1, np.nan))
                rc = float("nan")
                if f"{key}|rate_pre" in d.files:
                    rate = d[f"{key}|rate_pre"][d[f"{key}|pre"]]
                    proj = g1 * np.sign(e)                     # G1 along the sign of the e-prop estimate
                    if np.std(rate) > 0:
                        rc = float(np.corrcoef(rate, proj / (np.abs(e) + 1e-12 * np.abs(e).max()))[0, 1])
                print(f"  {ts:9d} {key:12s} {cos(e, g1):9.3f} {cos(e, g):9.3f} {shuf:9.3f} {slope:10.3g} "
                      f"{eff:10.3g} {rc:9.3f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["WEIGHT_SHARED", "NODE"])
