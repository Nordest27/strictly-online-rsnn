"""Single-process, vectorised run of the realistic ALIF bias experiment (smooth reward).
Usage: python3 alif_bias_simple.py N_SAMPLES [BATCH]"""
import sys, time, numpy as np
import alif_bias_mp as m
m.HS = (5e-4,)                                   # one finite-difference step (validated in the jagged run)
N = int(float(sys.argv[1])); B = int(sys.argv[2]) if len(sys.argv) > 2 else 25000
nb = max(N // B, 2)
cos = lambda p, q: p @ q / np.linalg.norm(p) / np.linalg.norm(q)
halves = [None, None]; t0 = time.time()
for i in range(nb):
    r = m.worker((2000 + i, 1, B, "slow"))
    h = halves[i % 2]
    if h is None: halves[i % 2] = r
    else:
        h["fd"] += r["fd"]; h["n"] += r["n"]; h["rate"] += r["rate"]
        for k in m.EST: h["m"][k] += r["m"][k]; h["sq"][k] += r["sq"][k]
    if i == 0: print(f"first batch: {time.time()-t0:.1f}s -> estimated total {nb*(time.time()-t0)/60:.1f} min", flush=True)
n = halves[0]["n"] + halves[1]["n"]
fdA, fdB = halves[0]["fd"][0] / halves[0]["n"], halves[1]["fd"][0] / halves[1]["n"]
truth = (halves[0]["fd"][0] + halves[1]["fd"][0]) / n
rate = (halves[0]["rate"] + halves[1]["rate"]) / (n * m.T)
print(f"\n== slow reward: sigma={m.SIGMA:.4f}, firing prob/step={rate:.3f}, N={n:,}, {time.time()-t0:.0f}s")
print(f"   truth halves agree: cos={cos(fdA, fdB):.3f}")
for k in m.EST:
    mk = (halves[0]["m"][k] + halves[1]["m"][k]) / n
    var = (halves[0]["sq"][k] + halves[1]["sq"][k]) / n - mk**2
    cA = cos(halves[0]["m"][k] / halves[0]["n"], fdA); cB = cos(halves[1]["m"][k] / halves[1]["n"], fdB)
    print(f"   {k:12s} cos={cos(mk, truth):6.3f} (halves {cA:.3f}/{cB:.3f})  scale={mk @ truth / (truth @ truth):7.2f}"
          f"  rel.noise/sample={np.sqrt(var.sum()) / np.linalg.norm(mk):7.1f}")
