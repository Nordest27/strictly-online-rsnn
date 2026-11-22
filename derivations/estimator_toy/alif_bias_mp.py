"""Parallel, streaming version of the realistic ALIF bias experiment.
sigma = exp(-5); inputs Poisson p=0.7 in active periods with silent gaps; 10% output firing.
Truth by central finite differences at two step sizes (common random numbers)."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, time, sys
from multiprocessing import Pool
import alif_bias_realistic as ar

alpha = np.exp(-1 / 10); rho = np.exp(-1 / 300)
v_th, beta, gamma_pd, n_ref = 0.61, 0.0174, 0.3, 3
EST = ["dist_eprop", "trace_nopsi", "step_pretr", "step_psi", "exact"]
D, T, SIGMA = 10, 150, np.exp(-5)
HS = (5e-4, 2e-4)

def setup():
    rng = np.random.default_rng(5)
    X = ar.make_inputs(rng, D, T)
    w = rng.normal(0.04, 0.03, D)
    t = np.arange(T)
    rewards = {"slow": np.where((t > 50) & (t < 100), 1.0, -0.3),
               "jagged": np.random.default_rng(7).normal(0, 1, T)}
    return X, w, rewards

def worker(args):
    seed, nbatch, B, cname = args
    X, w, rewards = setup(); c = rewards[cname]; s2 = SIGMA**2
    Wset = np.vstack([w] + [w + s * h * np.eye(D)[d] for h in HS for s in (1, -1) for d in range(D)])
    M = Wset.shape[0]; drive = Wset @ X
    rng = np.random.default_rng(seed)
    out = dict(fd=np.zeros((len(HS), D)), m={k: np.zeros(D) for k in EST}, sq={k: np.zeros(D) for k in EST}, n=0, rate=0.0)
    for _ in range(nbatch):
        eps = rng.normal(0, SIGMA, (B, T))
        v = np.zeros((M, B)); a = np.zeros((M, B)); z = np.zeros((M, B)); ref = np.zeros((M, B)); F = np.zeros((M, B))
        xi = np.zeros(B); ev = np.zeros(D); ea = np.zeros((B, D)); S = {k: np.zeros((B, D)) for k in EST}
        for t in range(T):
            xin = X[:, t - 1] if t > 0 else np.zeros(D)
            a = rho * a + z; A = v_th + beta * a
            v = alpha * v + (drive[:, t - 1][:, None] if t > 0 else 0.0) - v_th * z + eps[None, :, t]
            ref = np.maximum(ref - 1, 0); refractory = ref > 0
            z = ((v >= A) & ~refractory).astype(float); ref = np.where(z > 0, n_ref, ref)
            F += c[t] * z
            psi = np.where(refractory[0], 0.0, (gamma_pd / v_th) * np.maximum(0, 1 - np.abs(v[0] - A[0]) / v_th))
            ev = alpha * ev + xin
            e = psi[:, None] * (ev[None, :] - beta * ea)
            xi = alpha * xi + eps[:, t]
            sc_tr = (1 - alpha**2) * xi / s2; sc_st = eps[:, t] / s2
            S["dist_eprop"] += sc_tr[:, None] * e
            S["trace_nopsi"] += sc_tr[:, None] * ev[None, :]
            S["step_pretr"] += sc_st[:, None] * ev[None, :]
            S["step_psi"] += sc_st[:, None] * e
            S["exact"] += sc_st[:, None] * xin[None, :]
            ea = psi[:, None] * ev[None, :] + (rho - psi[:, None] * beta) * ea
            out["rate"] += z[0].sum()
        for ih, h in enumerate(HS):
            base = 1 + ih * 2 * D
            out["fd"][ih] += ((F[base:base + D] - F[base + D:base + 2 * D]) / (2 * h)).sum(1)
        Fc = F[0] - F[0].mean()
        for k in EST:
            est = Fc[:, None] * S[k]; out["m"][k] += est.sum(0); out["sq"][k] += (est**2).sum(0)
        out["n"] += B
    return out

def main(cname, N, workers=28, B=4000):
    nb_total = N // B; per = [nb_total // workers + (i < nb_total % workers) for i in range(workers)]
    jobs = [(1000 + i, per[i], B, cname) for i in range(workers) if per[i] > 0]
    t0 = time.time()
    with Pool(len(jobs)) as p:
        res = p.map(worker, jobs)
    cos = lambda p, q: p @ q / np.linalg.norm(p) / np.linalg.norm(q)
    def agg(rs):
        n = sum(r["n"] for r in rs)
        fd = sum(r["fd"] for r in rs) / n
        m = {k: sum(r["m"][k] for r in rs) / n for k in EST}
        sq = {k: sum(r["sq"][k] for r in rs) / n for k in EST}
        return n, fd, m, sq
    n, fd, m, sq = agg(res)
    _, fdA, mA, _ = agg(res[0::2]); _, fdB, mB, _ = agg(res[1::2])
    rate = sum(r["rate"] for r in res) / (n * T)
    truth = fd[-1]                                   # finer step as reference
    print(f"\n== {cname} reward: sigma={SIGMA:.4f}, firing prob/step={rate:.3f}, N={n:,}, {time.time()-t0:.0f}s on {len(jobs)} workers")
    print(f"   truth stability: cos(h={HS[0]}, h={HS[1]}) = {cos(fd[0], fd[1]):.3f}; halves (h={HS[1]}) = {cos(fdA[-1], fdB[-1]):.3f}")
    for k in EST:
        var = sq[k] - m[k]**2
        print(f"   {k:12s} cos={cos(m[k], truth):6.3f} (halves {cos(mA[k], fdA[-1]):.3f}/{cos(mB[k], fdB[-1]):.3f})"
              f"  scale={m[k] @ truth / (truth @ truth):7.2f}  rel.noise/sample={np.sqrt(var.sum()) / np.linalg.norm(m[k]):7.1f}")

if __name__ == "__main__":
    main("jagged", int(8e6))
    main("slow", int(24e6))
