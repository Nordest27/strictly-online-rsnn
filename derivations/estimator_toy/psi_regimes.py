"""Does psi reduce relative noise when the neuron is sparse or saturated?
ALIF toy, jagged full-trajectory reward, sigma = exp(-5). Direct Monte-Carlo:
  relative noise = sqrt(total variance) / |mean|  (per estimator)
  direction      = cos(mean, mean of exact estimator)  (exact is unbiased)
Regimes set by the mean input weight."""
import sys, time, numpy as np
import alif_bias_mp as m
alpha, rho, v_th, beta, gamma_pd, n_ref = m.alpha, m.rho, m.v_th, m.beta, m.gamma_pd, m.n_ref
D, T, SIG = m.D, m.T, m.SIGMA
EST = ["dist_eprop", "trace_nopsi", "step_pretr", "step_psi", "exact"]
X, w0, rewards = m.setup(); c = rewards["jagged"]

def run(w, N, B=25000, seed=0):
    rng = np.random.default_rng(seed); s2 = SIG**2; drive = w @ X
    S1 = {k: np.zeros(D) for k in EST}; S2 = {k: 0.0 for k in EST}; n = 0
    stats = dict(fire=0.0, open=0.0, below=0.0, above=0.0)
    for _ in range(N // B):
        eps = rng.normal(0, SIG, (B, T))
        v = np.zeros(B); a = np.zeros(B); z = np.zeros(B); ref = np.zeros(B); F = np.zeros(B)
        xi = np.zeros(B); ev = np.zeros(D); ea = np.zeros((B, D)); C = {k: np.zeros((B, D)) for k in EST}
        for t in range(T):
            xin = X[:, t - 1] if t > 0 else np.zeros(D)
            a = rho * a + z; A = v_th + beta * a
            v = alpha * v + (drive[t - 1] if t > 0 else 0.0) - v_th * z + eps[:, t]
            ref = np.maximum(ref - 1, 0); refractory = ref > 0
            z = ((v >= A) & ~refractory).astype(float); ref = np.where(z > 0, n_ref, ref)
            F += c[t] * z
            psi = np.where(refractory, 0.0, (gamma_pd / v_th) * np.maximum(0, 1 - np.abs(v - A) / v_th))
            stats["fire"] += z.mean(); stats["open"] += (psi > 0).mean()
            stats["below"] += (v < A - v_th).mean(); stats["above"] += (v > A + v_th).mean()
            ev = alpha * ev + xin
            e = psi[:, None] * (ev[None, :] - beta * ea)
            xi = alpha * xi + eps[:, t]
            sc_tr = (1 - alpha**2) * xi / s2; sc_st = eps[:, t] / s2
            C["dist_eprop"] += sc_tr[:, None] * e
            C["trace_nopsi"] += sc_tr[:, None] * ev[None, :]
            C["step_pretr"] += sc_st[:, None] * ev[None, :]
            C["step_psi"] += sc_st[:, None] * e
            C["exact"] += sc_st[:, None] * xin[None, :]
            ea = psi[:, None] * ev[None, :] + (rho - psi[:, None] * beta) * ea
        Fc = F - F.mean()
        for k in EST:
            est = Fc[:, None] * C[k]; S1[k] += est.sum(0); S2[k] += (est**2).sum()
        n += B
    steps = (N // B) * T
    stats = {k: 100 * v_ / steps for k, v_ in stats.items()}
    mean = {k: S1[k] / n for k in EST}
    rel = {k: np.sqrt(S2[k] / n - mean[k] @ mean[k]) / np.linalg.norm(mean[k]) for k in EST}
    cos = {k: mean[k] @ mean["exact"] / np.linalg.norm(mean[k]) / np.linalg.norm(mean["exact"]) for k in EST}
    return stats, rel, cos

if __name__ == "__main__":
    N = int(float(sys.argv[1])) if len(sys.argv) > 1 else 500_000
    rng = np.random.default_rng(5); jitter = rng.normal(0, 1, D)
    print(f"{'mean w':>6s} {'fire%':>6s} {'psi open%':>9s} {'far below%':>10s} {'far above%':>10s} | "
          + " | ".join(f"{k:>11s}" for k in EST) + "   (cos to exact / rel.noise)")
    for mw in (0.01, 0.02, 0.04, 0.08, 0.15, 0.3):
        w = mw + 0.75 * mw * jitter                  # same relative spread as the baseline
        t0 = time.time(); st, rel, cos = run(w, N)
        print(f"{mw:6.2f} {st['fire']:6.1f} {st['open']:9.1f} {st['below']:10.1f} {st['above']:10.1f} | "
              + " | ".join(f"{cos[k]:5.3f}/{rel[k]:5.1f}" for k in EST) + f"   [{time.time()-t0:.0f}s]", flush=True)
