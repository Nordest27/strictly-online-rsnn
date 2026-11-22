"""ALIF bias toy in a regime closer to the experiments:
sigma = exp(-5), inputs Poisson p=0.7/step in active periods with silent gaps."""
import numpy as np, sys, time
import alif_bias as ab

def make_inputs(rng, D, T, p=0.7):
    X = np.zeros((D, T))
    for d in range(D):
        active = np.zeros(T, bool); t = rng.integers(0, 15)
        while t < T:                                   # alternating active / silent periods
            on = rng.integers(15, 35); off = rng.integers(10, 30)
            active[t:t + on] = True; t += on + off
        X[d] = active * rng.binomial(1, p, T)
    return X

def run(label, sigma, c, N, X, w, h, B=20000):
    ab.s2 = sigma**2
    rng = np.random.default_rng(123)
    D, T = X.shape
    Wset = np.vstack([w] + [w + h * np.eye(D)[d] for d in range(D)] + [w - h * np.eye(D)[d] for d in range(D)])
    fd = np.zeros(D); fdh = [np.zeros(D), np.zeros(D)]
    m = {k: np.zeros(D) for k in ab.EST}; mh = {k: [np.zeros(D), np.zeros(D)] for k in ab.EST}; sq = {k: np.zeros(D) for k in ab.EST}
    rate = 0; nb = N // B; t0 = time.time()
    for ib in range(nb):
        eps = rng.normal(0, sigma, (B, T))
        Z, S = ab.simulate(Wset, X, eps, True)
        F = Z @ c
        fdb = ((F[1:1 + D] - F[1 + D:]) / (2 * h)).mean(1); fd += fdb; fdh[ib % 2] += fdb
        Fc = F[0] - F[0].mean()
        for k in ab.EST:
            est = Fc[:, None] * S[k]; mk = est.mean(0)
            m[k] += mk; mh[k][ib % 2] += mk; sq[k] += (est**2).mean(0)
        rate += Z[0].mean()
    fd /= nb; rate /= nb
    cos = lambda p, q: p @ q / np.linalg.norm(p) / np.linalg.norm(q)
    print(f"\n== {label}: sigma={sigma:.4f}, h={h}, firing prob/step={rate:.3f}, N={N}, {time.time()-t0:.0f}s")
    print(f"   truth halves agree: cos={cos(fdh[0], fdh[1]):.3f}")
    for k in ab.EST:
        mk = m[k] / nb; var = sq[k] / nb - mk**2
        print(f"   {k:12s} cos={cos(mk, fd):6.3f} (halves {cos(mh[k][0], fdh[0]):.3f}/{cos(mh[k][1], fdh[1]):.3f})"
              f"  scale={mk @ fd / (fd @ fd):8.2f}  rel.noise/sample={np.sqrt(var.sum()) / np.linalg.norm(mk):8.1f}")
    return fd, {k: m[k] / nb for k in ab.EST}

if __name__ == "__main__":
    D, T = 10, 150
    rng = np.random.default_rng(5)
    X = make_inputs(rng, D, T)
    w = rng.normal(float(sys.argv[1]) if len(sys.argv) > 1 else 0.02, 0.03, D)
    # firing-rate calibration only
    Z, _ = ab.simulate(w[None, :], X, rng.normal(0, np.exp(-5), (2000, T)), False)
    print(f"input active fraction={X.mean():.3f}, mean weight={w.mean():.3f}, output firing prob/step={Z.mean():.3f}")
