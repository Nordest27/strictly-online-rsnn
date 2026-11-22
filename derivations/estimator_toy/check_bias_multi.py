"""Second check: several synapses (different input patterns), other alpha,
sharper reward. Vector gradient over D weights; reports per-weight values and
the cosine between the trace estimator and the true gradient."""
import numpy as np
rng = np.random.default_rng(1)

def run(alpha, k, theta, T=30, D=6, N=4_000_000, B=200_000, sigma=1.0):
    X = np.zeros((D, T))
    X[0] = rng.uniform(0, 1, T)                      # noisy input
    X[1] = (np.arange(T) % 7 == 0).astype(float)     # sparse periodic spikes
    X[2] = (np.arange(T) > T // 2).astype(float)     # late input
    X[3] = (np.arange(T) < T // 3).astype(float)     # early input
    X[4] = np.sin(np.arange(T) / 3) ** 2             # smooth oscillation
    X[5] = rng.binomial(1, 0.2, T)                   # random spikes
    w = rng.normal(0.3, 0.1, D)
    c = rng.normal(0, 1, T)
    sig = lambda a: 1 / (1 + np.exp(-a))
    F = lambda v: (c * sig(k * (v - theta))).sum(1)
    dF = lambda v: c * k * sig(k * (v - theta)) * (1 - sig(k * (v - theta)))
    drive = w @ X                                     # deterministic input per step
    Zbar = np.array([[sum(alpha ** (t - r) * X[d, r] for r in range(t + 1)) for t in range(T)] for d in range(D)])
    Kmat = alpha ** np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
    Ztil = Zbar @ Kmat.T
    ex = np.zeros(D); tr = np.zeros(D); g = np.zeros(T)
    for _ in range(N // B):
        eps = rng.normal(0, sigma, (B, T))
        xp = rng.normal(0, sigma / np.sqrt(1 - alpha**2), B); up = 0.0
        v = np.empty((B, T)); xi = np.empty((B, T))
        for t in range(T):
            up = alpha * up + drive[t]; xp = alpha * xp + eps[:, t]
            xi[:, t] = xp; v[:, t] = up + xp
        Fv = F(v); Fc = Fv - Fv.mean()
        ex += (Fc[:, None] * (eps @ X.T)).mean(0) / sigma**2
        tr += (Fc[:, None] * ((1 - alpha**2) * xi @ Zbar.T)).mean(0) / sigma**2
        g += dF(v).mean(0)
    nb = N // B; ex /= nb; tr /= nb; g /= nb
    truth = Zbar @ g; pred = Ztil @ g
    cos = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
    print(f"alpha={alpha}, k={k}")
    print("  truth (pathwise)   :", np.round(truth, 3))
    print("  exact estimator MC :", np.round(ex, 3))
    print("  trace estimator MC :", np.round(tr, 3))
    print("  trace predicted    :", np.round(pred, 3))
    print(f"  cos(trace, truth) = {cos(tr, truth):.3f}   cos(exact, truth) = {cos(ex, truth):.3f}"
          f"   max rel. err trace vs predicted = {np.max(np.abs(tr - pred) / np.abs(pred)):.3f}")

run(alpha=0.7, k=3.0, theta=0.8)
run(alpha=0.905, k=3.0, theta=1.2)
