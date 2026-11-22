"""Check the small-noise closed form for the variance of perturbation estimators.
Linear leaky membrane with accumulating noise (xi^0 = 0), smooth sigmoid reward.
Prediction (Isserlis, reward linearised in the noise):
    mean_i = sigma^2 (g_eps . a_i)
    rel.noise^2 = 1 + |g_eps|^2 sum_i |a_i|^2 / sum_i (g_eps . a_i)^2
with a_i the score profile of each estimator on the innovations eps."""
import numpy as np
rng = np.random.default_rng(0)
T, D, alpha, k, theta = 40, 6, 0.9, 3.0, 1.0
X = rng.binomial(1, 0.4, (D, T)).astype(float); w = rng.normal(0.15, 0.05, D); c = rng.normal(0, 1, T)
sig = lambda u: 1 / (1 + np.exp(-u))
Fr = lambda v: (c * sig(k * (v - theta))).sum(-1)
dF = lambda v: c * k * sig(k * (v - theta)) * (1 - sig(k * (v - theta)))
# noise-free membrane and presynaptic trace (input x^{t-1} enters at step t; here index t uses X[:, t])
u = np.zeros(T); up = 0.0; zbar = np.zeros((D, T)); acc = np.zeros(D)
for t in range(T):
    up = alpha * up + w @ X[:, t]; u[t] = up
    acc = alpha * acc + X[:, t]; zbar[:, t] = acc
Fw = np.tril(alpha ** np.subtract.outer(np.arange(T), np.arange(T)).clip(0)) * (np.subtract.outer(np.arange(T), np.arange(T)) >= 0)
g = dF(u)                                  # reward sensitivity to the membrane (small-noise: at noise-free v)
g_eps = Fw.T @ g                           # sensitivity to each innovation: sum_{t>=s} alpha^{t-s} g_t

for sigma in (0.02, 0.005):
    s2 = sigma**2
    profiles = {                             # a_i[s]: coefficient of eps^s in synapse i's score
        "exact":       X / s2,
        "step_pretr":  zbar / s2,
        "trace_nopsi": (1 - alpha**2) / s2 * (zbar @ Fw),   # sum_t xi^t zbar^t = sum_s eps^s sum_{t>=s} alpha^{t-s} zbar^t
    }
    N, B = 2_000_000, 200_000
    m = {k_: np.zeros(D) for k_ in profiles}; sq = {k_: np.zeros(D) for k_ in profiles}
    for _ in range(N // B):
        eps = rng.normal(0, sigma, (B, T))
        v = u + eps @ Fw.T
        Fc = Fr(v); Fc -= Fc.mean()
        for k_, a in profiles.items():
            est = Fc[:, None] * (eps @ a.T)
            m[k_] += est.mean(0); sq[k_] += (est**2).mean(0)
    nb = N // B
    print(f"\nsigma={sigma}:  truth (pathwise) = {np.round(zbar @ g, 3)}")
    for k_, a in profiles.items():
        mk = m[k_] / nb; var = sq[k_] / nb - mk**2
        rel_mc = np.sqrt(var.sum()) / np.linalg.norm(mk)
        pred_mean = s2 * a @ g_eps
        rel_pred = np.sqrt(1 + (g_eps @ g_eps) * (a**2).sum() / ((a @ g_eps)**2).sum())
        print(f"   {k_:12s} mean MC={np.round(mk, 3)}  pred={np.round(pred_mean, 3)}   rel.noise MC={rel_mc:6.2f}  pred={rel_pred:6.2f}")
