"""Monte-Carlo check of the perturbation-trace bias characterisation.

Single leaky neuron with accumulating Gaussian membrane noise:
    v^t = alpha v^{t-1} + w x^{t-1} + eps^t,  eps^t ~ N(0, sigma^2)
Noise-free part u^t, noise part xi^t (AR(1), started stationary).
Reward F(v) = sum_t c_t * sigmoid(k (v^t - theta))  (smooth, nonlinear).

Compares, for d E[F] / dw:
  (1) truth: finite differences with common random numbers
  (2) exact score estimator:  F * sum_t eps^t x^{t-1} / sigma^2
  (3) trace estimator:        F * sum_t (1-alpha^2) xi^t zbar^t / sigma^2
  (4) predicted value of (3): sum_s g_s ztilde^s,  ztilde^s = sum_t alpha^|t-s| zbar^t,
      g_s = E[dF/dv^s]
  (5) truth via pathwise:     sum_s g_s zbar^s
"""
import numpy as np

rng = np.random.default_rng(0)
T, alpha, sigma, w = 25, 0.9, 1.0, 0.5
k, theta = 1.5, 1.0
x = rng.uniform(0, 1, T)          # x[t-1] enters at step t (t = 1..T)
c = rng.normal(0, 1, T)           # reward weights per step
N, B = 4_000_000, 200_000         # samples, batch size

def membrane(w, eps, xi0):
    u = np.zeros(T); v = np.empty((eps.shape[0], T)); xi = np.empty_like(v)
    up, xp = 0.0, xi0
    for t in range(T):
        up = alpha * up + w * x[t]
        xp = alpha * xp + eps[:, t]
        u[t] = up; xi[:, t] = xp; v[:, t] = up + xp
    return u, xi, v

sig = lambda a: 1 / (1 + np.exp(-a))
F = lambda v: (c * sig(k * (v - theta))).sum(1)
dF = lambda v: c * k * sig(k * (v - theta)) * (1 - sig(k * (v - theta)))

zbar = np.array([sum(alpha ** (t - r) * x[r] for r in range(t + 1)) for t in range(T)])
K = alpha ** np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
ztilde = K @ zbar

acc = dict(fd=0, exact=0, trace=0, g=np.zeros(T), Fm=0)
h = 1e-3
nb = N // B
for _ in range(nb):
    eps = rng.normal(0, sigma, (B, T))
    xi0 = rng.normal(0, sigma / np.sqrt(1 - alpha**2), B)
    u, xi, v = membrane(w, eps, xi0)
    _, _, vp = membrane(w + h, eps, xi0)
    _, _, vm = membrane(w - h, eps, xi0)
    Fv = F(v)
    Fc = Fv - Fv.mean()           # baseline: does not change expectations
    acc['fd'] += ((F(vp) - F(vm)) / (2 * h)).mean()
    acc['exact'] += (Fc * (eps * x).sum(1) / sigma**2).mean()
    acc['trace'] += (Fc * ((1 - alpha**2) * xi * zbar).sum(1) / sigma**2).mean()
    acc['g'] += dF(v).mean(0)
for key in ('fd', 'exact', 'trace'):
    acc[key] /= nb
g = acc['g'] / nb

print(f"(1) truth, finite differences      : {acc['fd']: .4f}")
print(f"(5) truth, pathwise sum g_s zbar_s  : {g @ zbar: .4f}")
print(f"(2) exact score estimator (MC)      : {acc['exact']: .4f}")
print(f"(3) trace estimator (MC)            : {acc['trace']: .4f}")
print(f"(4) predicted trace value g.ztilde  : {g @ ztilde: .4f}")
