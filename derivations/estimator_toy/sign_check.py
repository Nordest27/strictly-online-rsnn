"""Sign check: train a single ALIF neuron's input weights with the perturbation update
exactly as in eprop_compiler.py (DeltaG += reward * e * NoiseTrace / sigma^2; optimiser
does w -= lr * DeltaG) versus the flipped sign, and track the expected reward.
Weight-dist placement (shared post noise, per-synapse trace), sparse decaying reward,
whole-trajectory credit (reward minus batch baseline times the summed score)."""
import numpy as np
import final_tables as ft
ev = ft.ev
alpha, rho, v_th, beta_a, gamma_pd, n_ref = ft.alpha, ft.rho, ft.v_th, ft.beta_a if hasattr(ft, "beta_a") else ft.beta, ft.gamma_pd, ft.n_ref
D, T, X, targets = ft.D, ft.T, ft.X, ft.targets
kappa_y, kappa_r, SIG = ft.kappa_y, ft.kappa_r, ft.SIG_S

def batch(w, B, rng):
    drive = w @ X
    v = np.zeros(B); a = np.zeros(B); z = np.zeros(B); ref = np.zeros(B); y = np.zeros(B); r = np.zeros(B); F = np.zeros(B)
    zf = np.zeros(D); ea = np.zeros((B, D)); xi = np.zeros((B, D)); S = np.zeros((B, D))
    for t in range(T):
        xin = X[:, t - 1] if t > 0 else np.zeros(D)
        a = rho * a + z; A = v_th + beta_a * a
        nz = rng.normal(0, 1, B)[:, None]
        applied = SIG * nz * xin[None, :]                     # eps = SynSig * Noise, on presynaptic spikes
        v = alpha * v + (drive[t - 1] if t > 0 else 0.0) - v_th * z + applied.sum(1)
        ref = np.maximum(ref - 1, 0); refr = ref > 0
        z = ((v >= A) & ~refr).astype(float); ref = np.where(z > 0, n_ref, ref)
        y = kappa_y * y + z
        R = 0.0
        if t in targets:
            thr, sgn = targets[t]; R = sgn * np.tanh((y - thr) / 0.5)
        r = kappa_r * r + R; F += r
        psi = np.where(refr, 0.0, (gamma_pd / v_th) * np.maximum(0, 1 - np.abs(v - A) / v_th))
        zf = alpha * zf + xin
        e = psi[:, None] * (zf[None, :] - beta_a * ea); ea = psi[:, None] * zf[None, :] + (rho - psi[:, None] * beta_a) * ea
        xi = alpha * xi + applied                              # NoiseTrace
        S += e * xi / SIG**2                                   # e * NoiseTrace / SynSig^2  (as in the code)
    return F, S

def train(sign, steps=40, B=4000, lr=2e-3, seed=0):
    rng = np.random.default_rng(seed); w = ev.w.copy(); hist = []
    m = np.zeros(D); s = np.zeros(D)
    for k in range(steps):
        F, S = batch(w, B, rng)
        hist.append(F.mean())
        DeltaG = sign * ((F - F.mean())[:, None] * S).mean(0)   # code: DeltaG += reward * score  (sign=+1)
        m = 0.9 * m + 0.1 * DeltaG; s = 0.999 * s + 0.001 * (DeltaG - m)**2 + 1e-12   # AdaBelief-like
        w = w - lr * m / (np.sqrt(s / (1 - 0.999**(k + 1))) + 1e-8)                   # Variable -= ...
    F, _ = batch(w, B, rng); hist.append(F.mean())
    return np.array(hist)

for sign, name in ((+1, "code's sign   (DeltaG += reward*score)"), (-1, "flipped sign  (DeltaG -= reward*score)")):
    h = train(sign)
    print(f"{name}: E[F] start {h[0]:+.3f} -> end {h[-1]:+.3f}   (every 10 steps: {np.round(h[::10], 3)})")
