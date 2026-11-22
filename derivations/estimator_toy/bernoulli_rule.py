"""Spike-decision REINFORCE with an exact, local, state-dependent baseline (fresh-noise version):
score = (z_t - p_t) * pre,  p_t = Phi((v_t - xi_t - A_t)/s),  s = stationary std of the accumulated noise.
Raw (uncentred) task reward, + sign (ascent). Variants: plain (z-p)*pre, and the full Bernoulli score
(z-p) * phi / (s p (1-p)) * pre (clipped p). Compared with the centred-reward flip reference."""
import numpy as np
from math import erf, sqrt
import gate_difference_rule as gd
cc = gd.cc
D, T, X, targets = cc.D, cc.T, cc.X, cc.targets
alpha, rho, v_th, beta_a, n_ref, LAM, SIG = cc.alpha, cc.rho, cc.v_th, cc.beta_a, cc.n_ref, cc.LAM, cc.SIG
Phi = np.vectorize(lambda u: 0.5 * (1 + erf(u / sqrt(2))))
class norm:
    _erf = np.vectorize(lambda u: erf(u))
    @staticmethod
    def cdf(u): return 0.5 * (1 + norm._erf(np.asarray(u) / sqrt(2)))
    @staticmethod
    def pdf(u): return np.exp(-0.5 * np.asarray(u)**2) / sqrt(2 * np.pi)

def sim(w, B, rng, variant):
    drive = w @ X
    v = np.zeros(B); a = np.zeros(B); z = np.zeros(B); ref = np.zeros(B); y = np.zeros(B); r = np.zeros(B); F = np.zeros(B)
    zf = np.zeros(D); xt = np.zeros(B); var = 0.0; S = []; R = []
    for t in range(T):
        xin = X[:, t - 1] if t > 0 else np.zeros(D)
        nz = rng.normal(0, 1, B); s_in = SIG * xin.sum(); pert = s_in * nz
        a = rho * a + z; A = v_th + beta_a * a
        v = alpha * v + (drive[t - 1] if t > 0 else 0.0) - v_th * z + pert; xt = alpha * xt + pert
        var = alpha**2 * var + s_in**2; s = np.sqrt(max(var, 1e-12))     # std of accumulated noise (known)
        ref = np.maximum(ref - 1, 0); refr = ref > 0
        z = ((v >= A) & ~refr).astype(float); ref = np.where(z > 0, n_ref, ref)
        s = max(s_in, 1e-12)                                    # std of THIS step's fresh noise
        u = (v - pert - A) / s if s_in > 0 else np.where(v - A >= 0, 50.0, -50.0)
        p = np.where(refr, 0.0, norm.cdf(u))
        if variant == "plain":
            g = (z - p)
        else:
            pc = np.clip(p, 1e-3, 1 - 1e-3); g = np.where(refr, 0.0, (z - p) * norm.pdf(u) / (s * pc * (1 - pc)))
        y = cc.kappa_y * y + z; Rt = 0.0
        if t in targets: thr, sgn = targets[t]; Rt = sgn * np.tanh((y - thr) / 0.5)
        r = cc.kappa_r * r + Rt; F += r
        zf = alpha * zf + xin
        S.append(g[:, None] * zf[None, :]); R.append(r.copy())
    return np.array(S), np.array(R), F

def train(variant, steps=30, B=3000, lr=3e-3, seed=0):
    rng = np.random.default_rng(seed); w = cc.ev.w.copy(); m_ = np.zeros(D); s_ = np.zeros(D); h = []
    for k in range(steps):
        S, R, F = sim(w, B, rng, variant); h.append(F.mean())
        RL = np.zeros((B, D)); DG = np.zeros(D)
        for t in range(T):
            RL = LAM * RL + S[t]; DG += (R[t][:, None] * RL).mean(0)
        DG /= T
        m_ = 0.9 * m_ + 0.1 * DG; s_ = 0.999 * s_ + 0.001 * (DG - m_)**2 + 1e-18
        w = w + lr * m_ / (np.sqrt(s_ / (1 - 0.999**(k + 1))) + 1e-12)
    h.append(sim(w, B, rng, variant)[2].mean()); return np.array(h)

for variant in ("plain", "bernoulli"):
    h = train(variant)
    print(f"(z - p) {variant:9s} raw reward, + sign: E[F] {h[0]:+.2f} -> {h[-1]:+.2f}   ({np.round(h[::10], 2)})", flush=True)
