"""Spike-flip rule: REINFORCE on spike decisions. flip_t = z_t - z_cf,t where z_cf is the spike the
neuron would have emitted without its accumulated noise this step (v - xi_tot >= threshold, same
refractory state). Score = pre_t * flip_t (no 1/sigma^2, no continuous noise). Lambda trace, raw
(uncentred) task reward. Both signs tested; also per-moment-centred reward as a reference."""
import numpy as np
import gate_difference_rule as gd
cc = gd.cc
D, T, X, targets = cc.D, cc.T, cc.X, cc.targets
alpha, rho, v_th, beta_a, gamma_pd, n_ref, LAM, SIG = cc.alpha, cc.rho, cc.v_th, cc.beta_a, cc.gamma_pd, cc.n_ref, cc.LAM, cc.SIG

def sim(w, B, rng):
    drive = w @ X
    v = np.zeros(B); a = np.zeros(B); z = np.zeros(B); ref = np.zeros(B); y = np.zeros(B); r = np.zeros(B); F = np.zeros(B)
    zf = np.zeros(D); xt = np.zeros(B); Sflip = []; R = []; nflip = 0
    for t in range(T):
        xin = X[:, t - 1] if t > 0 else np.zeros(D)
        nz = rng.normal(0, 1, B); pert = SIG * nz * xin.sum()
        a = rho * a + z; A = v_th + beta_a * a
        v = alpha * v + (drive[t - 1] if t > 0 else 0.0) - v_th * z + pert; xt = alpha * xt + pert
        ref = np.maximum(ref - 1, 0); refr = ref > 0
        z = ((v >= A) & ~refr).astype(float); zcf = ((v - xt >= A) & ~refr).astype(float)
        ref = np.where(z > 0, n_ref, ref)
        flip = z - zcf; nflip += np.abs(flip).sum()
        y = cc.kappa_y * y + z; Rt = 0.0
        if t in targets: thr, sgn = targets[t]; Rt = sgn * np.tanh((y - thr) / 0.5)
        r = cc.kappa_r * r + Rt; F += r
        zf = alpha * zf + xin
        Sflip.append(flip[:, None] * zf[None, :]); R.append(r.copy())
    return np.array(Sflip), np.array(R), F, nflip / (B * T)

def train(sign, centred=False, steps=30, B=3000, lr=3e-3, seed=0):
    rng = np.random.default_rng(seed); w = cc.ev.w.copy(); m_ = np.zeros(D); s_ = np.zeros(D); h = []; fr = []
    for k in range(steps):
        S, R, F, f = sim(w, B, rng); h.append(F.mean()); fr.append(f)
        RL = np.zeros((B, D)); DG = np.zeros(D)
        for t in range(T):
            RL = LAM * RL + S[t]; rew = R[t] - (R[t].mean() if centred else 0.0)
            DG += (rew[:, None] * RL).mean(0)
        DG /= T
        m_ = 0.9 * m_ + 0.1 * DG; s_ = 0.999 * s_ + 0.001 * (DG - m_)**2 + 1e-18
        w = w + sign * lr * m_ / (np.sqrt(s_ / (1 - 0.999**(k + 1))) + 1e-12)     # sign=+1: ascend R*pre*flip
    h.append(sim(w, B, rng)[2].mean()); return np.array(h), np.mean(fr)

for sign, centred, name in ((+1, False, "flip, +sign, raw reward"), (-1, False, "flip, -sign, raw reward"), (+1, True, "flip, +sign, centred (ref)")):
    h, f = train(sign, centred)
    print(f"{name:28s} E[F] {h[0]:+.2f} -> {h[-1]:+.2f}   ({np.round(h[::10], 2)})   flips per neuron-step {f:.4f}", flush=True)
