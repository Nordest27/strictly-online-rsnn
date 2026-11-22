"""Online drift removal (running means only, no batch statistics), task reward, single ALIF
neuron with the code's weight-dist update. Variants:
  code       : current code (uncentred reward, code sign)
  rew_ema    : correct sign, reward minus per-neuron running mean
  score_ema  : correct sign, score minus per-synapse running mean
  both       : correct sign, both centred
  batch      : correct sign, batch-centred reward (reference; not available online)"""
import numpy as np
import final_tables as ft
ev = ft.ev; D, T, X, targets = ft.D, ft.T, ft.X, ft.targets
kappa_y, kappa_r = ft.kappa_y, ft.kappa_r
alpha, rho, v_th, beta_a, gamma_pd, n_ref, SIG = ft.alpha, ft.rho, ft.v_th, ft.beta, ft.gamma_pd, ft.n_ref, ft.SIG_S
LAM = 0.8 ** (1 / 30); BETA = 0.98            # running-mean decay (~50 steps)

def batch(w, B, rng, mode, state):
    drive = w @ X
    v = np.zeros(B); a = np.zeros(B); z = np.zeros(B); ref = np.zeros(B); y = np.zeros(B); r = np.zeros(B); F = np.zeros(B)
    zf = np.zeros(D); ea = np.zeros((B, D)); xi = np.zeros((B, D)); RL = np.zeros((B, D)); DG = np.zeros((B, D))
    rbar, sbar = state                         # running means carried across training steps (online)
    for t in range(T):
        xin = X[:, t - 1] if t > 0 else np.zeros(D); a = rho * a + z; A = v_th + beta_a * a
        nz = rng.normal(0, 1, B)[:, None]; applied = SIG * nz * xin[None, :]
        v = alpha * v + (drive[t - 1] if t > 0 else 0.0) - v_th * z + applied.sum(1)
        ref = np.maximum(ref - 1, 0); refr = ref > 0; z = ((v >= A) & ~refr).astype(float); ref = np.where(z > 0, n_ref, ref)
        y = kappa_y * y + z; R = 0.0
        if t in targets: thr, sgn = targets[t]; R = sgn * np.tanh((y - thr) / 0.5)
        r = kappa_r * r + R; F += r
        psi = np.where(refr, 0.0, (gamma_pd / v_th) * np.maximum(0, 1 - np.abs(v - A) / v_th))
        zf = alpha * zf + xin; e = psi[:, None] * (zf[None, :] - beta_a * ea); ea = psi[:, None] * zf[None, :] + (rho - psi[:, None] * beta_a) * ea
        xi = alpha * xi + applied
        score = e * xi / SIG**2
        if mode in ("score_ema", "both"):
            sbar = BETA * sbar + (1 - BETA) * score; score = score - sbar
        RL = LAM * RL + score
        rew = r.copy()
        if mode in ("rew_ema", "both"):
            rbar = BETA * rbar + (1 - BETA) * rew; rew = rew - rbar
        if mode == "batch": rew = rew - rew.mean()
        sign = +1 if mode == "code" else -1
        DG += sign * rew[:, None] * RL
    return DG.mean(0) / T, F.mean(), (rbar, sbar)

def train(mode, steps=30, B=3000, lr=3e-3, seed=0):
    rng = np.random.default_rng(seed); w = ev.w.copy(); m = np.zeros(D); s = np.zeros(D); h = []
    state = (np.zeros(B), np.zeros((B, D)))
    for k in range(steps):
        DG, f, state = batch(w, B, rng, mode, state); h.append(f)
        m = 0.9 * m + 0.1 * DG; s = 0.999 * s + 0.001 * (DG - m)**2 + 1e-18
        w = w - lr * m / (np.sqrt(s / (1 - 0.999**(k + 1))) + 1e-12)
    h.append(batch(w, B, rng, mode, state)[1]); return np.array(h)

if __name__ == "__main__":
  for mode in ("code", "rew_ema", "score_ema", "both", "batch"):
    h = train(mode)
    print(f"{mode:10s} E[F] {h[0]:+.2f} -> {h[-1]:+.2f}   ({np.round(h[::10], 2)})", flush=True)
