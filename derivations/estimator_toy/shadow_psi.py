"""Drift removal by construction: psi (and the e-prop eligibility) computed on a noise-free
shadow membrane that receives the same synaptic input, spikes and resets on its own, but
never sees the neuron's own perturbation. Correct sign, UNCENTRED task reward, weight-dist
noise as in the code. Compared with the current code and the per-moment baseline reference."""
import numpy as np
import centering_check as cc
D, T, X, targets = cc.D, cc.T, cc.X, cc.targets
alpha, rho, v_th, beta_a, gamma_pd, n_ref, LAM, SIG = cc.alpha, cc.rho, cc.v_th, cc.beta_a, cc.gamma_pd, cc.n_ref, cc.LAM, cc.SIG

def batch(w, B, rng, mode, state):
    drive = w @ X
    v = np.zeros(B); a = np.zeros(B); z = np.zeros(B); ref = np.zeros(B)
    vs = np.zeros(B); as_ = np.zeros(B); zs = np.zeros(B); refs = np.zeros(B)      # shadow neuron
    y = np.zeros(B); r = np.zeros(B); F = np.zeros(B)
    zf = np.zeros(D); ea = np.zeros((B, D)); xi = np.zeros((B, D)); RL = np.zeros((B, D)); DG = np.zeros((B, D))
    for t in range(T):
        xin = X[:, t - 1] if t > 0 else np.zeros(D); inp = drive[t - 1] if t > 0 else 0.0
        nz = rng.normal(0, 1, B)[:, None]; applied = SIG * nz * xin[None, :]
        # real neuron (noisy)
        a = rho * a + z; A = v_th + beta_a * a
        v = alpha * v + inp - v_th * z + applied.sum(1)
        ref = np.maximum(ref - 1, 0); refr = ref > 0; z = ((v >= A) & ~refr).astype(float); ref = np.where(z > 0, n_ref, ref)
        # shadow neuron (same input, no own noise)
        as_ = rho * as_ + zs; As = v_th + beta_a * as_
        vs = alpha * vs + inp - v_th * zs
        refs = np.maximum(refs - 1, 0); refrs = refs > 0; zs = ((vs >= As) & ~refrs).astype(float); refs = np.where(zs > 0, n_ref, refs)
        y = cc.kappa_y * y + z; R = 0.0
        if t in targets: thr, sgn = targets[t]; R = sgn * np.tanh((y - thr) / 0.5)
        r = cc.kappa_r * r + R; F += r
        if mode.startswith("shadow"):
            psi = np.where(refrs, 0.0, (gamma_pd / v_th) * np.maximum(0, 1 - np.abs(vs - As) / v_th))
        else:
            psi = np.where(refr, 0.0, (gamma_pd / v_th) * np.maximum(0, 1 - np.abs(v - A) / v_th))
        zf = alpha * zf + xin
        e = psi[:, None] * (zf[None, :] - beta_a * ea); ea = psi[:, None] * zf[None, :] + (rho - psi[:, None] * beta_a) * ea
        xi = alpha * xi + applied
        RL = LAM * RL + e * xi / SIG**2
        rew = r.copy()
        if mode.endswith("batch"): rew = rew - rew.mean()
        sign = +1 if mode in ("code", "shadow_codesign") else -1
        DG += sign * rew[:, None] * RL
    return DG.mean(0) / T, F.mean(), state

cc.batch = batch
for mode in ("code", "shadow", "shadow_codesign", "shadow_batch"):
    h = cc.train(mode)
    print(f"{mode:16s} E[F] {h[0]:+.2f} -> {h[-1]:+.2f}   ({np.round(h[::10], 2)})", flush=True)
