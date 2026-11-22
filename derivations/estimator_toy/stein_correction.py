"""Exact online drift correction via Stein's lemma, task reward, correct sign, NO baseline.
corrected score = (zbar - beta*eps_a) * [psi(v)*xi_ji - Cov(xi_ji, xi_j)*psi'(v)] / sigma^2
Placements: weight dist (shared noise) with exact Cov (uses the per-step sum of active sigmas)
and with an approximate Cov (sum replaced by its running mean); node dist (Cov = Var(xi_j))."""
import numpy as np
import centering_check as cc
D, T, X, targets = cc.D, cc.T, cc.X, cc.targets
alpha, rho, v_th, beta_a, gamma_pd, n_ref, LAM = cc.alpha, cc.rho, cc.v_th, cc.beta_a, cc.gamma_pd, cc.n_ref, cc.LAM
SIG_S, SIG_N = cc.SIG, np.exp(-4.61)

def batch(w, B, rng, mode, state):
    drive = w @ X
    v = np.zeros(B); a = np.zeros(B); z = np.zeros(B); ref = np.zeros(B); y = np.zeros(B); r = np.zeros(B); F = np.zeros(B)
    zf = np.zeros(D); ea = np.zeros((B, D)); xi = np.zeros((B, D)); xin_ = np.zeros(B)
    cov = np.zeros(D); RL = np.zeros((B, D)); DG = np.zeros((B, D)); nbar = 0.0 if isinstance(state, tuple) else state
    node = mode.startswith("node")
    for t in range(T):
        xin = X[:, t - 1] if t > 0 else np.zeros(D); a = rho * a + z; A = v_th + beta_a * a
        if node:
            eps = rng.normal(0, SIG_N, B); pert = eps
            xin_ = alpha * xin_ + eps                          # neuron noise trace xi_j
            cov = alpha**2 * cov + SIG_N**2                    # Var(xi_j) (same for all synapses)
            xi_use = np.broadcast_to(xin_[:, None], (B, D)); s2 = SIG_N**2
        else:
            nz = rng.normal(0, 1, B)[:, None]; applied = SIG_S * nz * xin[None, :]; pert = applied.sum(1)
            xi = alpha * xi + applied; xi_use = xi; s2 = SIG_S**2
            S = SIG_S * xin.sum()                              # sum of active sigmas this step
            if mode == "weight_approx":
                nbar = 0.99 * nbar + 0.01 * S; S = nbar         # running-mean approximation
            cov = alpha**2 * cov + SIG_S * xin * S             # Cov(xi_ji, xi_j)
        v = alpha * v + (drive[t - 1] if t > 0 else 0.0) - v_th * z + pert
        ref = np.maximum(ref - 1, 0); refr = ref > 0; z = ((v >= A) & ~refr).astype(float); ref = np.where(z > 0, n_ref, ref)
        y = cc.kappa_y * y + z; R = 0.0
        if t in targets: thr, sgn = targets[t]; R = sgn * np.tanh((y - thr) / 0.5)
        r = cc.kappa_r * r + R; F += r
        d = v - A
        inwin = (np.abs(d) < v_th) & ~refr
        psi = np.where(inwin, (gamma_pd / v_th) * (1 - np.abs(d) / v_th), 0.0)
        dpsi = np.where(inwin, -(gamma_pd / v_th**2) * np.sign(d), 0.0)      # psi'(v)
        zf = alpha * zf + xin
        pre = zf[None, :] - beta_a * ea
        ea = psi[:, None] * zf[None, :] + (rho - psi[:, None] * beta_a) * ea
        corr = 0.0 if mode == "uncorrected" else dpsi[:, None] * cov[None, :]
        score = pre * (psi[:, None] * xi_use - corr) / s2
        RL = LAM * RL + score
        DG += -1 * r[:, None] * RL                               # correct sign, uncentred reward
    return DG.mean(0) / T, F.mean(), nbar

cc.batch = batch
for mode in ("uncorrected", "weight_exact", "weight_approx", "node"):
    h = cc.train(mode)
    print(f"{mode:13s} E[F] {h[0]:+.2f} -> {h[-1]:+.2f}   ({np.round(h[::10], 2)})", flush=True)
