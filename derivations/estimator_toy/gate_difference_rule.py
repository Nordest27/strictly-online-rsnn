"""Clean drift estimator: d_t = (psi_real - psi_shadow) * xi / sigma^2  (shadow = noise-free copy).
1) Estimator quality: per-moment relative noise of the code's score vs the gate-difference estimate,
   both as estimators of the per-moment drift m_t (= batch mean of the code's score).
2) Learning (task reward, code's sign for the drift): current rule vs oracle drift (exact m_t)
   vs gate-difference rule."""
import numpy as np
import centering_check as cc
D, T, X, targets = cc.D, cc.T, cc.X, cc.targets
alpha, rho, v_th, beta_a, gamma_pd, n_ref, LAM, SIG = cc.alpha, cc.rho, cc.v_th, cc.beta_a, cc.gamma_pd, cc.n_ref, cc.LAM, cc.SIG

def step_all(w, B, rng, record=None):
    """Simulate a batch; return per-step arrays needed by the rules."""
    drive = w @ X
    v = np.zeros(B); a = np.zeros(B); z = np.zeros(B); ref = np.zeros(B)
    vs = np.zeros(B); as_ = np.zeros(B); zs = np.zeros(B); refs = np.zeros(B)
    y = np.zeros(B); r = np.zeros(B); F = np.zeros(B)
    zf = np.zeros(D); ea = np.zeros((B, D)); xi = np.zeros((B, D))
    S_code, S_diff, R = [], [], []
    for t in range(T):
        xin = X[:, t - 1] if t > 0 else np.zeros(D); inp = drive[t - 1] if t > 0 else 0.0
        nz = rng.normal(0, 1, B)[:, None]; applied = SIG * nz * xin[None, :]
        a = rho * a + z; A = v_th + beta_a * a; v = alpha * v + inp - v_th * z + applied.sum(1)
        ref = np.maximum(ref - 1, 0); refr = ref > 0; z = ((v >= A) & ~refr).astype(float); ref = np.where(z > 0, n_ref, ref)
        as_ = rho * as_ + zs; As = v_th + beta_a * as_; vs = alpha * vs + inp - v_th * zs
        refs = np.maximum(refs - 1, 0); refrs = refs > 0; zs = ((vs >= As) & ~refrs).astype(float); refs = np.where(zs > 0, n_ref, refs)
        y = cc.kappa_y * y + z; Rt = 0.0
        if t in targets: thr, sgn = targets[t]; Rt = sgn * np.tanh((y - thr) / 0.5)
        r = cc.kappa_r * r + Rt; F += r
        psi = np.where(refr, 0.0, (gamma_pd / v_th) * np.maximum(0, 1 - np.abs(v - A) / v_th))
        psis = np.where(refrs, 0.0, (gamma_pd / v_th) * np.maximum(0, 1 - np.abs(vs - As) / v_th))
        zf = alpha * zf + xin; pre = zf[None, :] - beta_a * ea
        ea = psi[:, None] * zf[None, :] + (rho - psi[:, None] * beta_a) * ea
        xi = alpha * xi + applied
        S_code.append(pre * psi[:, None] * xi / SIG**2)
        S_diff.append(pre * (psi - psis)[:, None] * xi / SIG**2)
        R.append(r.copy())
    return np.array(S_code), np.array(S_diff), np.array(R), F

def quality(w, B=40000, seed=0):
    rng = np.random.default_rng(seed)
    Sc, Sd, _, _ = step_all(w, B, rng)
    m = Sc.mean(1)                                     # per-moment drift (T, D)
    def relnoise(S):
        dev = S - m[:, None, :]
        return np.sqrt((dev**2).mean(1).sum(1)) / (np.linalg.norm(m, axis=1) + 1e-30)
    rc, rd = relnoise(Sc)[20:], relnoise(Sd)[20:]
    bias = np.mean([Sd.mean(1)[t] @ m[t] / (np.linalg.norm(Sd.mean(1)[t]) * np.linalg.norm(m[t]) + 1e-30) for t in range(20, T)])
    return np.median(rc), np.median(rd), bias

def train(mode, steps=30, B=3000, lr=3e-3, seed=0):
    rng = np.random.default_rng(seed); w = cc.ev.w.copy(); m_ = np.zeros(D); s_ = np.zeros(D); h = []
    for k in range(steps):
        Sc, Sd, R, F = step_all(w, B, rng); h.append(F.mean())
        if mode == "code":   S = Sc
        elif mode == "oracle_drift": S = np.broadcast_to(Sc.mean(1, keepdims=True), Sc.shape)
        else:                S = Sd
        RL = np.zeros((B, D)); DG = np.zeros(D)
        for t in range(T):
            RL = LAM * RL + S[t]; DG += (R[t][:, None] * RL).mean(0)     # code sign: DeltaG += R * trace
        DG /= T
        m_ = 0.9 * m_ + 0.1 * DG; s_ = 0.999 * s_ + 0.001 * (DG - m_)**2 + 1e-18
        w = w - lr * m_ / (np.sqrt(s_ / (1 - 0.999**(k + 1))) + 1e-12)
    h.append(step_all(w, B, rng)[3].mean()); return np.array(h)

if __name__ == "__main__":
    rng = np.random.default_rng(5); jit = rng.normal(0, 1, D)
    for name, w in (("near threshold", cc.ev.w.copy()), ("active", 0.08 + 0.75 * 0.08 * jit)):
        qc, qd, b = quality(w)
        print(f"{name:15s} median relative noise as drift estimator: code score {qc:.1f}, gate difference {qd:.1f}; per-moment cosine of gate-difference mean to drift {b:+.3f}", flush=True)
    for mode in ("code", "oracle_drift", "gate_diff"):
        h = train(mode)
        print(f"{mode:13s} E[F] {h[0]:+.2f} -> {h[-1]:+.2f}   ({np.round(h[::10], 2)})", flush=True)
