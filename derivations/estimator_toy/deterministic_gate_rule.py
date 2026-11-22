"""Option 1: deterministic gate sensitivity instead of the noise probe.
g_t = pre_t * psi'(v_t) * S_t, with S the per-synapse membrane sensitivity including the reset path:
    S <- alpha*S + x_i - v_th * psi_{t-1} * S       (psi as the spike's surrogate derivative)
Variants: S_reset (above), S_plain (no reset path, = presynaptic trace), and psi'*zbar.
1) Per-moment correlation of the batch mean of each candidate with the drift m_t.
2) Learning with each candidate as the score (code sign, lambda trace, task reward), vs oracle drift."""
import numpy as np
import gate_difference_rule as gd
cc = gd.cc
D, T, X, targets = cc.D, cc.T, cc.X, cc.targets
alpha, rho, v_th, beta_a, gamma_pd, n_ref, LAM, SIG = cc.alpha, cc.rho, cc.v_th, cc.beta_a, cc.gamma_pd, cc.n_ref, cc.LAM, cc.SIG

def sim(w, B, rng):
    drive = w @ X
    v = np.zeros(B); a = np.zeros(B); z = np.zeros(B); ref = np.zeros(B); y = np.zeros(B); r = np.zeros(B); F = np.zeros(B)
    zf = np.zeros(D); ea = np.zeros((B, D)); xi = np.zeros((B, D)); S = np.zeros((B, D)); psi_prev = np.zeros(B)
    out = {k: [] for k in ("code", "reset", "plain")}; R = []
    for t in range(T):
        xin = X[:, t - 1] if t > 0 else np.zeros(D)
        nz = rng.normal(0, 1, B)[:, None]; applied = SIG * nz * xin[None, :]
        a = rho * a + z; A = v_th + beta_a * a; v = alpha * v + (drive[t - 1] if t > 0 else 0.0) - v_th * z + applied.sum(1)
        S = alpha * S + xin[None, :] - v_th * psi_prev[:, None] * S
        ref = np.maximum(ref - 1, 0); refr = ref > 0; z = ((v >= A) & ~refr).astype(float); ref = np.where(z > 0, n_ref, ref)
        y = cc.kappa_y * y + z; Rt = 0.0
        if t in targets: thr, sgn = targets[t]; Rt = sgn * np.tanh((y - thr) / 0.5)
        r = cc.kappa_r * r + Rt; F += r
        d = v - A; inwin = (np.abs(d) < v_th) & ~refr
        psi = np.where(inwin, (gamma_pd / v_th) * (1 - np.abs(d) / v_th), 0.0)
        dpsi = np.where(inwin, -(gamma_pd / v_th**2) * np.sign(d), 0.0)
        zf = alpha * zf + xin; pre = zf[None, :] - beta_a * ea
        ea = psi[:, None] * zf[None, :] + (rho - psi[:, None] * beta_a) * ea
        xi = alpha * xi + applied
        out["code"].append(pre * psi[:, None] * xi / SIG**2)
        out["reset"].append(pre * dpsi[:, None] * S)
        out["plain"].append(pre * dpsi[:, None] * zf[None, :])
        psi_prev = psi; R.append(r.copy())
    return {k: np.array(v) for k, v in out.items()}, np.array(R), F

def correlation(w, B=40000, seed=0):
    rng = np.random.default_rng(seed); o, _, _ = sim(w, B, rng)
    m = o["code"].mean(1)[20:].ravel()
    return {k: np.corrcoef(m, o[k].mean(1)[20:].ravel())[0, 1] for k in ("reset", "plain")}

def train(mode, steps=30, B=3000, lr=3e-3, seed=0):
    rng = np.random.default_rng(seed); w = cc.ev.w.copy(); m_ = np.zeros(D); s_ = np.zeros(D); h = []
    for k in range(steps):
        o, R, F = sim(w, B, rng); h.append(F.mean())
        S = np.broadcast_to(o["code"].mean(1, keepdims=True), o["code"].shape) if mode == "oracle_drift" else o[mode]
        scale = 1.0 if mode == "oracle_drift" else -1.0       # drift ~ -(something) * dpsi-type terms; sign fitted below
        RL = np.zeros((B, D)); DG = np.zeros(D)
        for t in range(T):
            RL = LAM * RL + S[t]; DG += (R[t][:, None] * RL).mean(0)
        DG /= T
        m_ = 0.9 * m_ + 0.1 * DG; s_ = 0.999 * s_ + 0.001 * (DG - m_)**2 + 1e-18
        w = w - lr * SIGN[mode] * m_ / (np.sqrt(s_ / (1 - 0.999**(k + 1))) + 1e-12)
    h.append(sim(w, B, rng)[2].mean()); return np.array(h)

rng = np.random.default_rng(5); jit = rng.normal(0, 1, D)
signs = {}
for name, w in (("near threshold", cc.ev.w.copy()), ("active", 0.08 + 0.75 * 0.08 * jit)):
    c = correlation(w); signs[name] = c
    print(f"{name:15s} per-moment correlation with the drift: psi'*S (with reset) {c['reset']:+.3f}, psi'*zbar (no reset) {c['plain']:+.3f}", flush=True)
# use the sign that aligns each deterministic candidate with the drift (from the baseline regime)
SIGN = {"oracle_drift": 1.0, "reset": float(np.sign(signs['near threshold']['reset'])), "plain": float(np.sign(signs['near threshold']['plain']))}
for mode in ("oracle_drift", "reset", "plain"):
    h = train(mode)
    print(f"{mode:13s} E[F] {h[0]:+.2f} -> {h[-1]:+.2f}   ({np.round(h[::10], 2)})", flush=True)
