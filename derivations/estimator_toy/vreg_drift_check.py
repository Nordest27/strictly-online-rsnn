"""Train a single ALIF neuron's input weights with ONLY the perturbation-credited voltage
regulariser, exactly as the code does it: per-step, uncentred local reward -c*VReg_t,
credited through the lambda-trace RLNoiseTrace (no baseline), DeltaG += (-c*VReg) * RLNoiseTrace,
optimiser w -= lr * m / sqrt(s). Code sign vs flipped. Track mean VReg (voltage loss)."""
import numpy as np
import final_tables as ft
ev = ft.ev
alpha, rho, v_th, beta_a, gamma_pd, n_ref = ft.alpha, ft.rho, ft.v_th, ft.beta, ft.gamma_pd, ft.n_ref
D, T, X, SIG = ft.D, ft.T, ft.X, ft.SIG_S
LAM = 0.8 ** (1 / 30); C = 1.0

def batch(w, B, rng, sign, centred):
    drive = w @ X
    v = np.zeros(B); a = np.zeros(B); z = np.zeros(B); ref = np.zeros(B)
    zf = np.zeros(D); ea = np.zeros((B, D)); xi = np.zeros((B, D)); RL = np.zeros((B, D)); DG = np.zeros((B, D)); vl = 0.0
    for t in range(T):
        xin = X[:, t - 1] if t > 0 else np.zeros(D)
        a = rho * a + z; A = v_th + beta_a * a
        nz = rng.normal(0, 1, B)[:, None]; applied = SIG * nz * xin[None, :]
        v = alpha * v + (drive[t - 1] if t > 0 else 0.0) - v_th * z + applied.sum(1)
        ref = np.maximum(ref - 1, 0); refr = ref > 0
        z = ((v >= A) & ~refr).astype(float); ref = np.where(z > 0, n_ref, ref)
        vpost = v - v_th * z
        vreg = np.maximum(0, vpost - A) + np.maximum(0, -vpost - A); vl += vreg.mean()
        psi = np.where(refr, 0.0, (gamma_pd / v_th) * np.maximum(0, 1 - np.abs(v - A) / v_th))
        zf = alpha * zf + xin
        e = psi[:, None] * (zf[None, :] - beta_a * ea); ea = psi[:, None] * zf[None, :] + (rho - psi[:, None] * beta_a) * ea
        xi = alpha * xi + applied
        RL = LAM * RL + e * xi / SIG**2                               # RLNoiseTrace
        rew = -C * vreg
        if centred: rew = rew - rew.mean()
        DG += sign * rew[:, None] * RL                                # code: DeltaG += reward * RLNoiseTrace
    return DG.mean(0) / T, vl / T

def train(sign, centred, w0, steps=30, B=3000, lr=3e-3, seed=0):
    rng = np.random.default_rng(seed); w = w0.copy(); m = np.zeros(D); s = np.zeros(D); hist = []
    for k in range(steps):
        DG, vl = batch(w, B, rng, sign, centred); hist.append(vl)
        m = 0.9 * m + 0.1 * DG; s = 0.999 * s + 0.001 * (DG - m)**2 + 1e-18
        w = w - lr * m / (np.sqrt(s / (1 - 0.999**(k + 1))) + 1e-12)
    hist.append(batch(w, B, rng, sign, centred)[1])
    return np.array(hist)

rng = np.random.default_rng(5); jit = rng.normal(0, 1, D)
w0 = 0.08 + 0.75 * 0.08 * jit                 # active regime: the membrane overshoots
for centred in (False, True):
    for sign, name in ((+1, "code sign "), (-1, "flipped   ")):
        h = train(sign, centred, w0)
        print(f"{'centred  ' if centred else 'uncentred'} reward, {name}: voltage loss {h[0]:.4f} -> {h[-1]:.4f}   ({np.round(h[::10], 4)})", flush=True)
