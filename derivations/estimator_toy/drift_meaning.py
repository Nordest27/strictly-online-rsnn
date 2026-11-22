"""What does the drift (mean score) estimate? Compare, per synapse, the time-summed mean of the
code's score  m_ji = E[ sum_t e_ji^t * NoiseTrace_ji^t / SynSig^2 ]  with the gradient of the
expected spike count  dE[sum_t z_t]/dW_ji  (exact estimator, unbiased). Prediction: -m is aligned
with the firing-rate gradient (the drift ~ uniform broadcast e-prop: 'reward -> fire more')."""
import numpy as np
import final_tables as ft
ev = ft.ev
alpha, rho, v_th, beta_a, gamma_pd, n_ref = ft.alpha, ft.rho, ft.v_th, ft.beta, ft.gamma_pd, ft.n_ref
D, T, X, SIG = ft.D, ft.T, ft.X, ft.SIG_S
cos = lambda p, q: p @ q / np.linalg.norm(p) / np.linalg.norm(q)

def measure(w, N=200_000, B=20000, seed=0):
    rng = np.random.default_rng(seed); drive = w @ X
    M = np.zeros(D); G = np.zeros(D); n = 0; fire = 0.0
    for _ in range(N // B):
        v = np.zeros(B); a = np.zeros(B); z = np.zeros(B); ref = np.zeros(B); cnt = np.zeros(B)
        zf = np.zeros(D); ea = np.zeros((B, D)); xi = np.zeros((B, D)); Sc = np.zeros((B, D)); Se = np.zeros((B, D))
        for t in range(T):
            xin = X[:, t - 1] if t > 0 else np.zeros(D); a = rho * a + z; A = v_th + beta_a * a
            nz = rng.normal(0, 1, B)[:, None]; applied = SIG * nz * xin[None, :]
            v = alpha * v + (drive[t - 1] if t > 0 else 0.0) - v_th * z + applied.sum(1)
            ref = np.maximum(ref - 1, 0); refr = ref > 0; z = ((v >= A) & ~refr).astype(float); ref = np.where(z > 0, n_ref, ref)
            cnt += z; fire += z.mean()
            psi = np.where(refr, 0.0, (gamma_pd / v_th) * np.maximum(0, 1 - np.abs(v - A) / v_th))
            zf = alpha * zf + xin; e = psi[:, None] * (zf[None, :] - beta_a * ea); ea = psi[:, None] * zf[None, :] + (rho - psi[:, None] * beta_a) * ea
            xi = alpha * xi + applied; Sc += e * xi / SIG**2
            s = SIG * xin.sum()
            if s > 0: Se += nz * xin[None, :] / s
        M += Sc.sum(0); G += ((cnt - cnt.mean())[:, None] * Se).sum(0); n += B
    return 100 * fire / (N // B * T), M / n, G / n

rng = np.random.default_rng(5); jit = rng.normal(0, 1, D)
for name, w in (("near threshold", ev.w.copy()), ("active", 0.08 + 0.75 * 0.08 * jit), ("sparser", 0.025 + 0.75 * 0.025 * jit)):
    fire, M, G = measure(w)
    print(f"{name:15s} fire {fire:4.1f}% | cos(-mean score, dE[spike count]/dW) = {cos(-M, G):+.3f}", flush=True)
