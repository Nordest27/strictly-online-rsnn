"""Exact test: is the per-moment drift m_t (batch mean of the code's score at step t) equal to the
per-moment sensitivity of the expected gate, dE[psi_t]/dW_ji, computed with the unbiased exact
estimator (causal: noise up to t)? Reports, per regime, the correlation over all (t, synapse) pairs
and the mean per-moment cosine across synapses."""
import numpy as np
import final_tables as ft
ev = ft.ev
alpha, rho, v_th, beta_a, gamma_pd, n_ref = ft.alpha, ft.rho, ft.v_th, ft.beta, ft.gamma_pd, ft.n_ref
D, T, X, SIG = ft.D, ft.T, ft.X, ft.SIG_S

def run(w, B=200000, chunk=20000, seed=0):
    rng = np.random.default_rng(seed); drive = w @ X
    M = np.zeros((T, D)); G = np.zeros((T, D)); n = 0
    for _ in range(B // chunk):
        b = chunk
        v = np.zeros(b); a = np.zeros(b); z = np.zeros(b); ref = np.zeros(b)
        zf = np.zeros(D); ea = np.zeros((b, D)); xi = np.zeros((b, D)); Se = np.zeros((b, D))
        for t in range(T):
            xin = X[:, t - 1] if t > 0 else np.zeros(D); a = rho * a + z; A = v_th + beta_a * a
            nz = rng.normal(0, 1, b)[:, None]; applied = SIG * nz * xin[None, :]
            v = alpha * v + (drive[t - 1] if t > 0 else 0.0) - v_th * z + applied.sum(1)
            s = SIG * xin.sum()
            if s > 0: Se += nz * xin[None, :] / s                       # exact score up to t
            ref = np.maximum(ref - 1, 0); refr = ref > 0; z = ((v >= A) & ~refr).astype(float); ref = np.where(z > 0, n_ref, ref)
            psi = np.where(refr, 0.0, (gamma_pd / v_th) * np.maximum(0, 1 - np.abs(v - A) / v_th))
            zf = alpha * zf + xin; pre = zf[None, :] - beta_a * ea
            e = psi[:, None] * pre; ea = psi[:, None] * zf[None, :] + (rho - psi[:, None] * beta_a) * ea
            xi = alpha * xi + applied
            M[t] += (e * xi / SIG**2).sum(0)
            G[t] += ((psi - psi.mean())[:, None] * Se).sum(0) * pre.mean(0)  # dE[psi_t]/dW times presynaptic factor
        n += b
    M /= n; G /= n
    m, g = M[20:].ravel(), G[20:].ravel()
    corr = np.corrcoef(m, g)[0, 1]
    cosines = [M[t] @ G[t] / (np.linalg.norm(M[t]) * np.linalg.norm(G[t]) + 1e-30) for t in range(20, T)]
    return corr, np.mean(cosines)

rng = np.random.default_rng(5); jit = rng.normal(0, 1, D)
for name, w in (("near threshold", ev.w.copy()), ("active", 0.08 + 0.75 * 0.08 * jit)):
    corr, cmean = run(w)
    print(f"{name:15s} corr(drift, dE[psi_t]/dW x pre) over all (t, synapse) = {corr:+.3f}; mean per-moment cosine = {cmean:+.3f}", flush=True)
