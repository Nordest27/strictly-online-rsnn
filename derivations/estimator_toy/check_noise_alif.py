"""Predict relative noise and direction of the five estimators in the ALIF toy
(jagged reward, sigma = exp(-5)) from the small-noise closed form, with psi and the
eligibility evaluated on the noise-free trajectory. Reward sensitivity to each
innovation, g_eps[s] = E[(F - EF) eps^s] / sigma^2, is estimated by Monte Carlo in two
independent halves (A, B) so that its sampling noise does not bias |g|^2 or (g.a)^2.
Compare with the measured values from alif_bias_mp.py (jagged reward)."""
import numpy as np, time
import alif_bias_mp as m
alpha, rho, v_th, beta, gamma_pd, n_ref = m.alpha, m.rho, m.v_th, m.beta, m.gamma_pd, m.n_ref
D, T, SIG = m.D, m.T, m.SIGMA
X, w, rewards = m.setup(); c = rewards["jagged"]

def sim(eps):                                   # eps (B, T); returns F and (for B==1 deterministic) traces
    B = eps.shape[0]; drive = w @ X
    v = np.zeros(B); a = np.zeros(B); z = np.zeros(B); ref = np.zeros(B); F = np.zeros(B)
    ev = np.zeros(D); ea = np.zeros((B, D)); PSI = np.zeros((B, T)); EV = np.zeros((T, D)); E = np.zeros((B, T, D)); XIN = np.zeros((T, D))
    for t in range(T):
        xin = X[:, t - 1] if t > 0 else np.zeros(D)
        a = rho * a + z; A = v_th + beta * a
        v = alpha * v + (drive[t - 1] if t > 0 else 0.0) - v_th * z + eps[:, t]
        ref = np.maximum(ref - 1, 0); refractory = ref > 0
        z = ((v >= A) & ~refractory).astype(float); ref = np.where(z > 0, n_ref, ref)
        F += c[t] * z
        psi = np.where(refractory, 0.0, (gamma_pd / v_th) * np.maximum(0, 1 - np.abs(v - A) / v_th))
        ev = alpha * ev + xin
        e = psi[:, None] * (ev[None, :] - beta * ea)
        ea = psi[:, None] * ev[None, :] + (rho - psi[:, None] * beta) * ea
        PSI[:, t] = psi; EV[t] = ev; E[:, t] = e; XIN[t] = xin
    return F, PSI, EV, E, XIN

# 1) deterministic (noise-free) trajectory -> profiles on innovations
_, psi0, EV, E0, XIN = sim(np.zeros((1, T))); E0 = E0[0]
s2 = SIG**2; cst = (1 - alpha**2) / s2
Fw = np.tril(alpha ** np.clip(np.subtract.outer(np.arange(T), np.arange(T)), 0, None))   # Fw[t, s] = alpha^{t-s}, t>=s
fwd = lambda Y: Fw.T @ Y                       # sum_{t>=s} alpha^{t-s} Y[t]
prof = {"dist_eprop": cst * fwd(E0), "trace_nopsi": cst * fwd(EV), "step_pretr": EV / s2,
        "step_psi": E0 / s2, "exact": XIN / s2}          # each (T, D): a_i[s] = prof[s, i]

# 2) Monte-Carlo estimate of g_eps in two halves
rng = np.random.default_rng(99); B, N = 25000, 2_000_000; gA = np.zeros(T); gB = np.zeros(T); nA = nB = 0; t0 = time.time()
for i in range(N // B):
    eps = rng.normal(0, SIG, (B, T)); F = sim(eps)[0]; Fc = F - F.mean()
    g = (Fc[:, None] * eps).sum(0) / s2
    if i % 2: gB += g; nB += B
    else: gA += g; nA += B
gA /= nA; gB /= nB
print(f"g_eps estimated from {N:,} samples in {time.time()-t0:.0f}s; halves cos = {gA @ gB / np.linalg.norm(gA) / np.linalg.norm(gB):.3f}")

# 3) predictions (cross-half products remove sampling-noise bias)
gg = gA @ gB
truth = s2 * 0.5 * (prof["exact"].T @ gA + prof["exact"].T @ gB)
measured = {"dist_eprop": (0.993, 2.6), "trace_nopsi": (0.982, 2.5), "step_pretr": (0.980, 2.9), "step_psi": (0.994, 3.4), "exact": (1.000, 4.0)}
print(f"{'estimator':12s} {'cos pred':>9s} {'cos meas':>9s} {'rel.noise pred':>15s} {'rel.noise meas':>15s}")
for k, a in prof.items():
    ga = (a.T @ gA) * (a.T @ gB)                 # (g.a_i)^2, unbiased via halves
    mean = s2 * 0.5 * (a.T @ gA + a.T @ gB)
    cos = mean @ truth / np.linalg.norm(mean) / np.linalg.norm(truth)
    rel = np.sqrt(1 + gg * (a**2).sum() / ga.sum())
    print(f"{k:12s} {cos:9.3f} {measured[k][0]:9.3f} {rel:15.2f} {measured[k][1]:15.1f}")
