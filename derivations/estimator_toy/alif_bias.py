"""Bias / variance of perturbation-based gradient estimators in a single ALIF neuron.

Dynamics (per simulation step, noise accumulates in the membrane):
    a^t   = rho a^{t-1} + z^{t-1}                    (adaptation)
    A^t   = v_th + beta a^t                           (adaptive threshold)
    v^t   = alpha v^{t-1} + sum_i w_i x_i^{t-1} - v_th z^{t-1} + eps^t
    z^t   = H(v^t - A^t), blocked for n_ref steps after a spike
    psi^t = (gamma_pd / v_th) max(0, 1 - |v^t - A^t| / v_th), 0 while refractory

Reward: F = sum_t c_t z^t (whole trajectory, no TD / critic), so only the score
function is tested. Truth: central finite differences with common random numbers.

Estimators of dE[F]/dw_i (each multiplied by F - baseline):
  dist_eprop  : sum_t (1-alpha^2) xi^t / s2 * e_i^t      (e = psi (eps_v - beta eps_a))
  trace_nopsi : sum_t (1-alpha^2) xi^t / s2 * eps_v_i^t   (eps_v = presynaptic trace)
  step_pretr  : sum_t eps^t / s2 * eps_v_i^t
  step_psi    : sum_t eps^t / s2 * e_i^t
  exact       : sum_t eps^t / s2 * x_i^{t-1}
"""
import numpy as np, time, sys

alpha = np.exp(-1 / 10); rho = np.exp(-1 / 300)
v_th, beta, gamma_pd, n_ref = 0.61, 0.0174, 0.3, 3
EST = ["dist_eprop", "trace_nopsi", "step_pretr", "step_psi", "exact"]

def make_inputs(rng, D, T):
    X = np.zeros((D, T))
    X[0] = rng.binomial(1, 0.15, T)                       # Poisson-like
    X[1] = (np.arange(T) % 6 == 0)                        # regular
    X[2] = (np.arange(T) > T // 2) * rng.binomial(1, 0.4, T)   # late burst
    X[3] = (np.arange(T) < T // 3) * rng.binomial(1, 0.4, T)   # early burst
    X[4] = rng.binomial(1, 0.05, T)                       # sparse
    X[5] = rng.binomial(1, 0.3, T)                        # dense
    return X.astype(float)

def simulate(Wset, X, eps, need_traces):
    """Wset: (M, D) weight sets simulated with the same noise eps (B, T).
    Returns F-ready spike trains (M, B, T) and, if need_traces, estimator sums for set 0."""
    M, D = Wset.shape; B, T = eps.shape
    drive = Wset @ X                                       # (M, T): input at step t uses x^{t-1}
    v = np.zeros((M, B)); a = np.zeros((M, B)); z = np.zeros((M, B)); ref = np.zeros((M, B))
    Z = np.zeros((M, B, T))
    if need_traces:
        xi = np.zeros(B); ev = np.zeros(D); ea = np.zeros((B, D))
        S = {k: np.zeros((B, D)) for k in EST}
    for t in range(T):
        xin = X[:, t - 1] if t > 0 else np.zeros(D)
        a = rho * a + z
        A = v_th + beta * a
        inp = drive[:, t - 1][:, None] if t > 0 else 0.0
        v = alpha * v + inp - v_th * z + eps[None, :, t]
        ref = np.maximum(ref - 1, 0)
        refractory = ref > 0                       # refractory state before this step's spike
        z = ((v >= A) & ~refractory).astype(float)
        ref = np.where(z > 0, n_ref, ref)
        Z[:, :, t] = z
        if need_traces:
            psi = (gamma_pd / v_th) * np.maximum(0, 1 - np.abs(v[0] - A[0]) / v_th)
            psi = np.where(refractory[0], 0.0, psi)          # zero while refractory
            ev = alpha * ev + xin                                   # membrane eligibility (presynaptic trace)
            e = psi[:, None] * (ev[None, :] - beta * ea)            # e-prop ALIF eligibility
            xi = alpha * xi + eps[:, t]
            sc_tr = (1 - alpha**2) * xi / s2
            sc_st = eps[:, t] / s2
            S["dist_eprop"] += sc_tr[:, None] * e
            S["trace_nopsi"] += sc_tr[:, None] * ev[None, :]
            S["step_pretr"] += sc_st[:, None] * ev[None, :]
            S["step_psi"] += sc_st[:, None] * e
            S["exact"] += sc_st[:, None] * xin[None, :]
            ea = psi[:, None] * ev[None, :] + (rho - psi[:, None] * beta) * ea
    return Z, (S if need_traces else None)

def run(label, sigma, c, N, B=20000, D=6, T=120, h=0.02, seed=0):
    global s2
    s2 = sigma**2
    rng = np.random.default_rng(seed)
    X = make_inputs(rng, D, T)
    w = np.full(D, 0.25)
    Wset = np.vstack([w] + [w + h * np.eye(D)[d] for d in range(D)] + [w - h * np.eye(D)[d] for d in range(D)])
    fd = np.zeros(D); m = {k: np.zeros(D) for k in EST}; sq = {k: np.zeros(D) for k in EST}
    fdh = [np.zeros(D), np.zeros(D)]; mh = {k: [np.zeros(D), np.zeros(D)] for k in EST}; ib = 0
    rate = 0.0; nb = N // B; t0 = time.time()
    for _ in range(nb):
        eps = rng.normal(0, sigma, (B, T))
        Z, S = simulate(Wset, X, eps, True)
        F = Z @ c                                        # (M, B)
        fdb = ((F[1:1 + D] - F[1 + D:]) / (2 * h)).mean(1); fd += fdb; half = ib % 2; fdh[half] += fdb
        Fc = F[0] - F[0].mean()
        for k in EST:
            est = Fc[:, None] * S[k]
            m[k] += est.mean(0); sq[k] += (est**2).mean(0); mh[k][half] += est.mean(0)
        ib += 1
        rate += Z[0].mean()
    fd /= nb; rate /= nb
    cos = lambda p, q: p @ q / np.linalg.norm(p) / np.linalg.norm(q)
    print(f"\n== {label}: sigma={sigma}, firing prob/step={rate:.3f}, N={N}, {time.time()-t0:.0f}s")
    print(f"   truth (finite diff.)      : {np.round(fd, 4)}")
    for k in EST:
        mk = m[k] / nb; var = sq[k] / nb - mk**2
        rel_noise = np.sqrt(var.sum()) / np.linalg.norm(mk)
        c1, c2 = cos(mh[k][0], fdh[0]), cos(mh[k][1], fdh[1])
        print(f"   {k:12s} cos={cos(mk, fd):6.3f} (halves {c1:5.3f}/{c2:5.3f})  scale={mk @ fd / (fd @ fd):8.2f}  rel.noise/sample={rel_noise:8.1f}  mean={np.round(mk, 4)}")

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000
    T = 120
    t = np.arange(T)
    c_slow = np.where((t > 40) & (t < 80), 1.0, -0.3)                  # reward spikes in a window
    c_jag = np.random.default_rng(7).normal(0, 1, T)                   # jagged per-step reward
    for sigma in (0.05, 0.02):
        run("slow reward (window)", sigma, c_slow, N)
        run("jagged reward", sigma, c_jag, N)
