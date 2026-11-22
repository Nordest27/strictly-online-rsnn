"""Noise placement: node dist vs weight dist (shared post noise) vs independent synaptic noise.
Mirrors the ml_genn implementation:
  node dist   : membrane noise every step, eps ~ N(0, sigma_n^2), neuron trace xi_j,
                score (1-alpha^2) xi_j / sigma_n^2 * e_ji
  weight dist : one N(0,1) draw per neuron per step, applied through synapse i only on a
                presynaptic spike as SynSig_i * Noise; per-synapse trace NoiseTrace_ji,
                score (1-alpha^2) NoiseTrace_ji / SynSig_i^2 * e_ji
  ind noise   : as weight dist but an independent draw per synapse
Exact (unbiased) estimators:
  node        : eps / sigma_n^2 * z_i
  weight dist : Noise * z_i / sum_k SynSig_k z_k      (step-varying membrane variance)
  ind noise   : Noise_i * z_i / SynSig_i
Reward: sparse, decaying, readout-dependent events; causal credit with a lambda trace.
Truth for each placement: exact estimator with cumulative causal credit (unbiased).
Usage: python3 noise_placement.py N_SAMPLES"""
import sys, time, numpy as np
import alif_bias_events as ev

alpha, rho, v_th, beta, gamma_pd, n_ref = ev.alpha, ev.rho, ev.v_th, ev.beta, ev.gamma_pd, ev.n_ref
D, T, X, targets = ev.D, ev.T, ev.X, ev.targets
kappa_y, kappa_r = ev.kappa_y, ev.kappa_r
LAM = 0.8 ** (1 / 30)
SIG_N, SIG_S = np.exp(-4.61), np.exp(-5.0)

def run(w, placement, N, B=25000, seed=0):
    rng = np.random.default_rng(seed); drive = w @ X
    names = ["trace_rule", "exact"]
    M1 = {k: np.zeros(D) for k in names}; M2 = {k: 0.0 for k in names}; Mt = np.zeros(D); n = 0
    fire = open_ = 0.0
    for _ in range(N // B):
        v = np.zeros(B); a = np.zeros(B); z = np.zeros(B); ref = np.zeros(B); y = np.zeros(B); r = np.zeros(B)
        zf = np.zeros(D); ea = np.zeros((B, D))
        xi_n = np.zeros(B); xi_s = np.zeros((B, D))
        C = {k: np.zeros((B, D)) for k in names}; Ccum = np.zeros((B, D))
        E = {k: np.zeros((B, D)) for k in names}; Et = np.zeros((B, D))
        for t in range(T):
            xin = X[:, t - 1] if t > 0 else np.zeros(D)
            a = rho * a + z; A = v_th + beta * a
            if placement == "node":
                eps = rng.normal(0, SIG_N, B); pert = eps
                xi_n = alpha * xi_n + eps
                sc_tr = ((1 - alpha**2) * xi_n / SIG_N**2)[:, None]
                sc_ex = (eps / SIG_N**2)[:, None] * xin[None, :]
            else:
                if placement == "weight":
                    nz = rng.normal(0, 1, B)[:, None] * np.ones((1, D))
                else:
                    nz = rng.normal(0, 1, (B, D))
                applied = SIG_S * nz * xin[None, :]           # only on presynaptic spikes
                pert = applied.sum(1)
                xi_s = alpha * xi_s + applied                  # NoiseTrace += eps; ... NoiseTrace *= Alpha
                sc_tr = (1 - alpha**2) * xi_s / SIG_S**2
                if placement == "weight":
                    s = SIG_S * xin.sum()
                    sc_ex = (nz * xin[None, :] / s) if s > 0 else np.zeros((B, D))
                else:
                    sc_ex = nz * xin[None, :] / SIG_S
            v = alpha * v + (drive[t - 1] if t > 0 else 0.0) - v_th * z + pert
            ref = np.maximum(ref - 1, 0); refractory = ref > 0
            z = ((v >= A) & ~refractory).astype(float); ref = np.where(z > 0, n_ref, ref)
            y = kappa_y * y + z
            R = 0.0
            if t in targets:
                thr, sgn = targets[t]; R = sgn * np.tanh((y - thr) / 0.5)
            r = kappa_r * r + R
            psi = np.where(refractory, 0.0, (gamma_pd / v_th) * np.maximum(0, 1 - np.abs(v - A) / v_th))
            zf = alpha * zf + xin
            e = psi[:, None] * (zf[None, :] - beta * ea)
            ea = psi[:, None] * zf[None, :] + (rho - psi[:, None] * beta) * ea
            C["trace_rule"] = LAM * C["trace_rule"] + sc_tr * e
            C["exact"] = LAM * C["exact"] + sc_ex
            Ccum += sc_ex
            rc = (r - r.mean())[:, None]
            for k in names: E[k] += rc * C[k]
            Et += rc * Ccum
            fire += z.mean(); open_ += (psi > 0).mean()
        for k in names: M1[k] += E[k].sum(0); M2[k] += (E[k]**2).sum()
        Mt += Et.sum(0); n += B
    steps = (N // B) * T
    truth = Mt / n
    out = {}
    for k in names:
        mk = M1[k] / n
        out[k] = (mk @ truth / np.linalg.norm(mk) / np.linalg.norm(truth), np.sqrt(M2[k] / n - mk @ mk) / np.linalg.norm(mk))
    return 100 * fire / steps, 100 * open_ / steps, out

if __name__ == "__main__":
    N = int(float(sys.argv[1])) if len(sys.argv) > 1 else 200_000
    rng = np.random.default_rng(5); jitter = rng.normal(0, 1, D)
    regimes = {"baseline": ev.w.copy(), "saturated": 0.15 + 0.75 * 0.15 * jitter}
    w_inh = 0.12 + 0.75 * 0.12 * jitter; w_inh[6:] = -0.2; regimes["inhibited"] = w_inh
    print("cells: cosine to truth / relative noise")
    for name, w in regimes.items():
        for pl in ("node", "weight", "ind"):
            t0 = time.time(); fire, op, out = run(w, pl, N)
            print(f"{name:10s} {pl:6s} fire={fire:5.1f}% psi open={op:5.1f}% | "
                  + " | ".join(f"{k}: {c:6.3f}/{rn:6.1f}" for k, (c, rn) in out.items()) + f"  [{time.time()-t0:.0f}s]", flush=True)
