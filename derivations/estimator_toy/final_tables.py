"""Final tables for the appendix: one common setting.
Setting: single ALIF neuron, sparse decaying readout-dependent reward (alif_bias_events),
lambda credit trace, sigma from the paper (node e^-4.61, synaptic e^-5).
Truth: exact estimator of the given placement with cumulative causal credit (unbiased).
Main rule = weight dist as implemented (per-synapse trace, (1-alpha^2)/SynSig^2, e-prop e-trace).
Tables:
  A  five estimators, weight-dist placement, baseline regime
  B  weight-dist estimators across regimes (psi effect)
  C  trace rule vs exact for node / weight / ind placements, baseline and inhibited
Usage: python3 final_tables.py N_SAMPLES"""
import sys, time, numpy as np
import alif_bias_events as ev

alpha, rho, v_th, beta, gamma_pd, n_ref = ev.alpha, ev.rho, ev.v_th, ev.beta, ev.gamma_pd, ev.n_ref
D, T, X, targets = ev.D, ev.T, ev.X, ev.targets
kappa_y, kappa_r = ev.kappa_y, ev.kappa_r
LAM = 0.8 ** (1 / 30)
SIG_N, SIG_S = np.exp(-4.61), np.exp(-5.0)
FIVE = ["dist_eprop", "trace_nopsi", "step_psi", "step_pretr", "exact"]

def run(w, placement, N, B=25000, seed=0):
    rng = np.random.default_rng(seed); drive = w @ X
    names = FIVE if placement == "weight" else ["dist_eprop", "exact"]
    M1 = {k: np.zeros(D) for k in names}; M2 = {k: 0.0 for k in names}; Mt = np.zeros(D); Mt2 = 0.0; n = 0
    st = dict(fire=0.0, open=0.0, below=0.0, above=0.0)
    for _ in range(N // B):
        v = np.zeros(B); a = np.zeros(B); z = np.zeros(B); ref = np.zeros(B); y = np.zeros(B); r = np.zeros(B)
        zf = np.zeros(D); ea = np.zeros((B, D)); xi_n = np.zeros(B); xi_s = np.zeros((B, D))
        C = {k: np.zeros((B, D)) for k in names}; Ccum = np.zeros((B, D))
        E = {k: np.zeros((B, D)) for k in names}; Et = np.zeros((B, D))
        for t in range(T):
            xin = X[:, t - 1] if t > 0 else np.zeros(D)
            a = rho * a + z; A = v_th + beta * a
            if placement == "node":
                eps = rng.normal(0, SIG_N, B); pert = eps; xi_n = alpha * xi_n + eps
                trace_score = ((1 - alpha**2) * xi_n / SIG_N**2)[:, None] * np.ones((1, D))
                step_score = (eps / SIG_N**2)[:, None] * np.ones((1, D))
                exact = (eps / SIG_N**2)[:, None] * xin[None, :]
            else:
                nz = rng.normal(0, 1, B)[:, None] * np.ones((1, D)) if placement == "weight" else rng.normal(0, 1, (B, D))
                applied = SIG_S * nz * xin[None, :]; pert = applied.sum(1)
                xi_s = alpha * xi_s + applied
                trace_score = (1 - alpha**2) * xi_s / SIG_S**2
                step_score = applied / SIG_S**2
                if placement == "weight":
                    s = SIG_S * xin.sum(); exact = nz * xin[None, :] / s if s > 0 else np.zeros((B, D))
                else:
                    exact = nz * xin[None, :] / SIG_S
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
            terms = {"dist_eprop": trace_score * e, "trace_nopsi": trace_score * zf[None, :],
                     "step_psi": step_score * e, "step_pretr": step_score * zf[None, :], "exact": exact}
            for k in names: C[k] = LAM * C[k] + terms[k]
            Ccum += exact
            rc = (r - r.mean())[:, None]
            for k in names: E[k] += rc * C[k]
            Et += rc * Ccum
            st["fire"] += z.mean(); st["open"] += (psi > 0).mean()
            st["below"] += (v < A - v_th).mean(); st["above"] += (v > A + v_th).mean()
        for k in names: M1[k] += E[k].sum(0); M2[k] += (E[k]**2).sum()
        Mt += Et.sum(0); Mt2 += (Et**2).sum(); n += B
    steps = (N // B) * T; st = {k: 100 * v_ / steps for k, v_ in st.items()}
    truth = Mt / n
    truth_unc = np.sqrt(Mt2 / n - truth @ truth) / np.linalg.norm(truth) / np.sqrt(n)
    res = {}
    for k in names:
        mk = M1[k] / n
        res[k] = (mk @ truth / np.linalg.norm(mk) / np.linalg.norm(truth), np.sqrt(M2[k] / n - mk @ mk) / np.linalg.norm(mk))
    return st, truth_unc, res

def fmt(st, unc, res):
    return (f"fire {st['fire']:4.1f}% | psi open {st['open']:4.1f}% | far below {st['below']:4.1f}% | far above {st['above']:4.1f}% | "
            f"truth unc. {100*unc:4.1f}% || " + " | ".join(f"{k} {c:.3f}/{rn:.1f}" for k, (c, rn) in res.items()))

if __name__ == "__main__":
    N = int(float(sys.argv[1])) if len(sys.argv) > 1 else 400_000
    rng = np.random.default_rng(5); jit = rng.normal(0, 1, D)
    def ei(exc, inh):
        w = exc + 0.75 * exc * jit; w[6:] = -inh; return w
    regimes = {"baseline": ev.w.copy(), "inhibited (mild)": ei(0.08, 0.10), "inhibited (strong)": ei(0.12, 0.20),
               "saturated (mild)": 0.08 + 0.75 * 0.08 * jit, "saturated (strong)": 0.15 + 0.75 * 0.15 * jit}
    print("cells: cosine to truth / relative noise", flush=True)
    for name, w in regimes.items():
        t0 = time.time(); out = run(w, "weight", N)
        print(f"[B] weight | {name:18s} | " + fmt(*out) + f"  [{time.time()-t0:.0f}s]", flush=True)
    for name in ("baseline", "inhibited (strong)"):
        for pl in ("node", "ind"):
            t0 = time.time(); out = run(regimes[name], pl, N)
            print(f"[C] {pl:6s} | {name:18s} | " + fmt(*out) + f"  [{time.time()-t0:.0f}s]", flush=True)
