"""Per-moment structure of the drift. Inputs fixed, noise varies: the batch mean of the code's
score at each timestep, m_t (per synapse), is E[score_t | inputs]. Which local quantity tracks
it over time? Candidates (batch means at each t):
  eprop    : e_t = psi_t * (zbar - beta eps_a)                (e-prop eligibility)
  dpsi     : psi'_t * pre_t                                    (smooth Stein term)
  spike    : z_t * pre_t                                       (postsynaptic spike now)
  postspk  : (spike within the last 3 steps) * pre_t           (refractory / just reset)
Reports the correlation over time between m_t and each candidate, averaged over synapses,
and the share of m_t's variance over time (vs its time mean)."""
import numpy as np
import final_tables as ft
ev = ft.ev
alpha, rho, v_th, beta_a, gamma_pd, n_ref = ft.alpha, ft.rho, ft.v_th, ft.beta, ft.gamma_pd, ft.n_ref
D, T, X, SIG = ft.D, ft.T, ft.X, ft.SIG_S

def measure(w, B=40000, seed=0):
    rng = np.random.default_rng(seed); drive = w @ X
    v = np.zeros(B); a = np.zeros(B); z = np.zeros(B); ref = np.zeros(B); last = np.full(B, -99)
    zf = np.zeros(D); ea = np.zeros((B, D)); xi = np.zeros((B, D))
    out = {k: np.zeros((T, D)) for k in ("m", "eprop", "dpsi", "spike", "postspk")}
    for t in range(T):
        xin = X[:, t - 1] if t > 0 else np.zeros(D); a = rho * a + z; A = v_th + beta_a * a
        nz = rng.normal(0, 1, B)[:, None]; applied = SIG * nz * xin[None, :]
        v = alpha * v + (drive[t - 1] if t > 0 else 0.0) - v_th * z + applied.sum(1)
        ref = np.maximum(ref - 1, 0); refr = ref > 0; z = ((v >= A) & ~refr).astype(float); ref = np.where(z > 0, n_ref, ref)
        last = np.where(z > 0, t, last)
        d = v - A; inwin = (np.abs(d) < v_th) & ~refr
        psi = np.where(inwin, (gamma_pd / v_th) * (1 - np.abs(d) / v_th), 0.0)
        dpsi = np.where(inwin, -(gamma_pd / v_th**2) * np.sign(d), 0.0)
        zf = alpha * zf + xin; pre = zf[None, :] - beta_a * ea
        e = psi[:, None] * pre; ea = psi[:, None] * zf[None, :] + (rho - psi[:, None] * beta_a) * ea
        xi = alpha * xi + applied
        out["m"][t] = (e * xi / SIG**2).mean(0)
        out["eprop"][t] = e.mean(0); out["dpsi"][t] = (dpsi[:, None] * pre).mean(0)
        out["spike"][t] = (z[:, None] * pre).mean(0)
        out["postspk"][t] = (((t - last) <= 3) & ((t - last) >= 0))[:, None].astype(float).__mul__(pre).mean(0)
    m = out["m"][20:]
    share = np.mean(m.var(0) / (m.var(0) + m.mean(0)**2))
    res = {}
    for k in ("eprop", "dpsi", "spike", "postspk"):
        c = out[k][20:]
        res[k] = np.nanmean([np.corrcoef(m[:, j], c[:, j])[0, 1] for j in range(D) if c[:, j].std() > 0 and m[:, j].std() > 0])
    return share, res

rng = np.random.default_rng(5); jit = rng.normal(0, 1, D)
for name, w in (("near threshold", ev.w.copy()), ("active", 0.08 + 0.75 * 0.08 * jit)):
    share, res = measure(w)
    print(f"{name:15s} share of drift that varies over time: {share:.2f} | corr over time with: " +
          ", ".join(f"{k} {v:+.2f}" for k, v in res.items()), flush=True)
