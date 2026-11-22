"""E-prop's update as a control variate for the exact perturbation estimator.

Single ALIF neuron, node (membrane) noise sigma_n = e^-4.61, K = 4 leaky readouts
y_k = kappa_y y_k + Wout_k z, sparse reward events R_e = mean_k s_ek tanh((y_k - th_ek)/0.5)
delivered as a decaying trace; objective F = sum_t r_t (whole trajectory, no critic).

  exact      : (F - b) * S,                     S_i = sum_t eps^t x_i^{t-1} / sigma^2
  CV         : (F - b - beta * c) * S + beta * sum_t h_t zbar_i^t,   c = sum_t h_t xi^t
  eprop only : sum_t h_t zbar_i^t    (the e-prop-like correction term on its own)
  dist eprop : (F - b) * sum_t (1-alpha^2) xi^t / sigma^2 * e_i^t   (trace rule)
with h_t = L_t * psi_t, L_t = sum_k B_k dF/dy_k propagated back to spike t (e-prop learning
signal), B = Wout (symmetric) or random (random feedback).
For deterministic h (from the noise-free trajectory) CV is exactly unbiased for any beta:
E[c S] = sum_t h_t zbar^t by the Gaussian identity. With h from the noisy trajectory
(what e-prop computes) it is slightly biased. beta is fitted by regressing F on c.
Usage: python3 control_variate.py N_SAMPLES"""
import sys, time, numpy as np
import alif_bias_events as ev

alpha, rho, v_th, beta_a, gamma_pd, n_ref = ev.alpha, ev.rho, ev.v_th, ev.beta, ev.gamma_pd, ev.n_ref
D, T, X = ev.D, ev.T, ev.X
kappa_y, kappa_r = ev.kappa_y, ev.kappa_r
SIG = np.exp(-4.61); K = 4
rng0 = np.random.default_rng(11)
Wout = rng0.normal(0, 1, K)
Brand = rng0.normal(0, 1, K); Brand *= np.linalg.norm(Wout) / np.linalg.norm(Brand)
ev_times = sorted(ev.targets.keys())
S_ek = rng0.choice([-1.0, 1.0], (len(ev_times), K)); TH_ek = rng0.uniform(-0.5, 1.5, (len(ev_times), K))
Dr = np.array([sum(kappa_r ** (t - te) for t in range(te, T)) for te in ev_times])   # reward-trace total weight

def simulate(w, eps):
    """eps (B,T) -> spikes Z, psi, xi, e (B,T,D), F, and learning signals for both feedbacks."""
    B = eps.shape[0]; drive = w @ X
    v = np.zeros(B); a = np.zeros(B); z = np.zeros(B); ref = np.zeros(B)
    Z = np.zeros((B, T)); PSI = np.zeros((B, T)); XI = np.zeros((B, T)); E = np.zeros((B, T, D))
    xi = np.zeros(B); zf = np.zeros(D); ea = np.zeros((B, D)); ZF = np.zeros((T, D)); XIN = np.zeros((T, D))
    for t in range(T):
        xin = X[:, t - 1] if t > 0 else np.zeros(D)
        a = rho * a + z; A = v_th + beta_a * a
        v = alpha * v + (drive[t - 1] if t > 0 else 0.0) - v_th * z + eps[:, t]
        ref = np.maximum(ref - 1, 0); refractory = ref > 0
        z = ((v >= A) & ~refractory).astype(float); ref = np.where(z > 0, n_ref, ref)
        psi = np.where(refractory, 0.0, (gamma_pd / v_th) * np.maximum(0, 1 - np.abs(v - A) / v_th))
        zf = alpha * zf + xin; xi = alpha * xi + eps[:, t]
        e = psi[:, None] * (zf[None, :] - beta_a * ea); ea = psi[:, None] * zf[None, :] + (rho - psi[:, None] * beta_a) * ea
        Z[:, t] = z; PSI[:, t] = psi; XI[:, t] = xi; E[:, t] = e; ZF[t] = zf; XIN[t] = xin
    # readouts and reward
    Yk = np.zeros((B, K, T)); y = np.zeros((B, K))
    for t in range(T):
        y = kappa_y * y + Wout[None, :] * Z[:, t:t + 1]; Yk[:, :, t] = y
    F = np.zeros(B); dFdy = np.zeros((B, len(ev_times), K))
    for ie, te in enumerate(ev_times):
        u = (Yk[:, :, te] - TH_ek[ie]) / 0.5
        F += Dr[ie] * (S_ek[ie] * np.tanh(u)).mean(1)
        dFdy[:, ie] = Dr[ie] * S_ek[ie] * (1 - np.tanh(u)**2) / 0.5 / K
    # learning signal L_t = sum_e sum_k B_k dF/dy_k(e) kappa_y^(te - t) for t <= te
    decay = np.zeros((len(ev_times), T))
    for ie, te in enumerate(ev_times):
        decay[ie, :te + 1] = kappa_y ** (te - np.arange(te + 1))
    L = {name: np.einsum("bek,k,et->bt", dFdy, Bv, decay) for name, Bv in (("sym", Wout), ("rand", Brand))}
    return F, PSI, XI, E, ZF, XIN, L

def run(w, N, B=20000, seed=0):
    rng = np.random.default_rng(seed); s2 = SIG**2
    # noise-free trajectory -> deterministic h
    _, psi0, _, _, ZF, XIN, L0 = simulate(w, np.zeros((1, T)))
    h_nf = {k: (L0[k] * psi0)[0] for k in L0}
    # oracle: true sensitivity of F to the membrane from a separate pilot (Stein)
    geps = np.zeros(T); npil = 0
    for _ in range(max(1, N // B // 2)):
        ep = rng.normal(0, SIG, (B, T)); Fp = simulate(w, ep)[0]; geps += ((Fp - Fp.mean())[:, None] * ep).sum(0) / s2; npil += B
    geps /= npil; g_or = geps - alpha * np.append(geps[1:], 0.0)
    # pilot batch -> beta by regression of F on c
    eps = rng.normal(0, SIG, (B, T)); F, PSI, XI, E, _, _, L = simulate(w, eps); Fc = F - F.mean()
    betas = {}
    c = (np.broadcast_to(g_or, (B, T)) * XI).sum(1); betas[("oracle", "nf")] = (Fc @ c) / (c @ c)
    for fb in ("sym", "rand"):
        for src, h in (("nf", np.broadcast_to(h_nf[fb], (B, T))), ("noisy", L[fb] * PSI)):
            c = (h * XI).sum(1); betas[(fb, src)] = (Fc @ c) / (c @ c)
    names = ["exact", "dist_eprop", "CV_oracle"] + [f"CV_{fb}_{src}" for fb in ("sym", "rand") for src in ("nf", "noisy")] + ["eprop_sym", "eprop_rand", "eprop_sym_nopsi", "oracle_path"]
    M1 = {k: np.zeros(D) for k in names}; M2 = {k: 0.0 for k in names}; n = 0
    for i in range(N // B):
        eps = rng.normal(0, SIG, (B, T)); F, PSI, XI, E, ZF, XIN, L = simulate(w, eps); Fc = F - F.mean()
        S = eps @ XIN / s2                                        # exact score (B,D)
        est = {"exact": Fc[:, None] * S,
               "dist_eprop": Fc[:, None] * np.einsum("bt,btd->bd", (1 - alpha**2) * XI / s2, E)}
        for fb in ("sym", "rand"):
            for src in ("nf", "noisy"):
                h = np.broadcast_to(h_nf[fb], (B, T)) if src == "nf" else L[fb] * PSI
                bt = betas[(fb, src)]; c = (h * XI).sum(1)
                corr = h @ ZF                                        # sum_t h_t zbar^t  (B,D)
                est[f"CV_{fb}_{src}"] = (Fc - bt * c)[:, None] * S + bt * corr
            est[f"eprop_{fb}"] = (L[fb] * PSI) @ ZF
        ho = np.broadcast_to(g_or, (B, T)); bo = betas[("oracle", "nf")]; co = (ho * XI).sum(1)
        est["CV_oracle"] = (Fc - bo * co)[:, None] * S + bo * (g_or @ ZF)[None, :]
        est["eprop_sym_nopsi"] = L["sym"] @ ZF
        est["oracle_path"] = np.broadcast_to(g_or @ ZF, (B, D))
        for k in names: M1[k] += est[k].sum(0); M2[k] += (est[k]**2).sum()
        n += B
    truth = M1["exact"] / n
    out = {}
    for k in names:
        mk = M1[k] / n
        out[k] = (mk @ truth / np.linalg.norm(mk) / np.linalg.norm(truth), np.sqrt(M2[k] / n - mk @ mk) / np.linalg.norm(mk))
    unc = out["exact"][1] / np.sqrt(n)
    print(f"   fitted beta: " + ", ".join(f"{k[0]}/{k[1]}={v:.3g}" for k, v in betas.items()))
    return out, unc

if __name__ == "__main__":
    N = int(float(sys.argv[1])) if len(sys.argv) > 1 else 200_000
    print(f"cos(random feedback, forward readout weights) = {Brand @ Wout / np.linalg.norm(Brand) / np.linalg.norm(Wout):.2f}")
    rng = np.random.default_rng(5); jit = rng.normal(0, 1, D)
    w_inh = 0.12 + 0.75 * 0.12 * jit; w_inh[6:] = -0.2
    for name, w in (("near threshold", ev.w.copy()), ("inhibited", w_inh)):
        t0 = time.time(); out, unc = run(w, N)
        print(f"\n== {name}  (truth uncertainty {100*unc:.1f}%, {time.time()-t0:.0f}s)   cells: cosine to truth / relative noise")
        for k, (c, rn) in out.items():
            print(f"   {k:16s} {c:7.3f} / {rn:7.2f}")
