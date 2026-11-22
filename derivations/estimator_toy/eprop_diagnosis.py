"""Why does symmetric e-prop point the wrong way in the single-neuron toy?
Step 1 (code check): smooth deterministic neuron (z = sigmoid((v-A)/tau), continuous reset,
        no noise). Full single-neuron RTRL with the exact derivative must equal finite differences.
        Also report e-prop (reset derivative dropped) in the same smooth model.
Step 2/3 (spiking, noisy): pathwise estimates sum_t L_t dz_t/dW vs the true gradient of E[F]
        (mean of the exact perturbation estimator), for
          e-prop, default psi (width v_th)      e-prop, narrow psi (width ~ noise)
          RTRL with reset, default psi          RTRL with reset, narrow psi
Usage: python3 eprop_diagnosis.py N_SAMPLES"""
import sys, numpy as np
import control_variate as cv
ev = cv.ev
alpha, rho, v_th, beta_a, gamma_pd, n_ref = cv.alpha, cv.rho, cv.v_th, cv.beta_a, cv.gamma_pd, cv.n_ref
D, T, X, K, Wout = cv.D, cv.T, cv.X, cv.K, cv.Wout
kappa_y = cv.kappa_y; ev_times = cv.ev_times; S_ek, TH_ek, Dr = cv.S_ek, cv.TH_ek, cv.Dr
cosf = lambda p, q: p @ q / np.linalg.norm(p) / np.linalg.norm(q)

def reward_and_L(Z):
    """Z (B,T) spikes (possibly continuous) -> F (B,), L (B,T) = dF/dz_t (symmetric feedback)."""
    B = Z.shape[0]; y = np.zeros((B, K)); Yk = np.zeros((B, K, T))
    for t in range(T):
        y = kappa_y * y + Wout[None, :] * Z[:, t:t + 1]; Yk[:, :, t] = y
    F = np.zeros(B); dFdy = np.zeros((B, len(ev_times), K))
    for ie, te in enumerate(ev_times):
        u = (Yk[:, :, te] - TH_ek[ie]) / 0.5
        F += Dr[ie] * (S_ek[ie] * np.tanh(u)).mean(1)
        dFdy[:, ie] = Dr[ie] * S_ek[ie] * (1 - np.tanh(u)**2) / 0.5 / K
    decay = np.zeros((len(ev_times), T))
    for ie, te in enumerate(ev_times): decay[ie, :te + 1] = kappa_y ** (te - np.arange(te + 1))
    return F, np.einsum("bek,k,et->bt", dFdy, Wout, decay)

def run_model(w, eps, smooth_tau=None, psi_width=v_th):
    """Returns F, L, and dz/dW under e-prop (no reset derivative) and RTRL (with reset derivative).
    smooth_tau: if set, z = sigmoid((v-A)/tau), dz/dv exact, no refractoriness (deterministic check)."""
    B = eps.shape[0]; drive = w @ X
    v = np.zeros(B); a = np.zeros(B); z = np.zeros(B); ref = np.zeros(B)
    Sv_e = np.zeros((B, D)); Sa_e = np.zeros((B, D))          # e-prop eligibility vectors
    Sv_r = np.zeros((B, D)); Sa_r = np.zeros((B, D))          # RTRL sensitivities (with reset)
    dz_e = np.zeros((B, T, D)); dz_r = np.zeros((B, T, D)); Z = np.zeros((B, T))
    psi_prev = np.zeros(B); dzr_prev = np.zeros((B, D))
    for t in range(T):
        xin = X[:, t - 1] if t > 0 else np.zeros(D)
        # adaptation and threshold
        a_new = rho * a + z
        Sa_e = rho * Sa_e + psi_prev[:, None] * (Sv_e - beta_a * Sa_e)
        Sa_r = rho * Sa_r + dzr_prev
        a = a_new; A = v_th + beta_a * a
        # membrane (reset uses previous spike)
        v = alpha * v + (drive[t - 1] if t > 0 else 0.0) - v_th * z + eps[:, t]
        Sv_e = alpha * Sv_e + xin[None, :]                       # e-prop: reset derivative dropped
        Sv_r = alpha * Sv_r + xin[None, :] - v_th * dzr_prev      # RTRL: through the reset
        if smooth_tau is None:
            ref = np.maximum(ref - 1, 0); refractory = ref > 0
            z = ((v >= A) & ~refractory).astype(float); ref = np.where(z > 0, n_ref, ref)
            psi = np.where(refractory, 0.0, (gamma_pd / psi_width) * np.maximum(0, 1 - np.abs(v - A) / psi_width))
        else:
            s = 1 / (1 + np.exp(-(v - A) / smooth_tau)); z = s; psi = s * (1 - s) / smooth_tau
        de = psi[:, None] * (Sv_e - beta_a * Sa_e)
        dr = psi[:, None] * (Sv_r - beta_a * Sa_r)
        dz_e[:, t] = de; dz_r[:, t] = dr; Z[:, t] = z
        psi_prev = psi; dzr_prev = dr
    F, L = reward_and_L(Z)
    return F, L, dz_e, dz_r

if __name__ == "__main__":
    N = int(float(sys.argv[1])) if len(sys.argv) > 1 else 40_000
    w0 = ev.w.copy()
    # ---- Step 1: smooth deterministic code check
    tau = 0.05; h = 1e-5; zero = np.zeros((1, T))
    F0, L, dze, dzr = run_model(w0, zero, smooth_tau=tau)
    g_rtrl = np.einsum("bt,btd->bd", L, dzr)[0]; g_ep = np.einsum("bt,btd->bd", L, dze)[0]
    fd = np.array([(run_model(w0 + h * np.eye(D)[d], zero, smooth_tau=tau)[0][0] - run_model(w0 - h * np.eye(D)[d], zero, smooth_tau=tau)[0][0]) / (2 * h) for d in range(D)])
    print("Step 1, smooth deterministic neuron (tau=0.05):")
    print(f"   RTRL vs finite differences:  cos = {cosf(g_rtrl, fd):.4f}, max rel. err = {np.max(np.abs(g_rtrl - fd)) / np.max(np.abs(fd)):.2e}")
    print(f"   e-prop (no reset derivative) vs finite differences: cos = {cosf(g_ep, fd):.3f}")
    # ---- Steps 2/3: spiking, noisy
    rng = np.random.default_rng(5); jit = rng.normal(0, 1, D)
    w_inh = 0.12 + 0.75 * 0.12 * jit; w_inh[6:] = -0.2
    SIG = cv.SIG; sxi = SIG / np.sqrt(1 - alpha**2)
    for name, w in (("near threshold", w0), ("inhibited", w_inh)):
        rng = np.random.default_rng(0); B = 20000
        acc = {k: np.zeros(D) for k in ("truth", "ep_wide", "ep_narrow", "rtrl_wide", "rtrl_narrow")}; n = 0
        for _ in range(N // B):
            eps = rng.normal(0, SIG, (B, T))
            F, L, dze, dzr = run_model(w, eps, psi_width=v_th)
            _, _, dze_n, dzr_n = run_model(w, eps, psi_width=2 * sxi)
            S = eps @ np.array([X[:, t - 1] if t > 0 else np.zeros(D) for t in range(T)]) / SIG**2
            acc["truth"] += ((F - F.mean())[:, None] * S).sum(0)
            acc["ep_wide"] += np.einsum("bt,btd->d", L, dze); acc["rtrl_wide"] += np.einsum("bt,btd->d", L, dzr)
            acc["ep_narrow"] += np.einsum("bt,btd->d", L, dze_n); acc["rtrl_narrow"] += np.einsum("bt,btd->d", L, dzr_n)
            n += B
        tr = acc["truth"] / n
        print(f"\n{name}: cosine to the true gradient of E[F] (narrow psi width = {2*sxi:.3f})")
        for k in ("ep_wide", "rtrl_wide", "ep_narrow", "rtrl_narrow"):
            print(f"   {k:12s} {cosf(acc[k] / n, tr):6.3f}")
