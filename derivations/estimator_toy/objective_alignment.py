"""Is the code's trace estimator aligned or anti-aligned with the true gradient,
depending on the objective? Single ALIF neuron, weight-dist noise as in the code
(per-synapse NoiseTrace, e-prop eligibility with psi, score = e * NoiseTrace / SynSig^2,
no (1-alpha^2)), whole-trajectory credit. Truth = exact estimator (unbiased).
Objectives:
  VReg      : F = sum_t VReg(V_t), V read after reset (as the synapse code sees V_post)
  rate      : F = sum_t z_t (spike count)
  task      : sparse decaying readout-dependent reward (alif_bias_events)"""
import sys, numpy as np
import final_tables as ft
ev = ft.ev
alpha, rho, v_th, beta_a, gamma_pd, n_ref = ft.alpha, ft.rho, ft.v_th, ft.beta, ft.gamma_pd, ft.n_ref
D, T, X, targets = ft.D, ft.T, ft.X, ft.targets
kappa_y, kappa_r, SIG = ft.kappa_y, ft.kappa_r, ft.SIG_S

def run(w, N, B=20000, seed=0):
    rng = np.random.default_rng(seed); drive = w @ X
    acc = {o: {"trace": np.zeros(D), "exact": np.zeros(D), "tsq": 0.0} for o in ("VReg", "rate", "task")}; n = 0; fire = 0.0
    for _ in range(N // B):
        v = np.zeros(B); a = np.zeros(B); z = np.zeros(B); ref = np.zeros(B); y = np.zeros(B); r = np.zeros(B)
        Fv = np.zeros(B); Fr = np.zeros(B); Ft = np.zeros(B)
        zf = np.zeros(D); ea = np.zeros((B, D)); xi = np.zeros((B, D)); St = np.zeros((B, D)); Se = np.zeros((B, D))
        for t in range(T):
            xin = X[:, t - 1] if t > 0 else np.zeros(D)
            a = rho * a + z; A = v_th + beta_a * a
            nz = rng.normal(0, 1, B)[:, None]
            applied = SIG * nz * xin[None, :]
            v = alpha * v + (drive[t - 1] if t > 0 else 0.0) - v_th * z + applied.sum(1)
            ref = np.maximum(ref - 1, 0); refr = ref > 0
            z = ((v >= A) & ~refr).astype(float); ref = np.where(z > 0, n_ref, ref)
            v_post = v - v_th * z                                   # membrane after this step's reset
            Fv += np.maximum(0, v_post - A) + np.maximum(0, -v_post - A)
            Fr += z
            y = kappa_y * y + z; R = 0.0
            if t in targets:
                thr, sgn = targets[t]; R = sgn * np.tanh((y - thr) / 0.5)
            r = kappa_r * r + R; Ft += r
            psi = np.where(refr, 0.0, (gamma_pd / v_th) * np.maximum(0, 1 - np.abs(v - A) / v_th))
            zf = alpha * zf + xin
            e = psi[:, None] * (zf[None, :] - beta_a * ea); ea = psi[:, None] * zf[None, :] + (rho - psi[:, None] * beta_a) * ea
            xi = alpha * xi + applied
            St += e * xi / SIG**2                                    # the code's score
            s = SIG * xin.sum()
            if s > 0: Se += nz * xin[None, :] / s                     # exact score (shared noise)
            fire += z.mean()
        for o, F in (("VReg", Fv), ("rate", Fr), ("task", Ft)):
            Fc = (F - F.mean())[:, None]
            acc[o]["trace"] += (Fc * St).sum(0); acc[o]["exact"] += (Fc * Se).sum(0); acc[o]["tsq"] += ((Fc * Se)**2).sum()
        n += B
    cos = lambda p, q: p @ q / np.linalg.norm(p) / np.linalg.norm(q)
    out = {}
    for o in acc:
        tr = acc[o]["exact"] / n; unc = np.sqrt(acc[o]["tsq"] / n - tr @ tr) / np.linalg.norm(tr) / np.sqrt(n)
        out[o] = (cos(acc[o]["trace"] / n, tr), unc)
    return 100 * fire / (N // B * T), out

if __name__ == "__main__":
    N = int(float(sys.argv[1])) if len(sys.argv) > 1 else 200_000
    rng = np.random.default_rng(5); jit = rng.normal(0, 1, D)
    for name, w in (("near threshold (baseline)", ev.w.copy()),
                    ("active (mean w 0.08)", 0.08 + 0.75 * 0.08 * jit),
                    ("saturated (mean w 0.15)", 0.15 + 0.75 * 0.15 * jit)):
        fire, out = run(w, N)
        print(f"{name:28s} fire {fire:4.1f}% | " + " | ".join(f"{o}: cos {c:+.3f} (truth unc. {100*u:.0f}%)" for o, (c, u) in out.items()), flush=True)
