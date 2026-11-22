"""Correlation of psi with the live shadow's psi, and the shadow gate's variance relative to psi - psibar, per recording."""
import sys
import numpy as np
for path in sys.argv[1:]:
    d = np.load(path, allow_pickle=True)
    vth = float(d["v_thresh"]); beta = d["state_Beta"]
    def psi(v, a, ref):
        p = (1.0 / vth) * 0.3 * np.maximum(0.0, 1.0 - np.abs((v - (vth + beta * a)) / vth))
        return np.where(ref > 0, 0.0, p)
    ps = psi(d["state_ShadowV"], d["state_ShadowA"], d["state_ShadowRef"])
    p, pb = d["state_Psi"], d["state_PsiBar"]
    active = p.std((0, 1)) > 0
    cs = [np.corrcoef(p[..., j].ravel(), ps[..., j].ravel())[0, 1] for j in np.flatnonzero(active & (ps.std((0, 1)) > 0))]
    vr = np.var(p - ps, axis=(0, 1))[active] / (np.var(p - pb, axis=(0, 1))[active] + 1e-30)
    da = np.abs(d["state_A"] - d["state_ShadowA"]).mean()
    print(f"{path.split('/')[0]}: corr(psi, psi_shadow) median {np.median(cs):.2f}  "
          f"Var(psi - psi_s) / Var(psi - psibar) median {np.median(vr):.2f}  mean |A - A_shadow| {da:.3f}")
