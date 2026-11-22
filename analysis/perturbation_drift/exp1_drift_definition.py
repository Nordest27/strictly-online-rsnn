"""Experiment 1 (Appendix D.3): what the drift is.

From a network trained with the proposed rule, run many noise rollouts from a fixed state and compare,
for 40 target synapses, the mean score E[s] with
  (a) the noise-direction derivative of the eligibility (Stein's identity, exact for Gaussian noise),
  (b) the plain weight derivative dE[e]/dW,
  (c) the eligibility E[e] itself (control),
  (d) a local deterministic second-order surrogate built from e-prop quantities (no noise):
      pi * psi'(V-A) * pi  +  pi * sum_s dpsi_t(s) * psi_s * pi_s   (gate slope + spike/reset kernel).
Runtime ~3 min (single core).  Usage: python3 exp1_drift_definition.py
"""
import time
import numpy as np
from core import Config
from experiments_lib import trained_snapshot, snapshot_rollouts, corr_slope

T, R, S, KMAX = 30, 3000, 40, 10


def deterministic_surrogate(cfg, rec):
    X, Z, PS, PI, RF = rec["vA"], rec["z"], rec["psi"], rec["pi"], rec["refr"] > 0.5
    c = 0.3 / cfg.v_th; w = cfg.psi_width
    psif = lambda x: c * np.maximum(0, 1 - np.abs(x) / w)
    slope = np.where(X < 0, c / w, -c / w) * (np.abs(X) < w) * (~RF)
    term1 = PI * slope * PI
    term2 = np.zeros_like(term1)
    for t in range(X.shape[0]):
        acc = np.zeros(X.shape[1:], np.float32)
        for s in range(max(0, t - KMAX), t):
            d = t - s; spk = Z[s] > 0; sh = cfg.v_th * cfg.alpha ** d
            if d <= 2:   # the two refractory steps after a spike close the gate
                dpsi = -np.where(spk, psif(X[t] + sh), PS[t])
            else:        # the reset lowers the voltage by v_th * alpha^d
                dpsi = (np.where(spk, PS[t], np.where(RF[t], 0.0, psif(X[t] - sh)))
                        - np.where(spk, np.where(RF[t], 0.0, psif(X[t] + sh)), PS[t]))
            acc += dpsi * PS[s] * PI[s]
        term2[t] = PI[t] * acc
    return term1, term2


def main():
    cfg = Config()
    t0 = time.time()
    snap = trained_snapshot(cfg, train_steps=10000, seed=0, variant="proposed")
    res = snapshot_rollouts(cfg, snap, R=R, S=S, T=T, record=True)
    A, se, B, C, E = res["A"], res["A_se"], res["B"], res["C"], res["Ebar"]
    late = slice(5, T)
    print(f"{len(res['ti'])} synapses, {R} rollouts, {T}-step window  ({time.time() - t0:.0f}s)\n")
    print(f"{'mean score E[s] against':48s} {'corr':>6s} {'slope':>6s}  note")
    c, k = corr_slope(B, A[-1])
    print(f"{'(a) noise-direction derivative (Stein)':48s} {c:6.2f} {k:6.2f}  median |diff| = "
          f"{np.median(np.abs(A[-1] - B) / (se[-1] + 1e-12)):.1f} s.e.")
    c, k = corr_slope(C[late], A[late])
    print(f"{'(b) weight derivative dE[e]/dW':48s} {c:6.2f} {k:6.2f}  R^2 = {c * c:.2f}")
    c, k = corr_slope(E[late], A[late])
    print(f"{'(c) eligibility E[e] (control)':48s} {c:6.2f} {'':>6s}  R^2 = {c * c:.3f}")
    t1, t2 = deterministic_surrogate(cfg, res["rec"])
    for name, x in (("(d) deterministic surrogate, gate slope only", t1), ("(d) deterministic surrogate, spike kernel only", t2),
                    ("(d) deterministic surrogate, both terms", t1 + t2)):
        c, k = corr_slope(x.mean(1)[late], A[late])
        print(f"{name:48s} {c:6.2f} {k:6.2f}  R^2 = {c * c:.2f}")
    print(f"\nmean s.e. of E[s] relative to |E[s]|: {se[late].mean() / np.abs(A[late]).mean():.2f}")


if __name__ == "__main__":
    main()
