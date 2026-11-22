"""Experiment 3 (Appendix F): Hebbian structure of the drift.

Train one network with the proposed rule, then record the twin-estimated drift P of the score (gate psi)
for every synapse on 3000 steps sampled late in training, together with presynaptic traces and the
postsynaptic state. Tests:
  1. factorisation P_ij ~ h_j(t) * u_ij with u a presynaptic trace (free postsynaptic factor per step);
  2. the postsynaptic factor h by postsynaptic state (steps since last spike / distance to threshold);
  3. a nine-entry state table times the presynaptic trace against the time-accumulated drift,
     with controls: presynaptic trace alone, e-prop's eligibility, spike-based Hebbian product.
Runtime ~5 min.  Usage: python3 exp3_hebbian.py
"""
import time
import numpy as np
from core import Config, Learner, StaggeredTwins, f32

TRAIN, N_SAMPLES, EVERY = 10000, 3000, 7
LABELS = ["spiking now", "1 step after a spike", "2 steps after", "3 steps after",
          "V-A < -1.2", "-1.2 <= V-A < -0.6", "-0.6 <= V-A < -0.3", "-0.3 <= V-A < 0", "V-A >= 0, no spike"]


def record(cfg):
    L = Learner(cfg, ["proposed"], [0], rng_seed=0)
    L.twins = StaggeredTwins(cfg, L.lay, L.B)
    L.run(TRAIN, block=TRAIN, log=None)
    lay = L.lay
    D = {k: [] for k in ("P", "pre", "zb2", "zbar", "z", "tsl", "vA", "psi")}
    state = dict(zb2=np.zeros((1, lay.PRE), f32), tsl=np.full((1, lay.POST), 99))

    def hook(l, out, ex):
        state["zb2"] = cfg.alpha ** 2 * state["zb2"] + out["zpre"]
        state["tsl"] = np.where(l.st.z > 0, 0, np.minimum(state["tsl"] + 1, 99))
        if l.t % EVERY == 0 and len(D["P"]) < N_SAMPLES:
            for k, v in (("P", ex["P"][0]), ("pre", out["pi"][0]), ("zb2", state["zb2"][0]), ("zbar", l.st.zbar[0]),
                         ("z", l.st.z[0]), ("tsl", state["tsl"][0]), ("vA", (l.st.v - out["A"])[0]), ("psi", out["psi"][0])):
                D[k].append(np.asarray(v, np.float16 if k in ("P", "pre") else np.float32))
    L.hooks.append(hook)
    L.run(N_SAMPLES * EVERY // cfg.K + 1, block=10 ** 9, log=None)
    return L.M[0].astype(bool), {k: np.array(v) for k, v in D.items()}


def state_bin(tsl, vA):
    b = np.full(tsl.shape, 8)
    for k in range(4):
        b[tsl == k] = k
    far = tsl >= 4
    for k, (lo, hi) in enumerate([(-np.inf, -1.2), (-1.2, -0.6), (-0.6, -0.3), (-0.3, 0.0)]):
        b[far & (vA >= lo) & (vA < hi)] = 4 + k
    return b


def main():
    cfg = Config(); t0 = time.time()
    M, D = record(cfg)
    Mb = M[None]; P = D["P"].astype(np.float64); pre = D["pre"].astype(np.float64)
    u = pre * D["zb2"][:, :, None]                            # presynaptic trace: e-prop factor x alpha^2 trace
    num = (P * u * Mb).sum(1); den = (u * u * Mb).sum(1) + 1e-30
    h = num / den; w = den
    T = P.shape[0]
    print(f"{T} samples, {M.sum()} synapses  [{time.time() - t0:.0f}s]\n")
    res_step = 1 - ((P - h[:, None, :] * u) ** 2 * Mb).sum() / (P ** 2 * Mb).sum()
    Ps = (P * Mb).sum(0)
    cos_sum = lambda X: (Ps * (X * Mb).sum(0)).sum() / np.sqrt((Ps ** 2).sum() * (((X * Mb).sum(0)) ** 2).sum())
    print(f"1. factorisation P ~ h_j(t) u_ij: {res_step:.2f} of single-step sum P^2; "
          f"time-accumulated cosine {cos_sum(h[:, None, :] * u):.3f}")
    bins = state_bin(D["tsl"], D["vA"])
    print("\n2. postsynaptic factor h by postsynaptic state")
    print(f"   {'state':24s} {'share of drift':>14s} {'h':>8s}")
    hhat = np.zeros_like(h)
    for k, lab in enumerate(LABELS):
        sel = bins == k
        if sel.any() and w[sel].sum() > 0:
            hhat[sel] = (h[sel] * w[sel]).sum() / w[sel].sum()
            print(f"   {lab:24s} {w[sel].sum() / w.sum():14.2f} {hhat[sel].mean():+8.2f}")
    half = T // 2
    hh = np.zeros_like(h)
    for k in range(len(LABELS)):
        s1 = bins[:half] == k
        if s1.any() and w[:half][s1].sum() > 0:
            hh[bins == k] = (h[:half][s1] * w[:half][s1]).sum() / w[:half][s1].sum()
    P2, H2 = (P[half:] * Mb).sum(0), ((hh[:, None, :] * u)[half:] * Mb).sum(0)
    print("\n3. time-accumulated cosine with the drift")
    print(f"   state table x presynaptic trace          {cos_sum(hhat[:, None, :] * u):+.2f}"
          f"   (fitted on one half, tested on the other: {(P2 * H2).sum() / np.sqrt((P2 ** 2).sum() * (H2 ** 2).sum()):+.2f})")
    print(f"   presynaptic trace alone                  {cos_sum(u):+.2f}")
    print(f"   e-prop eligibility psi x pi              {cos_sum(D['psi'][:, None, :] * pre):+.2f}")
    print(f"   spike-based Hebbian, post spike x zbar   {cos_sum(D['z'][:, None, :] * D['zbar'][:, :, None]):+.2f}")


if __name__ == "__main__":
    main()
