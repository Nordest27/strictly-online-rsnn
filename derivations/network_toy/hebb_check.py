"""Is the predictable part (drift) of the weight-dist score a Hebbian (pre x post) term?

Uses the twin dump from net_toy_twin3.py (--dump): per sampled step, the twin-estimated predictable
part P_ij of the score, the presynaptic traces and the postsynaptic state.

1. Factorisation: per step and postsynaptic neuron j, fit P_ij ~ h_j * u_ij over its synapses, with
   u a purely presynaptic trace. Report the fraction of P explained.
2. What sets h_j: mean h per postsynaptic state bin (time since its last spike / distance to threshold)
   and the fraction of h's variance a state table explains.
3. Pre x post rule H_ij = h_hat(state_j) * u_ij vs P, compared with simple candidates.
"""
import sys
import numpy as np

d = np.load(sys.argv[1] if len(sys.argv) > 1 else "data/hebb_dump.npz")
M = d["M"].astype(bool)
P = d["P"].astype(np.float64)                     # (T, PRE, POST)
pre = d["pre"].astype(np.float64)
zb2, zbar = d["zb2"], d["zbar"]
z, tsl, vA, psi = d["z"], d["tsl"], d["vA"], d["psi"]
T = P.shape[0]
print(f"{T} samples, {M.sum()} synapses")

Mb = M[None]
cands = {
    "pre_term (e-prop presyn. factor)": pre,
    "pre_term x alpha^2 pre trace": pre * zb2[:, :, None],
    "zbar x alpha^2 pre trace": (zbar * zb2)[:, :, None] * np.ones_like(P),
}


def sep_fit(u):
    num = (P * u * Mb).sum(1); den = (u * u * Mb).sum(1) + 1e-30
    h = num / den                                  # (T, POST)
    res = ((P - h[:, None, :] * u) ** 2 * Mb).sum()
    return h, 1 - res / (P ** 2 * Mb).sum()


print("\n1. Factorisation P_ij ~ h_j(t) * u_ij  (fraction of sum P^2 explained)")
fits = {}
for name, u in cands.items():
    h, r2 = sep_fit(u); fits[name] = (h, u)
    print(f"   {name:36s} {r2:.3f}")

# reference: how much of P is explained by a free post factor with u = P's own row mean (upper-ish bound)
name = "pre_term x alpha^2 pre trace"
h, u = fits[name]
w = (u * u * Mb).sum(1)                            # weight of each (t, j) in P

# ---- 2. postsynaptic state bins ----
def state_bin(tsl, vA):
    b = np.full(tsl.shape, -1)
    b[tsl == 0] = 0                                # spiked this step
    b[tsl == 1] = 1; b[tsl == 2] = 2; b[tsl == 3] = 3   # refractory (psi = 0)
    far = tsl >= 4
    edges = [-np.inf, -1.2, -0.6, -0.3, 0.0, np.inf]
    for k in range(5):
        b[far & (vA >= edges[k]) & (vA < edges[k + 1])] = 4 + k
    return b
labels = ["spike now", "1 step after", "2 steps after", "3 steps after",
          "V-A < -1.2", "-1.2..-0.6", "-0.6..-0.3", "-0.3..0", ">= 0 (no spike)"]
bins = state_bin(tsl, vA)
print("\n2. Postsynaptic factor h by postsynaptic state (weighted by presynaptic activity)")
print(f"   {'state':18s} {'share of P weight':>17s} {'mean h':>10s} {'mean psi':>9s}")
hhat = np.zeros_like(h)
for k, lab in enumerate(labels):
    sel = bins == k
    if not sel.any():
        continue
    mh = (h[sel] * w[sel]).sum() / (w[sel].sum() + 1e-30)
    hhat[sel] = mh
    print(f"   {lab:18s} {w[sel].sum() / w.sum():17.3f} {mh:+10.3f} {psi[sel].mean():9.3f}")
var_tot = ((h - (h * w).sum() / w.sum()) ** 2 * w).sum()
var_res = ((h - hhat) ** 2 * w).sum()
print(f"   state table explains {1 - var_res / var_tot:.3f} of the (weighted) variance of h")

# ---- 3. pre x post rules vs P ----
def cos(a, b):
    return (a * b * Mb).sum() / np.sqrt((a * a * Mb).sum() * (b * b * Mb).sum())
H = hhat[:, None, :] * u
eprop = psi[:, None, :] * pre
hebb_spike = z[:, None, :] * zbar[:, :, None]
print("\n3. Cosine with P over all sampled steps and synapses")
print(f"   H = h_hat(post state) x pre trace      {cos(H, P):+.3f}")
print(f"   e-prop eligibility psi x pre_term      {cos(eprop, P):+.3f}")
print(f"   classic Hebbian post spike x zbar      {cos(hebb_spike, P):+.3f}")
# time-accumulated (what the lambda trace sees): sum over samples
Ps, Hs = (P * Mb).sum(0), (H * Mb).sum(0)
print(f"   summed over time: cos(sum H, sum P)    {(Ps * Hs).sum() / np.sqrt((Ps**2).sum() * (Hs**2).sum()):+.3f}")


def cos_sum(a):
    s_ = (a * Mb).sum(0)
    return (Ps * s_).sum() / np.sqrt((Ps ** 2).sum() * (s_ ** 2).sum())
print("\n   controls, summed over time (cos with sum P):")
print(f"   presynaptic trace alone (h = 1)        {cos_sum(u):+.3f}")
print(f"   e-prop eligibility psi x pre_term      {cos_sum(eprop):+.3f}")
print(f"   classic Hebbian post spike x zbar      {cos_sum(hebb_spike):+.3f}")
print(f"   free per-step h (upper bound)          {cos_sum(h[:, None, :] * u):+.3f}")
# split-half check: table fitted on first half, evaluated on second half
half = T // 2
hh = np.zeros_like(h)
for k in range(len(labels)):
    sel1 = bins[:half] == k
    if sel1.any():
        hh[bins == k] = (h[:half][sel1] * w[:half][sel1]).sum() / (w[:half][sel1].sum() + 1e-30)
P2, H2 = (P[half:] * Mb).sum(0), ((hh[:, None, :] * u)[half:] * Mb).sum(0)
print(f"   table from 1st half, tested on 2nd     {(P2 * H2).sum() / np.sqrt((P2**2).sum() * (H2**2).sum()):+.3f}")
