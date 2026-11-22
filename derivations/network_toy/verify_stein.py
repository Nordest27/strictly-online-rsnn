"""Check the definition of the drift: E[s] = E[D e]  (Stein) and how close it is to dE[e]/dW.

From a snapshot of a trained grid-toy network (weights from net_toy_grid.py --save-weights), with a fixed
input spike train, run many independent noise rollouts of T steps. For a set of target synapses ij:

  A  E[s_ij,t]                    mean score over rollouts (score = e * NoiseTrace / SynSig^2,
                                  NoiseTrace started at 0 at the snapshot)
  B  (1/sigma) d/dh E[e_ij,T(n + h v)] with v_{j,T-k} = alpha^k z_{i,T-k-1}: Stein's identity says B = A
                                  exactly (at t = T); checks the derivation and its bookkeeping
  C  dE[e_ij,t]/dw_ij              plain weight derivative of the eligibility (the "P ~ de/dW" claim)

All groups share the same noise samples (common random numbers). Usage:
  python3 verify_stein.py data/verify_weights.npz [R] [S]
"""
import sys, time
import numpy as np

f32 = np.float32
wf = sys.argv[1] if len(sys.argv) > 1 else "data/verify_weights.npz"
R = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
S = int(sys.argv[3]) if len(sys.argv) > 3 else 30
d = np.load(wf)
W, M, NIN, N, NF, rates = d["W"], d["M"], int(d["NIN"]), int(d["N"]), int(d["NF"]), d["rates"]
PRE, POST = W.shape
v_th, n_ref, beta_a = 0.61, 3, 0.0174
alpha, rho = np.exp(-1 / 10), np.exp(-1 / 300)
SIG = np.exp(-5.0)
T = 30
H_W, H_N = 0.01, 0.3
rng = np.random.default_rng(1)


def psi_of(v, A, refr):
    return np.where(refr, 0.0, (0.3 / v_th) * np.maximum(0, 1 - np.abs(v - A) / v_th)).astype(f32)


# ---- warm-up a single network to get a snapshot state ----
v = np.zeros(POST, f32); ad = np.zeros(POST, f32); zs = np.zeros(POST, f32); ref = np.zeros(POST, np.int32)
zbar = np.zeros(PRE, f32); epsA = np.zeros((PRE, POST), f32); zin = np.zeros(NIN, f32)
for t in range(400):
    zpre = np.concatenate([zin, zs[:N]])
    nz = rng.standard_normal(POST).astype(f32)
    I = zpre @ W + SIG * nz * (zpre @ M)
    ad = rho * ad + zs; A = v_th + beta_a * ad
    v = alpha * v + I - v_th * zs
    ref = np.maximum(ref - 1, 0); refr = ref > 0
    zs = ((v >= A) & ~refr).astype(f32); ref = np.where(zs > 0, n_ref, ref)
    psi = psi_of(v, A, refr)
    zbar = alpha * zbar + zpre
    epsA = psi[None, :] * zbar[:, None] + (rho - psi[None, :] * beta_a) * epsA
    zin = (rng.random(NIN) < rates).astype(f32)
print(f"snapshot: hidden {zs[:N].mean():.2f} spikes/step now; mean psi {psi.mean():.3f}")

# ---- target synapses: existing, presynaptically active, postsynaptic neuron in or near psi range ----
ii, jj = np.nonzero(M)
ok = (zbar[ii] > 0.3)
ii, jj = ii[ok], jj[ok]
pick = rng.choice(len(ii), min(S, len(ii)), replace=False)
ti, tj = ii[pick], jj[pick]
S = len(ti)
print(f"{S} target synapses ({(ti < NIN).sum()} from inputs)")

# fixed input spike train for the window
ZIN = (rng.random((T, NIN)) < rates).astype(f32)

# ---- batch: groups 0 base | 1..S +h on w_s | S+1..2S -h on w_s | 2S+1..3S noise +h v_s | 3S+1..4S noise -h v_s
G = 4 * S + 1
B = G * R
grp = np.repeat(np.arange(G), R)
rid = np.tile(np.arange(R), G)
dw = np.zeros(B, f32)
dw[(grp >= 1) & (grp <= S)] = H_W
dw[(grp > S) & (grp <= 2 * S)] = -H_W
w_tgt = np.where(grp <= S, grp - 1, grp - S - 1).clip(0, S - 1)      # which synapse a weight group perturbs
dn = np.zeros(B, f32)
dn[(grp > 2 * S) & (grp <= 3 * S)] = H_N
dn[grp > 3 * S] = -H_N
n_tgt = np.where(grp <= 3 * S, grp - 2 * S - 1, grp - 3 * S - 1).clip(0, S - 1)
NOISE = np.random.default_rng(7).standard_normal((T, R, POST)).astype(f32)

vB = np.repeat(v[None], B, 0); adB = np.repeat(ad[None], B, 0); zsB = np.repeat(zs[None], B, 0)
refB = np.repeat(ref[None], B, 0); zbarB = np.repeat(zbar[None], B, 0)
epsS = np.repeat(epsA[ti, tj][None], B, 0)                          # (B, S) adaptation eligibility of targets
xiS = np.zeros((B, S), f32)
zprev_i = np.zeros((B, S), f32)                                      # z_{i, t-1} of each target, per rollout
base = grp == 0
SC = np.zeros((T, R, S), f32)                                        # base-group scores only
Ep = np.zeros((T, S)); Em = np.zeros((T, S)); Ebase = np.zeros((T, S)); Np = np.zeros(S); Nm = np.zeros(S)
gidx = [np.nonzero(grp == g)[0] for g in range(G)]
t0 = time.time()
for t in range(T):
    zpre = np.concatenate([np.broadcast_to(ZIN[t], (B, NIN)), zsB[:, :N]], 1)
    nz = NOISE[t][rid].copy()                                         # common random numbers across groups
    # noise-direction perturbation (Stein check, target time T-1): v_{j, tau} = alpha^(T-1-tau) z_{i, tau-1}
    sel = dn != 0
    if sel.any():
        b_ = np.nonzero(sel)[0]; s_ = n_tgt[b_]
        nz[b_, tj[s_]] += dn[b_] * alpha ** (T - 1 - t) * zpre[b_, ti[s_]]
    I = zpre @ W + SIG * nz * (zpre @ M)
    selw = dw != 0
    b_ = np.nonzero(selw)[0]; s_ = w_tgt[b_]
    I[b_, tj[s_]] += dw[b_] * zpre[b_, ti[s_]]                        # weight perturbation w_s + dw
    adB = rho * adB + zsB; A = v_th + beta_a * adB
    vB = alpha * vB + I - v_th * zsB
    refB = np.maximum(refB - 1, 0); refr = refB > 0
    zsB = ((vB >= A) & ~refr).astype(f32); refB = np.where(zsB > 0, n_ref, refB)
    psi = psi_of(vB, A, refr)
    zbarB = alpha * zbarB + zpre
    zi = zpre[:, ti]; ps = psi[:, tj]; zb = zbarB[:, ti]
    e = ps * (zb - beta_a * epsS)
    epsS = ps * zb + (rho - ps * beta_a) * epsS
    xiS = xiS + SIG * nz[:, tj] * zi                                  # NoiseTrace of the targets (base noise)
    SC[t] = (e * xiS / SIG ** 2)[base]
    Ebase[t] = e[base].mean(0)
    for s in range(S):
        Ep[t, s] = e[gidx[1 + s], s].mean(); Em[t, s] = e[gidx[S + 1 + s], s].mean()
        if t == T - 1:
            Np[s] = e[gidx[2 * S + 1 + s], s].mean(); Nm[s] = e[gidx[3 * S + 1 + s], s].mean()
    xiS *= alpha
print(f"rollouts done in {time.time() - t0:.0f}s (B = {B})")

A_ = SC.mean(1)                                                       # (T, S)  E[s]
A_se = SC.std(1) / np.sqrt(R)
C_ = (Ep - Em) / (2 * H_W)                                            # (T, S)  dE[e]/dw
B_ = (Np - Nm) / (2 * H_N) / SIG                                      # (S,)  Stein RHS at t = T-1


def corr(x, y):
    x, y = x.ravel(), y.ravel(); return np.corrcoef(x, y)[0, 1]
def slope(x, y):
    x, y = x.ravel(), y.ravel(); return (x * y).sum() / (x * x).sum()

print("\nA vs B  (Stein identity, t = T-1, per synapse)")
print(f"   corr {corr(B_, A_[T-1]):+.3f}   slope A/B {slope(B_, A_[T-1]):.3f}   "
      f"median |A-B| / SE(A) {np.median(np.abs(A_[T-1] - B_) / (A_se[T-1] + 1e-12)):.2f}")
late = slice(5, T)
print("A vs C  (drift vs plain weight derivative of the eligibility, all t >= 5)")
print(f"   corr {corr(C_[late], A_[late]):+.3f}   slope A/C {slope(C_[late], A_[late]):.3f}")
print(f"   (control) corr of A with E[e] itself: {corr(Ebase[late], A_[late]):+.3f}")
print(f"   corr of dE[e]/dw with E[e]:            {corr(Ebase[late], C_[late]):+.3f}")
# partial: does dE[e]/dw explain A beyond E[e]?  least squares A ~ a*E[e] + c*C
X = np.stack([Ebase[late].ravel(), C_[late].ravel()], 1); y = A_[late].ravel()
coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
r2 = 1 - ((y - X @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
r2e = np.corrcoef(Ebase[late].ravel(), y)[0, 1] ** 2; r2c = np.corrcoef(C_[late].ravel(), y)[0, 1] ** 2
print(f"   R^2 of A: E[e] alone {r2e:.3f} | dE[e]/dw alone {r2c:.3f} | both {r2:.3f}")
print(f"   mean SE of A relative to |A|: {np.mean(A_se[late]) / np.mean(np.abs(A_[late])):.2f}")
np.savez("data/verify_stein_out.npz", A=A_, A_se=A_se, B=B_, C=C_, ti=ti, tj=tj)
