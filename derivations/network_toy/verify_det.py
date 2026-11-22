"""Can the drift be computed without noise? Compare a deterministic second-order surrogate trace with the
measured drift E[s] (same setup as verify_stein.py: trained grid-toy weights, fixed input spike train,
many noise rollouts, target synapses ij).

Deterministic estimate of d e_t / dW (local, uses only the neuron's own state and e-prop quantities):
  term1 (subthreshold)  pi_t * psi'(x_t) * pi_t            x = V - A, psi' = slope of e-prop's gate,
                                                            pi = zbar - beta*epsA = e-prop's d(V - A)/dW
  term2 (spike events)  pi_t * sum_{s<t, t-s<=KMAX} dpsi_t(s) * psi_s * pi_s
        psi_s * pi_s : e-prop's surrogate for how the weight changes the chance of a spike at s
        dpsi_t(s)    : how a spike at s changes the gate at t (with spike minus without):
                       t-s in {1,2} -> gate closed (refractory); later -> reset lowers V by v_th*alpha^(t-s)
                       (computed from the actual state, undoing or adding the reset; adaptation ignored)
Usage: python3 verify_det.py data/verify_weights.npz [R] [S]
"""
import sys, time
import numpy as np

f32 = np.float32
wf = sys.argv[1] if len(sys.argv) > 1 else "data/verify_weights.npz"
R = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
S = int(sys.argv[3]) if len(sys.argv) > 3 else 40
d = np.load(wf)
W, M, NIN, N, rates = d["W"], d["M"], int(d["NIN"]), int(d["N"]), d["rates"]
PRE, POST = W.shape
v_th, n_ref, beta_a = 0.61, 3, 0.0174
alpha, rho = np.exp(-1 / 10), np.exp(-1 / 300)
SIG = np.exp(-5.0)
T, KMAX = 30, 10
rng = np.random.default_rng(1)
cpsi = 0.3 / v_th


def psif(x):
    return cpsi * np.maximum(0, 1 - np.abs(x) / v_th)


def psi_of(v, A, refr):
    return np.where(refr, 0.0, psif(v - A)).astype(f32)


# ---- warm-up and targets: identical to verify_stein.py (same seeds) ----
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
ii, jj = np.nonzero(M)
ok = (zbar[ii] > 0.3)
ii, jj = ii[ok], jj[ok]
pick = rng.choice(len(ii), min(S, len(ii)), replace=False)
ti, tj = ii[pick], jj[pick]
S = len(ti)
ZIN = (rng.random((T, NIN)) < rates).astype(f32)
NOISE = np.random.default_rng(7).standard_normal((T, R, POST)).astype(f32)

B = R
vB = np.repeat(v[None], B, 0); adB = np.repeat(ad[None], B, 0); zsB = np.repeat(zs[None], B, 0)
refB = np.repeat(ref[None], B, 0); zbarB = np.repeat(zbar[None], B, 0)
epsS = np.repeat(epsA[ti, tj][None], B, 0)
xiS = np.zeros((B, S), f32)
X = np.zeros((T, B, S), f32); Z = np.zeros((T, B, S), f32); PS = np.zeros((T, B, S), f32)
PI = np.zeros((T, B, S), f32); RF = np.zeros((T, B, S), bool); SC = np.zeros((T, B, S), f32)
t0 = time.time()
for t in range(T):
    zpre = np.concatenate([np.broadcast_to(ZIN[t], (B, NIN)), zsB[:, :N]], 1)
    nz = NOISE[t]
    I = zpre @ W + SIG * nz * (zpre @ M)
    adB = rho * adB + zsB; A = v_th + beta_a * adB
    vB = alpha * vB + I - v_th * zsB
    refB = np.maximum(refB - 1, 0); refr = refB > 0
    zsB = ((vB >= A) & ~refr).astype(f32); refB = np.where(zsB > 0, n_ref, refB)
    psi = psi_of(vB, A, refr)
    zbarB = alpha * zbarB + zpre
    zi = zpre[:, ti]; ps = psi[:, tj]; zb = zbarB[:, ti]
    pi_ = zb - beta_a * epsS
    e = ps * pi_
    epsS = ps * zb + (rho - ps * beta_a) * epsS
    xiS = xiS + SIG * nz[:, tj] * zi
    SC[t] = e * xiS / SIG ** 2
    xiS *= alpha
    X[t] = (vB - A)[:, tj]; Z[t] = zsB[:, tj]; PS[t] = ps; PI[t] = pi_; RF[t] = refr[:, tj]
print(f"rollouts done in {time.time() - t0:.0f}s")

# ---- deterministic second-order trace ----
slope = np.where(X < 0, cpsi / v_th, -cpsi / v_th) * (np.abs(X) < v_th) * (~RF)
term1 = PI * slope * PI
term2 = np.zeros_like(term1)
for t in range(T):
    acc = np.zeros((B, S), f32)
    for s in range(max(0, t - KMAX), t):
        dd = t - s
        spk = Z[s] > 0
        shift = v_th * alpha ** dd
        if dd <= 2:
            # with a spike at s the gate is closed at t; without it the gate is open (undo the reset if it happened)
            psi_without = np.where(spk, psif(X[t] + shift), PS[t])
            dpsi = -psi_without
        else:
            psi_with = np.where(spk, PS[t], np.where(RF[t], 0.0, psif(X[t] - shift)))
            psi_without = np.where(spk, np.where(RF[t], 0.0, psif(X[t] + shift)), PS[t])
            dpsi = psi_with - psi_without
        acc += dpsi * PS[s] * PI[s]
    term2[t] = PI[t] * acc

A_ = SC.mean(1)
late = slice(5, T)


def stats(name, Dm):
    x, y = Dm[late].ravel(), A_[late].ravel()
    c = np.corrcoef(x, y)[0, 1]; k = (x * y).sum() / (x * x).sum()
    print(f"   {name:32s} corr {c:+.3f}   R^2 {c * c:.3f}   slope {k:.2f}")


print("\nE[deterministic trace] vs measured drift E[s] (all t >= 5, 40 synapses):")
stats("term1 (subthreshold slope)", term1.mean(1))
stats("term2 (spike events)", term2.mean(1))
stats("term1 + term2", (term1 + term2).mean(1))
# best linear combination (upper bound for this pair of terms)
Xm = np.stack([term1.mean(1)[late].ravel(), term2.mean(1)[late].ravel()], 1); y = A_[late].ravel()
coef, *_ = np.linalg.lstsq(Xm, y, rcond=None)
r2 = 1 - ((y - Xm @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"   best weighting a*term1 + b*term2: a={coef[0]:.2f} b={coef[1]:.2f}  R^2 {r2:.3f}")
ref_ = np.load("data/verify_stein_out.npz")
print(f"   (reference: noise-measured weight derivative dE[e]/dW explained R^2 0.906 of E[s])")
