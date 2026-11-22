"""Does estimating the drift (d e/dW) with noise suffer from the number of perturbed neurons, like REINFORCE?

Trained grid-toy network, fixed input spike train, R noise rollouts of T steps from one state. The target
synapses' postsynaptic neurons are always perturbed; a fraction f of the other neurons is perturbed too.
Per target synapse ij, at the end of the window:
  drift estimate      s_ij = e_ij * xi_ij / sigma^2         (its mean is the drift P_ij)
  REINFORCE estimate  g_ij = V_T * xi_ij / sigma^2          (its mean is dE[V_T]/dW along the noise kernel;
                                                             V_T = value readout at the end of the window)
SNR = |mean| / sd over rollouts, median over synapses. Prediction: SNR(s) ~ flat in f, SNR(g) falls with f.
Usage: python3 verify_snr.py data/verify_weights.npz [R] [S]
"""
import sys
import numpy as np

f32 = np.float32
wf = sys.argv[1] if len(sys.argv) > 1 else "data/verify_weights.npz"
R = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
S = int(sys.argv[3]) if len(sys.argv) > 3 else 40
d = np.load(wf)
W, M, NIN, N, NF, rates, Wv = d["W"], d["M"], int(d["NIN"]), int(d["N"]), int(d["NF"]), d["rates"], d["Wv"]
PRE, POST = W.shape
v_th, n_ref, beta_a = 0.61, 3, 0.0174
alpha, rho, alpha_o = np.exp(-1 / 10), np.exp(-1 / 300), np.exp(-1 / 10)
SIG = np.exp(-5.0)
T = 30
rng = np.random.default_rng(1)


def psi_of(v, A, refr):
    return np.where(refr, 0.0, (0.3 / v_th) * np.maximum(0, 1 - np.abs(v - A) / v_th)).astype(f32)


v = np.zeros(POST, f32); ad = np.zeros(POST, f32); zs = np.zeros(POST, f32); ref = np.zeros(POST, np.int32)
zbar = np.zeros(PRE, f32); epsA = np.zeros((PRE, POST), f32); zin = np.zeros(NIN, f32); vv = np.zeros(1, f32)
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
    vv = alpha_o * vv + zs[N + NF:] @ Wv
    zin = (rng.random(NIN) < rates).astype(f32)
ii, jj = np.nonzero(M)
# targets: pilot run with all neurons perturbed; pick the 3 neurons whose own noise most detectably moves V_T
# and whose gate is open, then their active afferent synapses
def rollout(mask, Rr, seed, track=None):
    NO = np.random.default_rng(seed).standard_normal((T, Rr, POST)).astype(f32) * mask[None, None, :]
    vB = np.repeat(v[None], Rr, 0); adB = np.repeat(ad[None], Rr, 0); zsB = np.repeat(zs[None], Rr, 0)
    refB = np.repeat(ref[None], Rr, 0); vvB = np.repeat(vv[None], Rr, 0)
    eta = np.zeros((Rr, POST), f32); psum = np.zeros((Rr, POST), f32)
    for t in range(T):
        zpre = np.concatenate([np.broadcast_to(ZIN[t], (Rr, NIN)), zsB[:, :N]], 1)
        c = zpre @ M
        I = zpre @ W + SIG * NO[t] * c
        eta = alpha * eta + NO[t] * c
        adB = rho * adB + zsB; A = v_th + beta_a * adB
        vB = alpha * vB + I - v_th * zsB
        refB = np.maximum(refB - 1, 0); refr = refB > 0
        zsB = ((vB >= A) & ~refr).astype(f32); refB = np.where(zsB > 0, n_ref, refB)
        psum += psi_of(vB, A, refr)
        vvB = alpha_o * vvB + zsB[:, N + NF:] @ Wv
    return vvB[:, 0], eta, psum / T
ZIN = (rng.random((T, NIN)) < rates).astype(f32)
VT, eta, pmean = rollout(np.ones(POST, bool), 3000, 11)
gsnr = np.abs((VT[:, None] * eta).mean(0)) / ((VT[:, None] * eta).std(0) + 1e-30)
score_n = gsnr * (pmean.mean(0) > 0.02)
top = np.argsort(-score_n)[:3]
print("pilot: chosen neurons", top, "REINFORCE SNR", np.round(gsnr[top], 3), "mean psi", np.round(pmean.mean(0)[top], 3))
ok = (zbar[ii] > 0.05) & np.isin(jj, top)
ii, jj = ii[ok], jj[ok]
pick = rng.choice(len(ii), min(S, len(ii)), replace=False)
ti, tj = ii[pick], jj[pick]
S = len(ti)
print(f"{S} target synapses; {POST} neurons can be perturbed")

target_post = np.zeros(POST, bool); target_post[np.unique(tj)] = True
others = np.nonzero(~target_post)[0]
order = np.random.default_rng(3).permutation(others)
print(f"{'fraction of other neurons perturbed':>36s} | {'SNR drift estimate':>18s} | {'SNR REINFORCE':>14s} | sd(V_T)")
for frac in (0.0, 0.25, 0.5, 1.0):
    mask = target_post.copy(); mask[order[:int(round(frac * len(order)))]] = True
    NOISE = np.random.default_rng(7).standard_normal((T, R, POST)).astype(f32) * mask[None, None, :]
    vB = np.repeat(v[None], R, 0); adB = np.repeat(ad[None], R, 0); zsB = np.repeat(zs[None], R, 0)
    refB = np.repeat(ref[None], R, 0); zbarB = np.repeat(zbar[None], R, 0); vvB = np.repeat(vv[None], R, 0)
    epsS = np.repeat(epsA[ti, tj][None], R, 0); xiS = np.zeros((R, S), f32)
    for t in range(T):
        zpre = np.concatenate([np.broadcast_to(ZIN[t], (R, NIN)), zsB[:, :N]], 1)
        nz = NOISE[t]
        I = zpre @ W + SIG * nz * (zpre @ M)
        adB = rho * adB + zsB; A = v_th + beta_a * adB
        vB = alpha * vB + I - v_th * zsB
        refB = np.maximum(refB - 1, 0); refr = refB > 0
        zsB = ((vB >= A) & ~refr).astype(f32); refB = np.where(zsB > 0, n_ref, refB)
        psi = psi_of(vB, A, refr)
        zbarB = alpha * zbarB + zpre
        ps = psi[:, tj]; zb = zbarB[:, ti]
        e = ps * (zb - beta_a * epsS)
        epsS = ps * zb + (rho - ps * beta_a) * epsS
        xiS = xiS + SIG * nz[:, tj] * zpre[:, ti]
        s_end = e * xiS / SIG ** 2
        if t == 0: s_sum = np.zeros_like(s_end)
        if t >= 5: s_sum += s_end
        xiS_end = xiS.copy()
        xiS *= alpha
        vvB = alpha_o * vvB + zsB[:, N + NF:] @ Wv
    g = vvB[:, :1] * xiS_end / SIG ** 2
    snr = lambda x: np.median(np.abs(x.mean(0)) / (x.std(0) + 1e-30))
    se_floor = 1 / np.sqrt(R)
    print(f"{frac:36.2f} | {snr(s_sum):18.3f} | {snr(g):14.3f} | {vvB.std():.4f}   ({mask.sum()} neurons perturbed)")
print(f"(SNR resolution ~ {1/np.sqrt(R):.3f}: values near it are indistinguishable from zero)")

# neuron-level REINFORCE (V_T x the neuron's own accumulated noise) for the 3 target neurons, more power
print("\nneuron-level REINFORCE SNR for the target neurons (median of 3), 20000 rollouts:")
for frac in (0.0, 0.25, 0.5, 1.0):
    mask = target_post.copy(); mask[order[:int(round(frac * len(order)))]] = True
    VT, eta, _ = rollout(mask, 20000, 21)
    gg = VT[:, None] * eta[:, top]
    print(f"   {frac:4.2f} ({mask.sum():2d} perturbed): SNR {np.median(np.abs(gg.mean(0)) / gg.std(0)):.4f}   "
          f"(resolution {1/np.sqrt(20000):.4f})")
