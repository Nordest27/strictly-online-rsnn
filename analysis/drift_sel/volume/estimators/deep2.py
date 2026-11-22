"""Depth test: two spiking hidden layers (time-locked: L2 integrates L1 spikes step by step through a rate trace) on the
narrow no-columns pick task (bridge pick; H = 24 per layer with HC=4; spiking motor competition from L2 with pool
homeostasis; per-weight Adam; count gate). Question: does credit for L1 work with depth, and which kind?

Credit kinds:
  route : motor group credit Cm routed back through reward-free spike-learned links: S2 (motor drive on h2),
          S1 (L2 input current on h1). cr2 = S2^T Cm * g2; cr1 = S1^T cr2 * g1 (each RMS-normalised)
  half  : each layer gets its own shared-noise channel (groups = learned sensitivity: L2 from S2, L1 from S2 . S1),
          on only in the second half of the window; clean estimate h2_c = 2 * first-half L2 counts (both layers
          noise-free then); dG = 1[a != a_c] (r - Q(h2_c)[a_c]) centred; cr_l = dG * X_l * g_l
Modes: frozen (no hidden learning), route, half, route_top / half_top (L1 frozen: only L2 learns),
       route_half (route + LAM * half, each RMS-normalised).
Direct feedback (one hop from the motor neurons to every hidden layer, no relaying):
  dfa        : fixed random links B_l (motor x layer l, N(0,1)): cr_l = B_l^T Cm * g_l (reward-based e-prop's feedback)
  dfa_sparse : the same, each link present with probability DFA_P (0.25)
  ldfa       : learned direct links: D_l regresses the motor drive on h_l (reward-free NLMS, as S2), cr_l = D_l^T Cm * g_l
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TASK", "pick")
import bridge as B

K, NE, NM, H, D, M = B.K, B.NE, B.NM, B.H, B.D, B.M
NTYPE, NCUE, DC = B.NTYPE, B.NCUE, B.DC
SIGH = float(os.environ.get("SIGH", "0.06")); LAM = float(os.environ.get("LAM", "0.5"))
TAU = 0.9                                         # L1 -> L2 rate trace


class RMS:
    def __init__(self): self.v = {}
    def __call__(self, k, x):
        self.v[k] = 0.99 * self.v.get(k, (x ** 2).mean() + 1e-12) + 0.01 * (x ** 2).mean()
        return x / np.sqrt(self.v[k] + 1e-12)


def run(mode, trials=int(os.environ.get("TR", "40000")), seed=int(os.environ.get("SEED", "0"))):
    rng = np.random.default_rng(seed); ar = np.arange(M)
    Vtab = np.random.default_rng(500 + seed).random((NCUE, NTYPE))
    W1 = rng.standard_normal((M, H, D)) / np.sqrt(DC); b1 = np.zeros((M, H))
    W2 = rng.standard_normal((M, H, H)) / np.sqrt(H) * 2.0; b2 = np.zeros((M, H))
    W3 = 0.3 * rng.standard_normal((M, NM, H)) / np.sqrt(B.HC)
    S2 = np.zeros((M, NM, H)); S1 = np.zeros((M, H, H)); D1 = np.zeros((M, NM, H))
    brng = np.random.default_rng(900 + seed)
    Bf1 = brng.standard_normal((M, NM, H)); Bf2 = brng.standard_normal((M, NM, H))
    if mode == "dfa_sparse":
        p = float(os.environ.get("DFA_P", "0.25"))
        Bf1 *= brng.random(Bf1.shape) < p; Bf2 *= brng.random(Bf2.shape) < p
    m1 = np.zeros((M, H)); m2 = np.zeros((M, H)); md = np.zeros((M, NM)); ma = np.zeros((M, H))
    Wq = np.zeros((M, K, H + 1)); base = np.full(M, 0.4); hb = np.zeros((M, K)); dgm = np.zeros(M)
    rms = RMS(); rew, best = [], []
    use_half = "half" in mode; use_route = "route" in mode
    l1_learns = mode not in ("frozen", "route_top", "half_top"); l2_learns = mode != "frozen"
    on = np.ones(B.T); on[: B.T // 2] = 0.0
    for t in range(trials):
        cue = rng.integers(0, NCUE, M); u = np.zeros((M, K, DC)); val = np.zeros((M, K))
        u[:, :, NTYPE + 1:] = np.eye(NCUE)[cue][:, None, :]
        for i in range(M):
            n = rng.integers(2, K + 1); locs = rng.choice(K, n, replace=False); ty = rng.choice(NTYPE, n, replace=False)
            u[i, locs, ty] = 1.0; u[i, locs, NTYPE] = 1.0; val[i, locs] = Vtab[cue[i], ty]
        u = u.reshape(M, D)
        # shared-noise groups from the learned sensitivities
        v2 = S2.reshape(M, K, NE, H).sum(2); v1 = np.einsum("mkh,mhj->mkj", v2, S1)
        Mix2 = np.eye(K)[v2.argmax(1)]; Mix1 = np.eye(K)[v1.argmax(1)]
        G1 = np.einsum("mis,tms->tmi", Mix1, rng.standard_normal((B.T, M, K))) * on[:, None, None]
        G2 = np.einsum("mis,tms->tmi", Mix2, rng.standard_normal((B.T, M, K))) * on[:, None, None]
        if not use_half:
            G1 = rng.standard_normal((B.T, M, H)); G2 = rng.standard_normal((B.T, M, H))   # private noise, same power
        X1 = G1.sum(0) / np.sqrt(on.sum()); X2 = G2.sum(0) / np.sqrt(on.sum())
        I1 = B.I0 + B.CIN * (np.einsum("mhd,md->mh", W1, u) + b1)
        V1 = np.zeros((M, H)); R1 = np.zeros((M, H)); V2 = np.zeros((M, H)); R2 = np.zeros((M, H)); tr = np.zeros((M, H))
        c1 = np.zeros((M, H)); c2 = np.zeros((M, H)); c2a = np.zeros((M, H)); a2s = np.zeros((M, H))
        for k in range(B.T):
            V1 = B.ALPHA * V1 + I1 + SIGH * G1[k]; V1 = np.where(R1 > 0, B.ALPHA * V1, V1)
            z1 = (V1 >= 1.0) & (R1 <= 0); V1 = V1 - z1; R1 = np.where(z1, 2, R1 - 1); c1 += z1
            tr = TAU * tr + (1 - TAU) * z1 / B.RNORM
            a2 = np.einsum("mij,mj->mi", W2, tr) + b2; a2s += a2
            V2 = B.ALPHA * V2 + B.I0 + B.CIN * a2 + SIGH * G2[k]; V2 = np.where(R2 > 0, B.ALPHA * V2, V2)
            z2 = (V2 >= 1.0) & (R2 <= 0); V2 = V2 - z2; R2 = np.where(z2, 2, R2 - 1); c2 += z2
            if k < B.T // 2:
                c2a += z2
        h1 = c1 / (B.T * B.RNORM); h2 = c2 / (B.T * B.RNORM); a2m = a2s / B.T
        g1 = (c1 > 0).astype(float); g2 = (c2 > 0).astype(float)
        drive = np.einsum("mjh,mh->mj", W3, h2) + np.repeat(hb, NE, 1)
        a, xm = B.motor_spk(B.M0 + B.CM * drive, rng)
        r = val[ar, a]; d = r - base
        Cm = np.repeat((d[:, None] * xm / B.SIGM).reshape(M, K, NE).mean(2), NE, 1) * B.CM
        # links (reward-free regressions)
        h2c = h2 - m2; dc = drive - md; h1c = h1 - m1; ac = a2m - ma
        m2 += 0.01 * (h2 - m2); md += 0.01 * (drive - md); m1 += 0.01 * (h1 - m1); ma += 0.01 * (a2m - ma)
        S2 += B.LINK * np.einsum("mj,mh->mjh", dc - np.einsum("mjh,mh->mj", S2, h2c), h2c) / ((h2c ** 2).sum(1)[:, None, None] + 1e-6)
        S1 += B.LINK * np.einsum("mj,mh->mjh", ac - np.einsum("mjh,mh->mj", S1, h1c), h1c) / ((h1c ** 2).sum(1)[:, None, None] + 1e-6)
        cr1 = np.zeros((M, H)); cr2 = np.zeros((M, H))
        if use_route:
            r2 = np.einsum("mjh,mj->mh", S2, Cm) * g2; r1 = np.einsum("mjh,mj->mh", S1, r2) * g1
            cr2 += rms("r2", r2); cr1 += rms("r1", r1)
        if mode in ("dfa", "dfa_sparse"):
            cr2 += rms("d2", np.einsum("mjh,mj->mh", Bf2, Cm) * g2); cr1 += rms("d1", np.einsum("mjh,mj->mh", Bf1, Cm) * g1)
        if mode == "ldfa":       # D1: motor drive regressed directly on h1 (total sensitivity through L2)
            D1 += B.LINK * np.einsum("mj,mh->mjh", dc - np.einsum("mjh,mh->mj", D1, h1c), h1c) / ((h1c ** 2).sum(1)[:, None, None] + 1e-6)
            cr2 += rms("d2", np.einsum("mjh,mj->mh", S2, Cm) * g2); cr1 += rms("d1", np.einsum("mjh,mj->mh", D1, Cm) * g1)
        if use_half:
            h2_c = 2 * c2a / (B.T * B.RNORM); hq_c = np.concatenate([h2_c, np.ones((M, 1))], 1)
            Qc = np.einsum("mkh,mh->mk", Wq, hq_c)
            a_c = (np.einsum("mjh,mh->mj", W3, h2_c).reshape(M, K, NE).sum(2) + NE * hb).argmax(1)
            dG = (a != a_c) * (r - Qc[ar, a_c]); dgm += 0.01 * (dG - dgm); dG = dG - dgm
            w = LAM if use_route else 1.0
            cr2 += w * rms("n2", dG[:, None] * X2 * g2); cr1 += w * rms("n1", dG[:, None] * X1 * g1)
        # learning
        hq = np.concatenate([h2, np.ones((M, 1))], 1); qa = np.einsum("mh,mh->m", Wq[ar, a], hq)
        Wq[ar, a] += 0.1 * (r - qa)[:, None] * hq / ((hq ** 2).sum(1, keepdims=True) + 1e-6)
        hb += 0.003 * (1.0 / K - np.eye(K)[a]); base += 0.01 * (r - base)
        rew.append(r); best.append(val.max(1))
        B.step("W3", W3, np.einsum("mj,mh->mjh", Cm, h2), t)
        if l2_learns:
            B.step("W2", W2, np.einsum("mh,mj->mhj", cr2, tr * 0 + h1), t); B.step("b2", b2, cr2, t)
        if l1_learns:
            B.step("W1", W1, np.einsum("mh,md->mhd", cr1, u), t); B.step("b1", b1, cr1, t)
    R = np.array(rew).reshape(-1, 2000, M).mean((1, 2)) / np.array(best).reshape(-1, 2000, M).mean((1, 2))
    run.hz = (float(c1.mean() / B.T * 1000), float(c2.mean() / B.T * 1000))
    return R


if __name__ == "__main__":
    for mode in sys.argv[1:]:
        R = run(mode)
        print(f"{mode:10s} reward/best per 2000: " + " ".join(f"{x:.2f}" for x in R)
              + f"  | Hz L1 {run.hz[0]:.0f} L2 {run.hz[1]:.0f}", flush=True)
