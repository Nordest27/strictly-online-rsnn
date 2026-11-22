"""Dynamic noise groups on the no-columns spiking pick task (bridge.py setup: 6 locations, 2..6 items, cue-dependent
values; 96 spiking hidden neurons that see everything and drive every motor group; spiking motor competition with pool
homeostasis; per-weight Adam). The hidden neurons learn from their OWN injected noise (Stein: delta * sum_t xi_h /
sigma, activity-gated); the motor stage from its group-pooled noise credit (as in bridge). What changes is the
coupling of the hidden noise:
  indep   : independent noise per neuron
  fixed   : neurons at random positions on a ring; noise = Gaussian-kernel mixture of white noise (width SIG_K,
            unit variance per neuron): fixed local groups
  waves   : the same, but the positions random-walk slowly (rate WAVE per trial): the groupings drift
  plastic : coupling C (non-negative, row-normalised to unit norm): C_ij grows when the two neurons' credits agree,
            dC_ij = ETA_C * (c_i c_j / <c^2> - DECAY * C_ij); starts as the fixed ring kernel
  waves_plastic : the ring kernel drifts AND a plastic part is added: C = kernel(t) + P, P learned as above
Measured: reward / best (per 2000 trials), and the column index: how concentrated each hidden neuron's influence is on
one location (max share of |W2| over locations, chance 1/6) and how much hidden neurons that are strongly coupled share
their preferred location.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TASK", "pick")
import bridge as B

K, NE, NM, H, D, M = B.K, B.NE, B.NM, B.H, B.D, B.M
NTYPE, NCUE, DC = B.NTYPE, B.NCUE, B.DC
SIGH = float(os.environ.get("SIGH", "0.06")); SIG_K = float(os.environ.get("SIG_K", "0.03"))
WAVE = float(os.environ.get("WAVE", "0.002")); ETA_C = float(os.environ.get("ETA_C", "0.01")); DECAY = 0.1


def kernel(pos):
    d = np.abs(pos[:, :, None] - pos[:, None, :]); d = np.minimum(d, 1 - d)
    Kx = np.exp(-d ** 2 / (2 * SIG_K ** 2))
    return Kx


def norm_rows(C):
    return C / (np.sqrt((C ** 2).sum(2, keepdims=True)) + 1e-12)


def lif_coupled(I, C, rng):
    """Hidden LIF with coupled noise: xi = C @ white (C rows unit norm). Returns counts and the injected noise sums."""
    V = np.zeros(I.shape); ref = np.zeros(I.shape); cnt = np.zeros(I.shape); xs = np.zeros(I.shape)
    for _ in range(B.T):
        xi = np.einsum("mij,mj->mi", C, rng.standard_normal(I.shape)) if C is not None else rng.standard_normal(I.shape)
        xs += xi
        V = B.ALPHA * V + I + SIGH * xi; V = np.where(ref > 0, B.ALPHA * V, V)
        z = (V >= 1.0) & (ref <= 0); V = V - z; ref = np.where(z, 2, ref - 1); cnt += z
    return cnt, xs


def run(mode, trials=int(os.environ.get("TR", "40000")), seed=int(os.environ.get("SEED", "0"))):
    rng = np.random.default_rng(seed)
    Vtab = np.random.default_rng(500 + seed).random((NCUE, NTYPE))
    W1 = rng.standard_normal((M, H, D)) / np.sqrt(DC); b1 = np.zeros((M, H))
    W2 = 0.3 * rng.standard_normal((M, NM, H)) / np.sqrt(B.HC)
    pos = rng.random((M, H)); P = np.zeros((M, H, H)); csq = 1.0
    base = np.full(M, 0.4); hb = np.zeros((M, K)); rew, best = [], []
    for t in range(trials):
        cue = rng.integers(0, NCUE, M); u = np.zeros((M, K, DC)); val = np.zeros((M, K))
        u[:, :, NTYPE + 1:] = np.eye(NCUE)[cue][:, None, :]
        for i in range(M):
            n = rng.integers(2, K + 1); locs = rng.choice(K, n, replace=False); ty = rng.choice(NTYPE, n, replace=False)
            u[i, locs, ty] = 1.0; u[i, locs, NTYPE] = 1.0; val[i, locs] = Vtab[cue[i], ty]
        u = u.reshape(M, D)
        if mode in ("indep", "frozen"):
            C = None
        else:
            if mode.endswith("waves") or mode.endswith("waves_plastic"):
                pos = (pos + WAVE * rng.standard_normal((M, H))) % 1.0
            Kx = kernel(pos)
            C = norm_rows(Kx + (P if mode.endswith("plastic") else 0.0))
        a1 = np.einsum("mhd,md->mh", W1, u) + b1
        cnt, xs = lif_coupled(B.I0 + B.CIN * a1, C, rng); h = cnt / (B.T * B.RNORM); g = (cnt > 0).astype(float)
        drive = np.einsum("mjh,mh->mj", W2, h) + np.repeat(hb, NE, 1)
        a, xm = B.motor_spk(B.M0 + B.CM * drive, rng)
        hb += 0.003 * (1.0 / K - np.eye(K)[a])
        r = val[np.arange(M), a]; d = r - base; base += 0.01 * (r - base); rew.append(r); best.append(val.max(1))
        Cm = np.repeat((d[:, None] * xm / B.SIGM).reshape(M, K, NE).mean(2), NE, 1) * B.CM
        ch = d[:, None] * xs / SIGH                               # the hidden neuron's own (coupled) noise credit
        base_mode = mode.replace("pool_", "").replace("route_", "")
        if mode.startswith("route_"):                             # credit routed back from the motor groups
            cr = np.einsum("mjh,mj->mh", W2, Cm) * g
        else:
            cr = ch * g
        if "pool" in mode and C is not None:                       # the group pools its members' credit
            Cs = np.maximum(C, 0.0); Cs = Cs / (Cs.sum(2, keepdims=True) + 1e-12)
            cr = np.einsum("mij,mj->mi", Cs, cr)
        Ch = cr * B.CIN
        B.step("W2", W2, np.einsum("mj,mh->mjh", Cm, h), t)
        if mode != "frozen":
            B.step("W1", W1, np.einsum("mh,md->mhd", Ch, u), t); B.step("b1", b1, Ch, t)
        if mode.endswith("plastic"):                            # couple neurons whose credits agree (local: both ends)
            csq = 0.99 * csq + 0.01 * float(np.mean(ch ** 2))
            P += ETA_C * (np.einsum("mi,mj->mij", ch, ch) / csq - DECAY * P)
            P = np.maximum(P, 0.0); P[:, np.arange(H), np.arange(H)] = 0.0
    R = np.array(rew).reshape(-1, 2000, M).mean((1, 2)) / np.array(best).reshape(-1, 2000, M).mean((1, 2))
    share = np.abs(W2).reshape(M, K, NE, H).sum(2); share = share / (share.sum(1, keepdims=True) + 1e-12)
    run.col = float(share.max(1).mean())
    if C is not None:
        pref = share.argmax(1)
        Cn = C.copy(); Cn[:, np.arange(H), np.arange(H)] = 0
        top = Cn.argmax(2)                                       # each neuron's most strongly coupled partner
        run.same = float(np.mean(pref[np.arange(M)[:, None], top] == pref))
    else:
        run.same = float("nan")
    return R


if __name__ == "__main__":
    for mode in sys.argv[1:] or ["indep", "fixed", "waves", "plastic", "waves_plastic"]:
        R = run(mode)
        print(f"{mode:14s} reward/best per 2000: " + " ".join(f"{x:.2f}" for x in R[::2])
              + f"  | column index {run.col:.2f} (chance {1 / K:.2f})  coupled partner same location {run.same:.2f}",
              flush=True)
