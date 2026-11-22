"""Noise regions shaped by REWARD (not by sensitivity), and drift credit, on the narrow no-columns spiking pick task
(same setup as sensgroups.py: bridge.py pick, HC=4 -> 24 hidden, spiking motor competition, pool homeostasis,
per-weight Adam, count gate).

Hidden noise is mixed from per-neuron white sources w (one per neuron): xi = A w, rows of A kept at unit norm (fixed
noise power per neuron, only the correlations change). Trial sums: W = sum_t w / sqrt(T), X = A W (unit variance).

Coupling rules (local: A_ij lives on the wire from source j to neuron i; uses xi_i, w_j and the broadcast delta):
  cov_reward : dA_ij = ETA_A * delta * (X_i W_j - A_ij)      exponential NES / CMA-like covariance step; the sign of
               delta is kept: joint pushes that beat the baseline are explored together more
  cov_sq     : the same with delta^2 instead of delta (the sign-free version, as dyngroups 'plastic')
  cov_act    : coupling from rewarded ACTIVITY fluctuations (drift-like): dA_ij = ETA_A * delta * a_i a_j (i != j),
               a = (count - running mean) / running sd
Hidden credit:
  noise modes (indep, cov_*): delta * X_i (coupled noise -> expected update = Sigma . grad: credit pooled over the region)
  drift      : delta * a_i (reward x own activity fluctuation; the fluctuation carries all upstream noise), indep noise
  drift_cov  : drift credit with cov_reward coupling of the noise
  frozen     : no hidden learning
Reported: reward / best per 2000 trials, mean |off-diagonal| correlation of the noise (A A^T), and how often a neuron's
most correlated partner prefers the same location (chance 1/6).
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TASK", "pick")
import bridge as B

K, NE, NM, H, D, M = B.K, B.NE, B.NM, B.H, B.D, B.M
NTYPE, NCUE, DC = B.NTYPE, B.NCUE, B.DC
SIGH = float(os.environ.get("SIGH", "0.06")); ETA_A = float(os.environ.get("ETA_A", "0.003"))


def run(mode, trials=int(os.environ.get("TR", "20000")), seed=int(os.environ.get("SEED", "0"))):
    rng = np.random.default_rng(seed)
    Vtab = np.random.default_rng(500 + seed).random((NCUE, NTYPE))
    W1 = rng.standard_normal((M, H, D)) / np.sqrt(DC); b1 = np.zeros((M, H))
    W2 = 0.3 * rng.standard_normal((M, NM, H)) / np.sqrt(B.HC)
    A = np.repeat(np.eye(H)[None], M, 0); eye = np.eye(H)
    hbar = np.zeros((M, H)); hvar = np.full((M, H), 1e-2); cbar = np.zeros((M, H))
    base = np.full(M, 0.4); hb = np.zeros((M, K)); rew, best = [], []
    learn_A = mode in ("cov_reward", "cov_sq", "cov_act", "drift_cov", "drift_cov_c")
    for t in range(trials):
        cue = rng.integers(0, NCUE, M); u = np.zeros((M, K, DC)); val = np.zeros((M, K))
        u[:, :, NTYPE + 1:] = np.eye(NCUE)[cue][:, None, :]
        for i in range(M):
            n = rng.integers(2, K + 1); locs = rng.choice(K, n, replace=False); ty = rng.choice(NTYPE, n, replace=False)
            u[i, locs, ty] = 1.0; u[i, locs, NTYPE] = 1.0; val[i, locs] = Vtab[cue[i], ty]
        u = u.reshape(M, D)
        a1 = np.einsum("mhd,md->mh", W1, u) + b1
        Vh = np.zeros((M, H)); ref = np.zeros((M, H)); cnt = np.zeros((M, H)); ws = np.zeros((M, H))
        I = B.I0 + B.CIN * a1
        for _ in range(B.T):
            w = rng.standard_normal((M, H)); ws += w
            xi = np.einsum("mij,mj->mi", A, w) if learn_A else w
            Vh = B.ALPHA * Vh + I + SIGH * xi; Vh = np.where(ref > 0, B.ALPHA * Vh, Vh)
            z = (Vh >= 1.0) & (ref <= 0); Vh = Vh - z; ref = np.where(z, 2, ref - 1); cnt += z
        Wn = ws / np.sqrt(B.T); X = np.einsum("mij,mj->mi", A, Wn) if learn_A else Wn
        h = cnt / (B.T * B.RNORM); g = (cnt > 0).astype(float)
        act = (cnt - hbar) / np.sqrt(hvar); hbar += 0.01 * (cnt - hbar); hvar += 0.01 * ((cnt - hbar) ** 2 - hvar)
        drive = np.einsum("mjh,mh->mj", W2, h) + np.repeat(hb, NE, 1)
        a, xm = B.motor_spk(B.M0 + B.CM * drive, rng)
        hb += 0.003 * (1.0 / K - np.eye(K)[a])
        r = val[np.arange(M), a]; d = r - base; base += 0.01 * (r - base); rew.append(r); best.append(val.max(1))
        Cm = np.repeat((d[:, None] * xm / B.SIGM).reshape(M, K, NE).mean(2), NE, 1) * B.CM
        if mode.startswith("drift"):
            cr = d[:, None] * act * g
            if mode.endswith("_c"):                          # centred drift: remove the push shared by all trials
                cbar += 0.01 * (cr - cbar); cr = cr - cbar
        elif mode == "frozen":
            cr = np.zeros((M, H))
        else:
            cr = d[:, None] * X * g
        Ch = cr * B.CIN
        B.step("W2", W2, np.einsum("mj,mh->mjh", Cm, h), t)
        if mode != "frozen":
            B.step("W1", W1, np.einsum("mh,md->mhd", Ch, u), t); B.step("b1", b1, Ch, t)
        if learn_A:
            if mode == "cov_act":
                dA = d[:, None, None] * np.einsum("mi,mj->mij", act, act) * (1 - eye)
            else:
                f = d ** 2 if mode == "cov_sq" else d
                dA = f[:, None, None] * (np.einsum("mi,mj->mij", X, Wn) - A)
            A += ETA_A * dA
            A /= np.sqrt((A ** 2).sum(2, keepdims=True)) + 1e-12
    R = np.array(rew).reshape(-1, 2000, M).mean((1, 2)) / np.array(best).reshape(-1, 2000, M).mean((1, 2))
    Sg = np.einsum("mik,mjk->mij", A, A); off = Sg * (1 - eye)
    run.corr = float(np.abs(off).sum((1, 2)).mean() / (H * (H - 1)))
    share = np.abs(W2).reshape(M, K, NE, H).sum(2); share = share / (share.sum(1, keepdims=True) + 1e-12)
    pref = share.argmax(1); top = off.argmax(2)
    run.same = float(np.mean(pref[np.arange(M)[:, None], top] == pref)) if learn_A else float("nan")
    run.hz = float(cnt.mean() / B.T * 1000)
    return R


if __name__ == "__main__":
    for mode in sys.argv[1:] or ["frozen", "indep", "cov_reward", "cov_sq", "cov_act", "drift", "drift_cov"]:
        R = run(mode)
        print(f"{mode:11s} reward/best per 2000: " + " ".join(f"{x:.2f}" for x in R)
              + f"  | noise |corr| {run.corr:.3f}  top partner same location {run.same:.2f}  hidden Hz {run.hz:.0f}",
              flush=True)
