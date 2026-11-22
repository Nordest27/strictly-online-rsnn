"""Noise / credit regions defined by learned output sensitivity (task-free), on the narrow no-columns spiking pick task
(bridge.py: 6 locations, 2..6 items, cue-dependent values; H hidden neurons (HC=4 -> 24) that see everything and drive
every motor group; spiking motor competition with pool homeostasis; per-weight Adam; activity gate).

Links S (motor x hidden) learn reward-free to regress the motor drive on the hidden activity (as bridge route_link).
Each hidden neuron's sensitivity vector v_i = its links summed per motor group (K numbers: how it moves each action).
Region coupling A_ij = max(0, cos(v_i, v_j)) ** POW (A_ii = 1): neurons that move the output the same way.
Modes:
  frozen, route_sym (true W2^T), route_link (S^T): controls without regions
  sens_route     : S^T credit pooled within regions (A normalised to sum 1)
  sens_route_coh : the same, and the hidden noise is coherent within regions (A normalised to unit norm)
  sens_noise     : the hidden neurons' own coherent noise credit (delta * sum xi / sigma), pooled within regions
Reported: reward / best per 2000 trials, the column index (how concentrated each hidden neuron's |W2| is on one
location, chance 1/6) and how often a neuron's strongest region partner prefers the same location.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TASK", "pick")
import bridge as B

K, NE, NM, H, D, M = B.K, B.NE, B.NM, B.H, B.D, B.M
NTYPE, NCUE, DC = B.NTYPE, B.NCUE, B.DC
SIGH = float(os.environ.get("SIGH", "0.02")); POW = float(os.environ.get("POW", "4"))


def run(mode, trials=int(os.environ.get("TR", "20000")), seed=int(os.environ.get("SEED", "0"))):
    rng = np.random.default_rng(seed)
    Vtab = np.random.default_rng(500 + seed).random((NCUE, NTYPE))
    W1 = rng.standard_normal((M, H, D)) / np.sqrt(DC); b1 = np.zeros((M, H))
    W2 = 0.3 * rng.standard_normal((M, NM, H)) / np.sqrt(B.HC)
    S = np.zeros((M, NM, H)); hbar = np.zeros((M, H)); dbar = np.zeros((M, NM))
    base = np.full(M, 0.4); hb = np.zeros((M, K)); rew, best = [], []
    A = np.repeat(np.eye(H)[None], M, 0)
    for t in range(trials):
        cue = rng.integers(0, NCUE, M); u = np.zeros((M, K, DC)); val = np.zeros((M, K))
        u[:, :, NTYPE + 1:] = np.eye(NCUE)[cue][:, None, :]
        for i in range(M):
            n = rng.integers(2, K + 1); locs = rng.choice(K, n, replace=False); ty = rng.choice(NTYPE, n, replace=False)
            u[i, locs, ty] = 1.0; u[i, locs, NTYPE] = 1.0; val[i, locs] = Vtab[cue[i], ty]
        u = u.reshape(M, D)
        if mode.startswith("sens"):                       # regions from the learned sensitivities (task-free)
            v = S.reshape(M, K, NE, H).sum(2)
            vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
            A = np.maximum(np.einsum("mki,mkj->mij", vn, vn), 0.0) ** POW
            A[:, np.arange(H), np.arange(H)] = 1.0
        coh = mode in ("sens_route_coh", "sens_noise")
        Cn = A / np.sqrt((A ** 2).sum(2, keepdims=True)) if coh else None
        a1 = np.einsum("mhd,md->mh", W1, u) + b1
        Vh = np.zeros((M, H)); ref = np.zeros((M, H)); cnt = np.zeros((M, H)); xs = np.zeros((M, H))
        I = B.I0 + B.CIN * a1
        for _ in range(B.T):
            w = rng.standard_normal((M, H))
            xi = np.einsum("mij,mj->mi", Cn, w) if coh else w
            xs += xi
            Vh = B.ALPHA * Vh + I + SIGH * xi; Vh = np.where(ref > 0, B.ALPHA * Vh, Vh)
            z = (Vh >= 1.0) & (ref <= 0); Vh = Vh - z; ref = np.where(z, 2, ref - 1); cnt += z
        h = cnt / (B.T * B.RNORM); g = (cnt > 0).astype(float)
        drive = np.einsum("mjh,mh->mj", W2, h) + np.repeat(hb, NE, 1)
        a, xm = B.motor_spk(B.M0 + B.CM * drive, rng)
        hb += 0.003 * (1.0 / K - np.eye(K)[a])
        r = val[np.arange(M), a]; d = r - base; base += 0.01 * (r - base); rew.append(r); best.append(val.max(1))
        Cm = np.repeat((d[:, None] * xm / B.SIGM).reshape(M, K, NE).mean(2), NE, 1) * B.CM
        # links: reward-free regression of the motor drive on the hidden activity
        hc = h - hbar; dc = drive - dbar; hbar += 0.01 * (h - hbar); dbar += 0.01 * (drive - dbar)
        S += B.LINK * np.einsum("mj,mh->mjh", dc - np.einsum("mjh,mh->mj", S, hc), hc) / ((hc ** 2).sum(1)[:, None, None] + 1e-6)
        if mode == "route_sym":
            cr = np.einsum("mjh,mj->mh", W2, Cm) * g
        elif mode in ("route_link", "sens_route", "sens_route_coh"):
            cr = np.einsum("mjh,mj->mh", S, Cm) * g
        elif mode == "sens_noise":
            cr = d[:, None] * xs / SIGH * g * B.CM
        else:
            cr = np.zeros((M, H))
        if mode.startswith("sens"):                       # pool within the region
            As = A / A.sum(2, keepdims=True)
            cr = np.einsum("mij,mj->mi", As, cr)
        Ch = cr * B.CIN
        B.step("W2", W2, np.einsum("mj,mh->mjh", Cm, h), t)
        if mode != "frozen":
            B.step("W1", W1, np.einsum("mh,md->mhd", Ch, u), t); B.step("b1", b1, Ch, t)
    R = np.array(rew).reshape(-1, 2000, M).mean((1, 2)) / np.array(best).reshape(-1, 2000, M).mean((1, 2))
    share = np.abs(W2).reshape(M, K, NE, H).sum(2); share = share / (share.sum(1, keepdims=True) + 1e-12)
    run.col = float(share.max(1).mean())
    pref = share.argmax(1); An = A.copy(); An[:, np.arange(H), np.arange(H)] = -1
    run.same = float(np.mean(pref[np.arange(M)[:, None], An.argmax(2)] == pref)) if mode.startswith("sens") else float("nan")
    run.hz = float(cnt.mean() / B.T * 1000)
    run.link = float(np.mean([np.corrcoef(S[i].ravel(), W2[i].ravel())[0, 1] for i in range(M)]))
    return R


if __name__ == "__main__":
    for mode in sys.argv[1:] or ["frozen", "route_sym", "route_link", "sens_route", "sens_route_coh", "sens_noise"]:
        R = run(mode)
        print(f"{mode:15s} reward/best per 2000: " + " ".join(f"{x:.2f}" for x in R)
              + f"  | column index {run.col:.2f} (chance {1 / K:.2f})  region partner same location {run.same:.2f}"
              + f"  hidden Hz {run.hz:.0f}  link corr {run.link:+.2f}", flush=True)
