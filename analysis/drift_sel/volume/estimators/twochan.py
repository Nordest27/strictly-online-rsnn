"""Two noise channels in the hidden layer: private (local) + shared (global/group), on the narrow no-columns spiking
pick task (as sensgroups.py: bridge pick, HC=4 -> 24 hidden, spiking motor competition, pool homeostasis, Adam, count
gate). Total hidden noise power fixed: xi = sqrt(1-F) * private + sqrt(F) * shared (each unit variance).
The hidden G credit reads ONLY the shared channel: delta * X_shared_i * gate (Stein on the low-rank channel: projected
gradient, lower variance). The private channel is left for a later drift term (unused here).
Shared-channel groups:
  indep     : F = 0, credit from the private noise (control)
  rand      : 6 random hard groups of 4 neurons, one source each
  all       : one source for all neurons (rank 1)
  reg_hard  : groups = each neuron's preferred motor group in the learned sensitivity (links S, reward-free) -> 6 sources
  reg_soft  : soft regions A = relu(cos(v_i, v_j))^POW, shared noise = A-mixed sources (rows unit norm)
  true_hard : groups = preferred location in the true |W2| (cheating control: perfect regions)
  frozen    : no hidden learning
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TASK", "pick")
import bridge as B

K, NE, NM, H, D, M = B.K, B.NE, B.NM, B.H, B.D, B.M
NTYPE, NCUE, DC = B.NTYPE, B.NCUE, B.DC
SIGH = float(os.environ.get("SIGH", "0.06")); F = float(os.environ.get("F", "0.5")); POW = float(os.environ.get("POW", "4"))
SHADOW = int(os.environ.get("SHADOW", "0"))   # 1: noise-free twin (same private + motor noise, no shared noise);
                                               # shared credit = 1[a != a_shadow] * (r - baseline)
                                               # 2: oracle r - r_shadow; 3: r - Q[a_shadow], Q_k = wq_k . h learned
                                               # (NLMS) from the actions actually taken (per-action value readout)


def mixing(mode, S, W2, rand_groups):
    """(M, H, H') matrix mapping shared sources to neurons, rows unit norm."""
    if mode == "all":
        return np.ones((M, H, 1))
    if mode == "rand":
        return np.eye(K)[rand_groups]
    if mode in ("reg_hard", "true_hard"):
        src = S if mode == "reg_hard" else np.abs(W2)
        v = src.reshape(M, K, NE, H).sum(2)
        return np.eye(K)[v.argmax(1)]
    if mode == "reg_soft":
        v = S.reshape(M, K, NE, H).sum(2); vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        A = np.maximum(np.einsum("mki,mkj->mij", vn, vn), 0.0) ** POW; A[:, np.arange(H), np.arange(H)] = 1.0
        return A / np.sqrt((A ** 2).sum(2, keepdims=True))
    return None


def run(mode, trials=int(os.environ.get("TR", "20000")), seed=int(os.environ.get("SEED", "0"))):
    rng = np.random.default_rng(seed)
    Vtab = np.random.default_rng(500 + seed).random((NCUE, NTYPE))
    W1 = rng.standard_normal((M, H, D)) / np.sqrt(DC); b1 = np.zeros((M, H))
    W2 = 0.3 * rng.standard_normal((M, NM, H)) / np.sqrt(B.HC)
    S = np.zeros((M, NM, H)); hbar = np.zeros((M, H)); dbar = np.zeros((M, NM))
    rand_groups = np.stack([rng.permutation(np.repeat(np.arange(K), H // K)) for _ in range(M)])
    base = np.full(M, 0.4); hb = np.zeros((M, K)); rew, best = [], []
    f = 0.0 if mode in ("indep", "frozen") else F
    run.flips = []; Wq = np.zeros((M, K, H + 1)); run.qerr = []
    for t in range(trials):
        cue = rng.integers(0, NCUE, M); u = np.zeros((M, K, DC)); val = np.zeros((M, K))
        u[:, :, NTYPE + 1:] = np.eye(NCUE)[cue][:, None, :]
        for i in range(M):
            n = rng.integers(2, K + 1); locs = rng.choice(K, n, replace=False); ty = rng.choice(NTYPE, n, replace=False)
            u[i, locs, ty] = 1.0; u[i, locs, NTYPE] = 1.0; val[i, locs] = Vtab[cue[i], ty]
        u = u.reshape(M, D)
        Mix = mixing(mode, S, W2, rand_groups) if f > 0 else None
        a1 = np.einsum("mhd,md->mh", W1, u) + b1
        I = B.I0 + B.CIN * a1
        P = rng.standard_normal((B.T, M, H))
        G = np.einsum("mis,tms->tmi", Mix, rng.standard_normal((B.T, M, Mix.shape[2]))) if Mix is not None else np.zeros((B.T, M, H))
        xp = P.sum(0); xg = G.sum(0)
        def hidden(shared_on):
            Vh = np.zeros((M, H)); ref = np.zeros((M, H)); cnt = np.zeros((M, H))
            for k in range(B.T):
                Vh = B.ALPHA * Vh + I + SIGH * (np.sqrt(1 - f) * P[k] + (np.sqrt(f) * G[k] if shared_on else 0.0))
                Vh = np.where(ref > 0, B.ALPHA * Vh, Vh)
                z = (Vh >= 1.0) & (ref <= 0); Vh = Vh - z; ref = np.where(z, 2, ref - 1); cnt += z
            return cnt
        cnt = hidden(True)
        h = cnt / (B.T * B.RNORM); g = (cnt > 0).astype(float)
        drive = np.einsum("mjh,mh->mj", W2, h) + np.repeat(hb, NE, 1)
        mseed = int(rng.integers(2 ** 62))
        a, xm = B.motor_spk(B.M0 + B.CM * drive, np.random.default_rng(mseed))
        if SHADOW and Mix is not None:
            hs = hidden(False) / (B.T * B.RNORM)
            a_sh, _ = B.motor_spk(B.M0 + B.CM * (np.einsum("mjh,mh->mj", W2, hs) + np.repeat(hb, NE, 1)),
                                  np.random.default_rng(mseed))
            flip = (a != a_sh).astype(float); run.flips.append(flip.mean())
        else:
            flip = np.ones(M)
        hb += 0.003 * (1.0 / K - np.eye(K)[a])
        r = val[np.arange(M), a]; d = r - base; base += 0.01 * (r - base); rew.append(r); best.append(val.max(1))
        Cm = np.repeat((d[:, None] * xm / B.SIGM).reshape(M, K, NE).mean(2), NE, 1) * B.CM
        hc = h - hbar; dc = drive - dbar; hbar += 0.01 * (h - hbar); dbar += 0.01 * (drive - dbar)
        S += B.LINK * np.einsum("mj,mh->mjh", dc - np.einsum("mjh,mh->mj", S, hc), hc) / ((hc ** 2).sum(1)[:, None, None] + 1e-6)
        X = xp if f == 0 else xg
        if SHADOW == 3 and Mix is not None:
            hq = np.concatenate([h, np.ones((M, 1))], 1); Q = np.einsum("mkh,mh->mk", Wq, hq)
            q_sh = Q[np.arange(M), a_sh]; dG = (r - q_sh) * flip
            if flip.any():
                run.qerr.append(np.abs(q_sh - val[np.arange(M), a_sh])[flip > 0].mean())
            err = r - Q[np.arange(M), a]
            Wq[np.arange(M), a] += 0.1 * err[:, None] * hq / ((hq ** 2).sum(1, keepdims=True) + 1e-6)
        elif SHADOW == 2 and Mix is not None:
            dG = (r - val[np.arange(M), a_sh]) * flip
        else:
            dG = d * flip
        Ch = dG[:, None] * X / np.sqrt(B.T) * g * B.CIN
        B.step("W2", W2, np.einsum("mj,mh->mjh", Cm, h), t)
        if mode != "frozen":
            B.step("W1", W1, np.einsum("mh,md->mhd", Ch, u), t); B.step("b1", b1, Ch, t)
    R = np.array(rew).reshape(-1, 2000, M).mean((1, 2)) / np.array(best).reshape(-1, 2000, M).mean((1, 2))
    if mode in ("reg_hard", "rand"):
        share = np.abs(W2).reshape(M, K, NE, H).sum(2); pref = share.argmax(1)
        grp = Mix.argmax(2)
        same = [(grp[m][:, None] == grp[m][None]) & ~np.eye(H, dtype=bool) for m in range(M)]
        run.same = float(np.mean([ (pref[m][:, None] == pref[m][None])[same[m]].mean() for m in range(M)]))
    else:
        run.same = float("nan")
    run.hz = float(cnt.mean() / B.T * 1000)
    return R


if __name__ == "__main__":
    for mode in sys.argv[1:] or ["frozen", "indep", "rand", "all", "reg_hard", "reg_soft", "true_hard"]:
        R = run(mode)
        print(f"{mode:10s} reward/best per 2000: " + " ".join(f"{x:.2f}" for x in R)
              + f"  | group mates same location {run.same:.2f}  hidden Hz {run.hz:.0f}"
              + (f"  flips {np.mean(run.flips):.3f}" if run.flips else "")
              + (f"  |Q - r_shadow| last {np.mean(run.qerr[-2000:]):.3f}" if run.qerr else ""), flush=True)
