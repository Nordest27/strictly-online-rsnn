"""Shadow-free substitutes for the noise-free twin of twochan.py (SHADOW=3), on the narrow no-columns spiking pick task
(bridge pick, HC=4 -> 24 hidden, spiking motor competition, pool homeostasis, Adam, count gate). Hidden noise: shared
channel only (F=1), groups = learned sensitivity regions (reg_hard). Hidden credit = dG * X_shared * gate.
Per-action value readout Q_k = wq_k . [h, 1] (NLMS, learned from the taken action only) is available to all modes.

dG by mode (CF env / argv):
  frozen    : no hidden learning
  plain     : r - b                                                        (no counterfactual; = twochan reg_hard)
  shadowQ   : 1[a != a_twin] (r - Q[a_twin])            twin = real noise-free copy (the reference to replace)
  oracle    : 1[a != a_twin] (r - r_twin)               (unobservable)
  qg        : r - Q[a_g],   a_g = argmax group drive (no motor noise; the network's greedy choice)
  qg_gate   : 1[a != a_g] (r - Q[a_g])
  vbase     : r - sum_k softmax(BETA * group drive)_k Q_k                  (state value from the Q readouts)
  lin       : 1[a != a_c] (r - Q[a_c]),  a_c = argmax group drive of h - gamma * X  (linearised twin; gamma_i learned
              locally by regressing the neuron's count on its own shared-noise sum)
  lin_det   : 1[a_g != a_c] (r - Q[a_c])                 (only the flips the hidden noise causes in the greedy choice)
  model     : Q[a_g] - Q[a_c]                                                (no reward: the critic as a local model)
  half      : shared noise only in the 2nd half of the hidden window; clean estimate h_c = 2 * first-half counts;
              a_c = argmax drive(h_c); dG = 1[a != a_c] (r - Q[a_c])
Diagnostic (DIAG=1, default): the real twin is always simulated (not used for learning except shadowQ/oracle) and the
correlation of dG with the oracle dG is reported, over the last 4000 trials.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TASK", "pick")
import bridge as B

K, NE, NM, H, D, M = B.K, B.NE, B.NM, B.H, B.D, B.M
NTYPE, NCUE, DC = B.NTYPE, B.NCUE, B.DC
SIGH = float(os.environ.get("SIGH", "0.06")); BETA = float(os.environ.get("BETA", "3")); LAM = float(os.environ.get("LAM", "1"))
DIAG = int(os.environ.get("DIAG", "1")); CENTER = int(os.environ.get("CENTER", "0"))
QCLEAN = int(os.environ.get("QCLEAN", "0"))  # 1: baselines from the noise-free estimate h_c (Q(h_c), a_g := a_c)
HALF = int(os.environ.get("HALF", "0"))      # 1: shared noise only in the 2nd half, h_c = 2 * first-half counts


def run(mode, trials=int(os.environ.get("TR", "40000")), seed=int(os.environ.get("SEED", "0"))):
    rng = np.random.default_rng(seed)
    Vtab = np.random.default_rng(500 + seed).random((NCUE, NTYPE))
    W1 = rng.standard_normal((M, H, D)) / np.sqrt(DC); b1 = np.zeros((M, H))
    W2 = 0.3 * rng.standard_normal((M, NM, H)) / np.sqrt(B.HC)
    S = np.zeros((M, NM, H)); hbar = np.zeros((M, H)); dbar = np.zeros((M, NM))
    Wq = np.zeros((M, K, H + 1)); gx = np.zeros((M, H)); gxx = np.full((M, H), 1.0); gam = np.zeros((M, H))
    base = np.full(M, 0.4); hb = np.zeros((M, K)); rew, best = [], []
    est, orc, flips = [], [], []; rr = rn = 1e-6; dgm = np.zeros(M); fl_e, fl_t, fl_n = [], [], []
    ar = np.arange(M)
    for t in range(trials):
        cue = rng.integers(0, NCUE, M); u = np.zeros((M, K, DC)); val = np.zeros((M, K))
        u[:, :, NTYPE + 1:] = np.eye(NCUE)[cue][:, None, :]
        for i in range(M):
            n = rng.integers(2, K + 1); locs = rng.choice(K, n, replace=False); ty = rng.choice(NTYPE, n, replace=False)
            u[i, locs, ty] = 1.0; u[i, locs, NTYPE] = 1.0; val[i, locs] = Vtab[cue[i], ty]
        u = u.reshape(M, D)
        v = S.reshape(M, K, NE, H).sum(2); Mix = np.eye(K)[v.argmax(1)]          # reg_hard groups
        a1 = np.einsum("mhd,md->mh", W1, u) + b1; I = B.I0 + B.CIN * a1
        G = np.einsum("mis,tms->tmi", Mix, rng.standard_normal((B.T, M, K)))
        on = np.ones(B.T)
        if mode in ("half", "route_half") or HALF:
            on[: B.T // 2] = 0.0
        G = G * on[:, None, None]; X = G.sum(0) / np.sqrt(on.sum())

        def hidden(shared_on):
            Vh = np.zeros((M, H)); ref = np.zeros((M, H)); cnt = np.zeros((M, H)); c1 = np.zeros((M, H))
            for k in range(B.T):
                Vh = B.ALPHA * Vh + I + SIGH * (G[k] if shared_on else 0.0)
                Vh = np.where(ref > 0, B.ALPHA * Vh, Vh)
                z = (Vh >= 1.0) & (ref <= 0); Vh = Vh - z; ref = np.where(z, 2, ref - 1); cnt += z
                if k < B.T // 2:
                    c1 += z
            return cnt, c1
        cnt, c1 = hidden(True)
        h = cnt / (B.T * B.RNORM); g = (cnt > 0).astype(float)
        gdrive = lambda hh: np.einsum("mjh,mh->mj", W2, hh).reshape(M, K, NE).sum(2) + NE * hb
        drive = np.einsum("mjh,mh->mj", W2, h) + np.repeat(hb, NE, 1)
        mseed = int(rng.integers(2 ** 62))
        a, xm = B.motor_spk(B.M0 + B.CM * drive, np.random.default_rng(mseed))
        hq = np.concatenate([h, np.ones((M, 1))], 1); Q = np.einsum("mkh,mh->mk", Wq, hq)
        r = val[ar, a]; d = r - base
        need_twin = DIAG or mode in ("shadowQ", "oracle")
        if need_twin:
            hs = hidden(False)[0] / (B.T * B.RNORM)
            a_tw, _ = B.motor_spk(B.M0 + B.CM * (np.einsum("mjh,mh->mj", W2, hs) + np.repeat(hb, NE, 1)),
                                  np.random.default_rng(mseed))
            ftw = (a != a_tw).astype(float); d_or = ftw * (r - val[ar, a_tw]); flips.append(ftw.mean())
        a_g = gdrive(h).argmax(1)
        if mode in ("half", "route_half") or HALF:
            h_c = 2 * c1 / (B.T * B.RNORM)
        else:
            h_c = h - gam * X / (B.T * B.RNORM)
        a_c = gdrive(h_c).argmax(1)
        if QCLEAN:
            Q = np.einsum("mkh,mh->mk", Wq, np.concatenate([h_c, np.ones((M, 1))], 1)); a_g = a_c
        if mode == "plain":
            dG = d
        elif mode == "shadowQ":
            dG = ftw * (r - Q[ar, a_tw])
        elif mode == "oracle":
            dG = d_or
        elif mode == "qg":
            dG = r - Q[ar, a_g]
        elif mode == "qg_gate":
            dG = (a != a_g) * (r - Q[ar, a_g])
        elif mode == "vbase":
            z = BETA * gdrive(h_c if QCLEAN else h) / NE; p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            dG = r - (p * Q).sum(1)
        elif mode in ("lin", "half", "route_half"):
            dG = (a != a_c) * (r - Q[ar, a_c])
        elif mode == "lin_det":
            dG = (a_g != a_c) * (r - Q[ar, a_c])
        elif mode in ("mlin", "mlin_a"):
            xg_m = xm.reshape(M, K, NE).sum(2) * B.SIGM
            a_cm = (B.T * B.CM * gdrive(h_c) + xg_m).argmax(1); a_nm = (B.T * B.CM * gdrive(h) + xg_m).argmax(1)
            fe = (a_nm != a_cm) if mode == "mlin" else (a != a_cm)
            dG = fe * (r - Q[ar, a_cm])
            if need_twin:
                fl_e.append(fe.astype(float)); fl_t.append(ftw); fl_n.append((a_nm == a).astype(float))
        elif mode == "adv":                                   # advantage of the taken action, clean features
            z = BETA * gdrive(h_c) / NE; p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            dG = Q[ar, a] - (p * Q).sum(1)
        elif mode == "model":
            dG = Q[ar, a_g] - Q[ar, a_c]
        else:
            dG = np.zeros(M)
        if CENTER:
            dgm += 0.01 * (dG - dgm); dG = dG - dgm
        if DIAG and t >= trials - 4000:
            est.append(dG.copy()); orc.append(d_or.copy())
        # learning
        hb += 0.003 * (1.0 / K - np.eye(K)[a])
        base += 0.01 * (r - base); rew.append(r); best.append(val.max(1))
        qa = np.einsum("mh,mh->m", Wq[ar, a], hq)
        Wq[ar, a] += 0.1 * (r - qa)[:, None] * hq / ((hq ** 2).sum(1, keepdims=True) + 1e-6)
        gx += 0.01 * (cnt * X - gx); gxx += 0.01 * (X * X - gxx); gam = gx / gxx       # local gain: count on own noise
        Cm = np.repeat((d[:, None] * xm / B.SIGM).reshape(M, K, NE).mean(2), NE, 1) * B.CM
        hc = h - hbar; dc = drive - dbar; hbar += 0.01 * (h - hbar); dbar += 0.01 * (drive - dbar)
        S += B.LINK * np.einsum("mj,mh->mjh", dc - np.einsum("mjh,mh->mj", S, hc), hc) / ((hc ** 2).sum(1)[:, None, None] + 1e-6)
        Ch = dG[:, None] * X * g * B.CIN
        if mode == "route_link":                            # reference: motor credit routed back through the links
            Ch = np.einsum("mjh,mj->mh", S, Cm) * g * B.CIN
        if mode.startswith("route_") and mode != "route_link":   # route_half: routed + shared-noise (half window) credit
            Cr = np.einsum("mjh,mj->mh", S, Cm) * g
            rr += 0.01 * ((Cr ** 2).mean() - rr); rn += 0.01 * ((Ch ** 2).mean() - rn)
            Ch = (Cr / np.sqrt(rr + 1e-12) + LAM * Ch / np.sqrt(rn + 1e-12)) * B.CIN
        B.step("W2", W2, np.einsum("mj,mh->mjh", Cm, h), t)
        if mode != "frozen":
            B.step("W1", W1, np.einsum("mh,md->mhd", Ch, u), t); B.step("b1", b1, Ch, t)
    R = np.array(rew).reshape(-1, 2000, M).mean((1, 2)) / np.array(best).reshape(-1, 2000, M).mean((1, 2))
    run.corr = float(np.corrcoef(np.ravel(est), np.ravel(orc))[0, 1]) if DIAG and np.std(est) > 0 else float("nan")
    run.flip = float(np.mean(flips[-4000:])) if flips else float("nan")
    run.hz = float(cnt.mean() / B.T * 1000)
    if fl_e:
        e = np.ravel(fl_e[-4000:]); tt = np.ravel(fl_t[-4000:])
        run.fl = f"  flip est: rate {e.mean():.2f}, recall {e[tt > 0].mean():.2f}, precision {tt[e > 0].mean():.2f}, motor-linear action = real {np.mean(fl_n[-4000:]):.2f}"
    else:
        run.fl = ""
    return R


if __name__ == "__main__":
    for mode in sys.argv[1:]:
        R = run(mode)
        print(f"{mode:8s} reward/best per 2000: " + " ".join(f"{x:.2f}" for x in R)
              + f"  | corr(dG, oracle) {run.corr:+.2f}  twin flips {run.flip:.2f}  hidden Hz {run.hz:.0f}" + run.fl, flush=True)
