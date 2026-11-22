"""Bridge from the working rate network to spiking, on the selection task (cue + 4 shuffled items; motor groups =
locations). Each stage can be rate or spiking:
  HID=rate : h = relu(W1 u + b1 + SH xi)                       (H units)
  HID=spk  : LIF over T steps with input current I0 + CIN (W1 u + b1) + SIGH xi; h = spike count / (T * RNORM);
             the gate g = mean pseudo-derivative over the trial / its running mean (or 1[count > 0] with GATE=count)
  MOT=rate : group sums of W2 h + SM xi_m (NE units per group), action = largest group sum
  MOT=spk  : LIF motor pools (NE per group) with shared inhibition, input current M0 + CM * W2 h + SIGM xi per step,
             action = the group with the most spikes in the window; motor noise credit uses the summed xi over the trial
Credit: Cm = group mean of delta * xi_m / noise (per unit, summed over the trial when spiking); hidden Ch = g * W2^T Cm
(route_sym), frozen: none. Plain SGD on W1, b1, W2 (lr). 8 learners in parallel.
Usage: HID=spk MOT=rate python bridge.py route_sym frozen
"""
import os, sys, numpy as np

TASK = os.environ.get("TASK", "select")
if TASK == "pick":      # 6 locations, 2..6 hold items (8 types), cue (4) sets the values; one hidden column per location
    K, NE, NTYPE, NCUE = 6, 20, 8, 4
    HC = int(os.environ.get("HC", "16"))
    DC = NTYPE + 1 + NCUE; H, D = K * HC, K * DC
else:
    K, NE, H, D = 4, 20, 80, 20
NM = K * NE
HID, MOT = os.environ.get("HID", "rate"), os.environ.get("MOT", "rate")
LR = float(os.environ.get("LR", "0.003")); SH = float(os.environ.get("SH", "0.05")); SM = float(os.environ.get("SM", "0.5"))
T, T0, ALPHA = 60, 20, float(np.exp(-0.1))
I0 = (1 - ALPHA) * 1.0                       # rheobase current: drive above it fires (relu-like f-I curve)
CIN = float(os.environ.get("CIN", "0.03")); SIGH = float(os.environ.get("SIGH", "0.02"))
RNORM = float(os.environ.get("RNORM", "0.1")); GATE = os.environ.get("GATE", "psi")
M0 = float(os.environ.get("M0", "0.085")); CM = float(os.environ.get("CM", "0.01")); SIGM = float(os.environ.get("SIGM", "0.12"))
M = 8
SCALE_T = float(os.environ.get("SCALE_T", "1"))
HOMEO = float(os.environ.get("HOMEO", "0"))
LINK = float(os.environ.get("LINK", "0.03"))
OPT = os.environ.get("OPT", "sgd"); ALR = float(os.environ.get("ALR", "1e-3"))


def lif(I, sigma, rng, steps=T):
    """I: (M, n) constant current; returns counts (whole trial), xi sums, mean psi."""
    V = np.zeros(I.shape); ref = np.zeros(I.shape); cnt = np.zeros(I.shape); xs = np.zeros(I.shape); psi = np.zeros(I.shape)
    for _ in range(steps):
        xi = rng.standard_normal(I.shape); xs += xi
        psi += 0.3 * np.maximum(0.0, 1.0 - np.abs(V - 1.0))
        V = ALPHA * V + I + sigma * xi; V = np.where(ref > 0, ALPHA * V, V)
        z = (V >= 1.0) & (ref <= 0); V = V - z; ref = np.where(z, 2, ref - 1); cnt += z
    return cnt, xs, psi / steps


def motor_spk(drive, rng, w_ie=-0.12, w_ei=0.04, w_ee=0.06):
    """drive (M, NM) constant current; shared inhibition, within-group excitation (the teacher). Returns action,
    xi sums over the trial."""
    Mm = drive.shape[0]
    V = np.zeros((Mm, NM)); ref = np.zeros((Mm, NM)); z = np.zeros((Mm, NM)); Vi = np.zeros((Mm, 20)); zi = np.zeros((Mm, 20))
    ri = np.zeros((Mm, 20)); cnt = np.zeros((Mm, NM)); xs = np.zeros((Mm, NM))
    for t in range(T):
        xi = rng.standard_normal((Mm, NM)); xs += xi
        rec = w_ee * np.repeat(z.reshape(Mm, K, NE).sum(2), NE, 1) - w_ee * z + w_ie * zi.sum(1, keepdims=True)
        V = ALPHA * V + drive + rec + SIGM * xi; V = np.where(ref > 0, ALPHA * V, V)
        Vi = ALPHA * Vi + w_ei * z.sum(1, keepdims=True) + 0.12 * rng.standard_normal((Mm, 20)); Vi = np.where(ri > 0, ALPHA * Vi, Vi)
        z = ((V >= 1.0) & (ref <= 0)).astype(float); V -= z; ref = np.where(z > 0, 2, ref - 1)
        zi = ((Vi >= 1.0) & (ri <= 0)).astype(float); Vi -= zi; ri = np.where(zi > 0, 2, ri - 1)
        if t >= T0:
            cnt += z
    votes = cnt.reshape(Mm, K, NE).sum(2)
    return np.argmax(votes + rng.random((Mm, K)) * 1e-3, 1), xs


def step(name, W, G, t, st={}):
    """plain SGD (OPT=sgd) or per-weight Adam (OPT=adam: scale-invariant, gradient ascent on reward)."""
    if OPT == "sgd":
        W += LR * G; return
    if t == 0 or name not in st:
        st[name] = [np.zeros_like(W), np.zeros_like(W)]
    m, v = st[name]
    m *= 0.9; m += 0.1 * G; v *= 0.999; v += 0.001 * G * G
    W += ALR * (m / (1 - 0.9 ** (t + 1))) / (np.sqrt(v / (1 - 0.999 ** (t + 1))) + 1e-8)


def run(rule, trials=int(os.environ.get('TR', '20000')), seed=int(os.environ.get('SEED', '0'))):
    rng = np.random.default_rng(seed)
    if TASK == "pick":
        Vtab = np.random.default_rng(500 + seed).random((NCUE, NTYPE))
        col_h = np.repeat(np.arange(K), HC); col_d = np.repeat(np.arange(K), DC)
        mask1 = (col_h[:, None] == col_d[None, :]).astype(float)                 # a column sees its location only
        mask2 = (np.repeat(np.arange(K), NE)[:, None] == col_h[None, :]).astype(float)   # and drives its own group
        if os.environ.get("COLUMNS", "1") == "0":     # no columns: every hidden neuron sees everything, drives everyone
            mask1 = np.ones((H, D)); mask2 = np.ones((NM, H))
    else:
        mask1 = np.ones((H, D)); mask2 = np.ones((NM, H))
    best_l, hit_l = [], []
    W1 = rng.standard_normal((M, H, D)) / np.sqrt(DC if TASK == "pick" else D) * mask1; b1 = np.zeros((M, H))
    W2 = 0.3 * rng.standard_normal((M, NM, H)) / np.sqrt(HC if TASK == "pick" else H) * mask2
    Bfb = rng.standard_normal((M, NM, H)) / np.sqrt(H) * mask2; Sl = np.zeros((M, NM, H)); hbar = np.zeros((M, H)); dbar = np.zeros((M, NM))
    base = np.full(M, 0.25 if TASK != 'pick' else 0.4); rew = []; gbar = np.full((M, H), 0.1); hb = np.zeros((M, K)); acts = []
    for t in range(trials):
        if TASK == "pick":
            cue = rng.integers(0, NCUE, M); u = np.zeros((M, K, DC)); val = np.zeros((M, K))
            u[:, :, NTYPE + 1:] = np.eye(NCUE)[cue][:, None, :]
            for i in range(M):
                n = rng.integers(2, K + 1); locs = rng.choice(K, n, replace=False); ty = rng.choice(NTYPE, n, replace=False)
                u[i, locs, ty] = 1.0; u[i, locs, NTYPE] = 1.0; val[i, locs] = Vtab[cue[i], ty]
            u = u.reshape(M, D)
        else:
            cue = rng.integers(0, K, M); perm = np.argsort(rng.random((M, K)), 1)
            u = np.concatenate([np.eye(K)[cue], np.eye(K)[perm].reshape(M, K * K)], 1)
            target = np.argmax(perm == cue[:, None], 1)
        a1 = np.einsum("mhd,md->mh", W1, u) + b1
        if HID == "rate":
            a1n = a1 + SH * rng.standard_normal((M, H)); h = np.maximum(a1n, 0); g = (a1n > 0).astype(float)
        else:
            cnt, _, psi = lif(I0 + CIN * a1, SIGH, rng); h = cnt / (T * RNORM)
            if GATE == "count":
                g = (cnt > 0).astype(float)
            else:
                gbar += 0.01 * (psi - gbar); g = psi / (gbar.mean(1, keepdims=True) + 1e-9)
        drive = np.einsum("mjh,mh->mj", W2, h) + np.repeat(hb, NE, 1)
        if MOT == "rate":
            xm = rng.standard_normal((M, NM)); mm = drive + SM * xm
            a = np.argmax(mm.reshape(M, K, NE).sum(2), 1); score = xm / SM
        else:
            a, xs = motor_spk(M0 + CM * drive, rng); score = xs / (SIGM * SCALE_T)
        hb += HOMEO * (1.0 / K - np.eye(K)[a]); acts.append(a)   # each motor group: fair share of wins (local)
        if TASK == "pick":
            r = val[np.arange(M), a]; best_l.append(val.max(1)); hit_l.append((r == val.max(1)).astype(float))
        else:
            r = (a == target).astype(float)
        d = r - base; base += 0.01 * (r - base); rew.append(r)
        Cm = np.repeat((d[:, None] * score).reshape(M, K, NE).mean(2), NE, 1)
        if MOT == "spk":
            Cm = Cm * CM                                   # d drive / d W2 h
        F = {"route_sym": W2, "route_fa": Bfb, "route_link": Sl}.get(rule)
        Ch = np.einsum("mjh,mj->mh", F, Cm) * g if F is not None else np.zeros((M, H))
        if rule == "route_link":               # links regress the motor drive on the hidden activity (reward-free)
            hc = h - hbar; dc = drive - dbar; hbar += 0.01 * (h - hbar); dbar += 0.01 * (drive - dbar)
            Sl += LINK * np.einsum("mj,mh->mjh", dc - np.einsum("mjh,mh->mj", Sl, hc), hc) / ((hc ** 2).sum(1)[:, None, None] + 1e-6)
            Sl *= mask2
        if HID == "spk":
            Ch = Ch * CIN
        step("W2", W2, np.einsum("mj,mh->mjh", Cm, h), t); W2 *= mask2
        if rule != "frozen":
            step("W1", W1, np.einsum("mh,md->mhd", Ch, u), t); step("b1", b1, Ch, t); W1 *= mask1
    run.hz = h.mean() * RNORM * 1000 if HID == "spk" else float("nan")
    run.link = np.mean([np.corrcoef(Sl[i].ravel(), W2[i].ravel())[0, 1] for i in range(M)]) if rule == "route_link" else float("nan")
    A = np.array(acts)[-2000:]
    run.ent = np.mean([-(np.bincount(A[:, i], minlength=K) / len(A) * np.log(np.bincount(A[:, i], minlength=K) / len(A) + 1e-12)).sum() for i in range(M)]) / np.log(K)
    if TASK == "pick":
        run.hit = np.array(hit_l).reshape(-1, 2000, M).mean((1, 2))
        return np.array(rew).reshape(-1, 2000, M).mean((1, 2)) / np.array(best_l).reshape(-1, 2000, M).mean((1, 2))
    return np.array(rew).reshape(-1, 2000, M).mean((1, 2))


if __name__ == "__main__":
    for rule in sys.argv[1:] or ["route_sym", "frozen"]:
        R = run(rule)
        print(f"HID={HID} MOT={MOT} {rule:10s} per 2000: " + " ".join(f"{x:.2f}" for x in R)
              + (f"  hidden Hz {run.hz:.0f}" if HID == "spk" else "") + f"  action entropy {run.ent:.2f}" + (f"  link corr {run.link:+.2f}" if rule == "route_link" else "")
              + (("  | P(best) " + " ".join(f"{x:.2f}" for x in run.hit[::2])) if TASK == "pick" else ""), flush=True)
