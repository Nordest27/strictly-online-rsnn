"""Toy 2: deeper network with competition layers. Does feedback that is LEARNED from the network's own activity
(state-dependent) beat fixed random feedback (feedback alignment, FA)?

Network (rates), layer l: a_l = W_l x_{l-1} + s n_l (node noise), x_l = f_l(a_l). Layer types: relu; gsm = soft
winner-take-all within groups (softmax(kappa a) per group: a competition computed in activity, winner changes with the
input); out = linear, action ~ softmax(beta u). Task: contextual bandit (noisy cluster samples, random correct action).

Credit C_l (at pre-activation a_l). Output layer: C_L = delta n_L / s (perturbation). Backward: C_l = d_l * (F^T C_{l+1})
with F the feedback (transport) matrix and d_l the unit's local derivative (state-dependent).
Feedback variants (all local: a unit uses its own activity, links use their two ends):
  fa_nod : F = fixed random B, d = relu mask only (competition derivative ignored)
  fa_d   : F = B, d = exact local derivative of each unit (diag: relu step; gsm kappa h(1-h))
  fa_ld  : F = B, d = LEARNED per-unit derivative: d_j = c_j . phi(x_j), phi = [1, x, x^2, 1[x>0]], learned by LMS on the
           unit's own noise response (x_j - xbar_j) n_j / s  (reward-free; Stein)
  tr_ld  : F = LEARNED transport T (link learns E[(a_i - abar_i) n_j d_j / s] / E[d_j^2] -> W_ij) + learned d
  tr_d   : learned T + exact diagonal derivative;  tr_x / fa_x : T or B + exact local Jacobian incl. the lateral
           within-group term -kappa h_i sum_j h_j y_j (carried by the competition's own lateral links)
  avgJ   : one state-averaged Jacobian per link (previous toy), no derivative gating
Prefix "m_" = own perturbation credit delta n_l / s added to the routed credit (the combination that worked in toy 1).
perturbation: own credit only. backprop: exact gradient (full group-softmax Jacobian).
Updates are per-layer RMS normalised (only the credit direction matters).

Usage: python jacobian_toy2.py [--trials 40000 --seeds 5 --rules ...]
"""
import argparse, time
import numpy as np

ARCH = [("relu", 64), ("gsm", 32), ("relu", 64), ("gsm", 16), ("out", None)]


def gsm(a, m, kappa):
    z = (kappa * a).reshape(-1, m); z = z - z.max(1, keepdims=True); e = np.exp(z)
    return (e / e.sum(1, keepdims=True)).ravel()


def feats(x, kind):
    return np.stack([np.ones_like(x), x, x * x, (x > 0).astype(float)], 1)


DCORR = {l: [] for l in range(8)}
FLIPS = []


def run(rule, seed, trials=40000, D=20, N_CL=16, K=4, arch=ARCH, m=4, kappa=3.0, s=0.3, beta=3.0, eta=0.001,
        t_rate=0.001, d_rate=0.002, block=2000, out_exact=False, d_clean=False, headless=False, motor_credit="pert", rel_noise=False, homeo=0.0, du_plain=False, repeat=1, mrate=0.01, xnoise=0.3, adapt=0.0, prior=0.0):
    rng = np.random.default_rng(seed); task = np.random.default_rng(1000 + seed)
    centres = task.standard_normal((N_CL, D)); labels = task.integers(0, K, N_CL)
    if headless:                 # no policy head: the last layer is a motor competition (group of K), action = winner
        arch = list(arch[:-1]) + [("gsm", K)]
        assert K == m
    sizes = [D] + [n if n is not None else K for _, n in arch]; kinds = [k for k, _ in arch]; L = len(arch)
    W = [rng.standard_normal((sizes[l + 1], sizes[l])) / np.sqrt(sizes[l]) for l in range(L)]
    B = [rng.standard_normal(w.shape) / np.sqrt(w.shape[1]) for w in W]
    T = [np.zeros_like(w) for w in W]; J = [np.zeros((sizes[l + 1], sizes[l])) for l in range(L)]
    C = [np.zeros((sizes[l + 1], 4)) for l in range(L)]            # learned derivative coefficients per unit
    abar = [np.zeros(sizes[l + 1]) for l in range(L)]; xbar = [np.zeros(sizes[l + 1]) for l in range(L)]
    B_direct = [rng.standard_normal((K, sizes[l + 1])) / np.sqrt(K) for l in range(L)]   # random direct feedback
    S_direct = [np.zeros((K, sizes[l + 1])) for l in range(L)]                            # learned sensitivities
    dsq = [np.ones(sizes[l + 1]) for l in range(L)]                # running E[d_j^2] (normalised LMS for T)
    cosacc = [[[] for _ in range(L - 1)] for _ in range(2)]        # [first block, last quarter][layer]
    rs = [[0.0] * L, [0.0] * L]                                    # running mean squares (routed, own) per layer
    mbar = np.zeros(K)
    vow = [np.zeros(sizes[l + 1]) for l in range(L)]; tcount = [0]
    cvr = [np.zeros(sizes[l + 1]) for l in range(L)]; vrr = [np.zeros(sizes[l + 1]) for l in range(L)]
    hb = [np.zeros(sizes[l + 1]) for l in range(L)]                # homeostatic biases (competition units)
    Sg = [s * np.ones(sizes[l + 1]) for l in range(L)]             # per-unit noise sd
    pvar = [np.ones(sizes[l + 1]) for l in range(L)]; pmean = [np.zeros(sizes[l + 1]) for l in range(L)]
    ms = [0.0] * L; base = 0.0; rewards = []
    own = rule.startswith("m_") or rule == "perturbation"; fb = rule[2:] if rule.startswith("m_") else rule

    def mix(l, routed, ownc):
        """Combine routed credit with the unbiased own perturbation credit. adapt: per neuron, weight the routed part by
        its regression coefficient on the own credit, w = cov(own, routed) / var(routed) (trusted only while it agrees);
        otherwise both parts at comparable scale (running RMS per layer)."""
        if adapt > 0:
            cvr[l] += adapt * (ownc * routed - cvr[l]); vrr[l] += adapt * (routed * routed - vrr[l])
            vow[l] += adapt * (ownc * ownc - vow[l])
            w_data = np.maximum(cvr[l], 0.0) / (vrr[l] + 1e-12)
            if prior > 0:                      # start trusting at the equal-scale weight; own evidence takes over
                w0 = np.sqrt(vow[l] / (vrr[l] + 1e-12)); nt = tcount[0]
                w_data = (nt * w_data + prior * w0) / (nt + prior)
            return ownc + w_data * routed
        rs[0][l] = 0.999 * rs[0][l] + 0.001 * float(np.mean(routed ** 2))
        rs[1][l] = 0.999 * rs[1][l] + 0.001 * float(np.mean(ownc ** 2))
        return routed / np.sqrt(rs[0][l] + 1e-12) + ownc / np.sqrt(rs[1][l] + 1e-12)

    def f(l, a):
        return np.maximum(a, 0) if kinds[l] == "relu" else gsm(a, m, kappa) if kinds[l] == "gsm" else a

    def dexact(l, a, x):
        return (a > 0).astype(float) if kinds[l] == "relu" else kappa * x * (1 - x) if kinds[l] == "gsm" else np.ones_like(a)

    for t in range(trials):
        tcount[0] = t
        if t % repeat == 0:
            c = task.integers(N_CL)
        x0 = centres[c] + xnoise * task.standard_normal(D)
        xs, As, Ns = [x0], [], []
        for l in range(L):
            n = rng.standard_normal(sizes[l + 1]); Sg[l] = s * (np.sqrt(pvar[l]) if rel_noise else 1.0); a = W[l] @ xs[-1] + hb[l] + Sg[l] * n
            As.append(a); Ns.append(n); xs.append(f(l, a))
            if rel_noise:                          # unit's own signal scale (noise-free drive), slow running stats
                a0 = a - Sg[l] * n; pmean[l] += 0.001 * (a0 - pmean[l])
                pvar[l] += 0.001 * ((a0 - pmean[l]) ** 2 - pvar[l]); pvar[l] = np.maximum(pvar[l], 1e-4)
        if headless:
            if not np.all(np.isfinite(xs[-1])):
                raise FloatingPointError(f"{rule} seed {seed}: diverged at {t}")
            xc = [x0]
            for l in range(L):                     # noise-free shadow pass
                xc.append(f(l, W[l] @ xc[-1] + hb[l]))
            act = int(np.argmax(xs[-1])); flip = act != int(np.argmax(xc[-1])); pi = None
            FLIPS.append(float(flip))
        else:
            u = xs[-1]; z = beta * u; z -= z.max(); pi = np.exp(z) / np.exp(z).sum()
            if not np.all(np.isfinite(pi)):
                raise FloatingPointError(f"{rule} seed {seed}: diverged at {t}")
            act = rng.choice(K, p=pi)
        r = float(act == labels[c]); delta = r - base; base += 0.01 * (r - base)
        rewards.append(r)
        # local derivatives
        dl = [None] * L
        for l in range(L - 1):
            if fb in ("fa_ld", "tr_ld"):
                dl[l] = np.clip(np.einsum("ik,ik->i", C[l], feats(xs[l + 1], kinds[l])), 0.0, 2.0)   # bounded gain
                if t >= trials * 3 // 4:
                    DCORR[l].append((dl[l], dexact(l, As[l], xs[l + 1])))
            elif fb.endswith("_ys"):    # measured gain: own output change from own (layer) noise vs a noise-free shadow
                dl[l] = (xs[l + 1] - f(l, As[l] - Sg[l] * Ns[l])) * Ns[l] / Sg[l]
            elif fb.endswith("_yp"):    # same, centred on the unit's running mean (no shadow)
                dl[l] = (xs[l + 1] - xbar[l]) * Ns[l] / Sg[l]
            elif fb == "fa_nod":
                dl[l] = (As[l] > 0).astype(float) if kinds[l] == "relu" else np.ones(sizes[l + 1])
            elif fb == "avgJ":
                dl[l] = (As[l] > 0).astype(float) if kinds[l] == "relu" else np.ones(sizes[l + 1])
            else:
                dl[l] = dexact(l, As[l], xs[l + 1])
        G = [None] * L

        def true_back(l, gl):    # exact backprop of a credit at a_l to a_{l-1} (full group-softmax Jacobian)
            gx = W[l].T @ gl; a, x = As[l - 1], xs[l]
            if kinds[l - 1] == "relu":
                return gx * (a > 0)
            xg, gg = x.reshape(-1, m), gx.reshape(-1, m)
            return (kappa * xg * (gg - (xg * gg).sum(1, keepdims=True))).ravel()
        if rule == "backprop":
            g = beta * (np.eye(K)[act] - pi) * delta
            for l in range(L - 1, -1, -1):
                G[l] = np.outer(g, xs[l])
                if l > 0:
                    gx = W[l].T @ g; a, x = As[l - 1], xs[l]
                    if kinds[l - 1] == "relu":
                        g = gx * (a > 0)
                    else:
                        xg, gg = x.reshape(-1, m), gx.reshape(-1, m)
                        g = (kappa * xg * (gg - (xg * gg).sum(1, keepdims=True))).ravel()
        elif fb.startswith(("dfa", "ldfa", "outpert")):
            # task enters only at the output (e); hidden units use task-free sensitivities to the output logits
            e = (delta * Ns[-1] / Sg[-1] * (float(flip) if motor_credit == "flip" else 1.0)) if headless else (beta * (np.eye(K)[act] - pi) * delta if out_exact else delta * Ns[-1] / Sg[-1])
            G[L - 1] = np.outer(e, xs[L - 1])
            if not headless:
                xc = [x0]
                for l in range(L):                     # noise-free (shadow) pass
                    xc.append(f(l, W[l] @ xc[-1] + hb[l]))
            drive = W[L - 1] @ xs[L - 1]
            if du_plain:                               # no global shadow: centre on the motor drive's running mean
                du = drive - mbar; mbar += mrate * (drive - mbar)
            else:
                du = drive - W[L - 1] @ xc[L - 1]      # output pre-activation change from all hidden noise
            gb = [None] * L
            if not headless:
                g = beta * (np.eye(K)[act] - pi) * delta
                for l in range(L - 1, 0, -1):          # exact credits, diagnostics only
                    g = true_back(l, g); gb[l - 1] = g
            for l in range(L - 1):
                if fb == "outpert":
                    Ch = (e @ du) * Ns[l] / Sg[l]
                else:
                    Dm = B_direct[l] if fb.startswith("dfa") else S_direct[l]
                    Ch = dl[l] * (Dm.T @ e)
                    if fb.startswith("ldfa"):          # task-free: E[du n_j g_j / s] / E[g_j^2] (normalised LMS)
                        gg = dl[l]; dsq[l] += 0.01 * (gg * gg - dsq[l])
                        S_direct[l] += t_rate * (np.outer(du, Ns[l] * gg / Sg[l]) - S_direct[l] * (gg * gg)[None, :]) / (dsq[l] + 1e-3)
                if own:                                # m_: + own perturbation credit, both parts at comparable scale
                    ownc = delta * Ns[l] / Sg[l]
                    Ch = mix(l, Ch, ownc)
                G[l] = np.outer(Ch, xs[l])
                if gb[l] is None:
                    continue
                nr, nt = np.linalg.norm(Ch), np.linalg.norm(gb[l])
                ph_ = 0 if t < block else 1 if t >= trials * 3 // 4 else None
                if nr > 0 and nt > 0 and ph_ is not None:
                    cosacc[ph_][l].append(float(Ch @ gb[l] / (nr * nt)))
        else:
            Cl = (delta * Ns[-1] / Sg[-1] * (float(flip) if motor_credit == "flip" else 1.0)) if headless else (beta * (np.eye(K)[act] - pi) * delta if out_exact else delta * Ns[-1] / Sg[-1])   # output's own credit
            for l in range(L - 1, -1, -1):
                G[l] = np.outer(Cl, xs[l])
                if l > 0:
                    if fb == "perturbation":
                        routed = 0.0
                    else:
                        F = B[l] if fb.startswith("fa") else T[l] if fb.startswith("tr") else J[l]
                        y_ = F.T @ Cl
                        if fb in ("fa_x", "tr_x") and kinds[l - 1] == "gsm":   # + lateral (within-group) term
                            xg, gg = xs[l].reshape(-1, m), y_.reshape(-1, m)
                            routed = (kappa * xg * (gg - (xg * gg).sum(1, keepdims=True))).ravel()
                        else:
                            routed = dl[l - 1] * y_
                        tb = true_back(l, Cl)     # feedback quality: routed vs exact backprop of the same credit
                        nr, nt = np.linalg.norm(routed), np.linalg.norm(tb)
                        if nr > 0 and nt > 0:
                            ph_ = 0 if t < block else 1 if t >= trials * 3 // 4 else None
                            if ph_ is not None:
                                cosacc[ph_][l - 1].append(float(routed @ tb / (nr * nt)))
                    ownc = delta * Ns[l - 1] / Sg[l - 1] * (float(flip) if headless and motor_credit == "flip" else 1.0) if own else 0.0
                    if own and fb != "perturbation":   # mix: both parts at comparable scale (running RMS per layer)
                        routed = mix(l - 1, routed, ownc); ownc = 0.0
                    Cl = routed + ownc
        for l in range(L):
            ms[l] = 0.999 * ms[l] + 0.001 * float(np.mean(G[l] ** 2))
            W[l] += eta * G[l] / (np.sqrt(ms[l] / (1 - 0.999 ** (t + 1))) + 1e-12)
        # reward-free learning of the feedback from noise-activity covariance (plain running means, no shadow)
        if fb in ("fa_ld", "tr_ld", "tr_d", "tr_x", "tr_ys", "tr_yp", "avgJ"):
            for l in range(L):
                da = As[l] - abar[l]
                if l > 0 and fb.startswith("tr"):
                    d = dl[l - 1]; dsq[l - 1] += 0.01 * (d * d - dsq[l - 1])
                    T[l] += t_rate * (np.outer(da, Ns[l - 1] * d / Sg[l - 1]) - T[l] * (d * d)[None, :]) / (dsq[l - 1] + 1e-3)
                if l > 0 and fb == "avgJ":
                    J[l] += t_rate * (np.outer(da, Ns[l - 1] / Sg[l - 1]) - J[l])
                if l < L - 1 and fb in ("fa_ld", "tr_ld"):
                    ph = feats(xs[l + 1], kinds[l])
                    ref = f(l, As[l] - Sg[l] * Ns[l]) if d_clean else xbar[l]   # shadow: same unit without its own noise
                    y = (xs[l + 1] - ref) * Ns[l] / Sg[l]
                    pred = np.einsum("ik,ik->i", C[l], ph)
                    C[l] += d_rate * (y - pred)[:, None] * ph / (1.0 + (ph * ph).sum(1, keepdims=True))   # NLMS
                abar[l] += 0.01 * da
        for l in range(L):
            xbar[l] += mrate * (xs[l + 1] - xbar[l])
            if homeo > 0 and kinds[l] == "gsm":    # each competing unit: fair share of its competition (local)
                hb[l] += homeo * (1.0 / m - xs[l + 1])
    rr = np.array(rewards)
    cosm = np.array([[np.mean(v) if v else np.nan for v in ph] for ph in cosacc])
    return rr[: len(rr) // block * block].reshape(-1, block).mean(1), cosm


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=40000)
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--eta", type=float, default=0.001)
    ap.add_argument("--kappa", type=float, default=3.0)
    ap.add_argument("--out-exact", action="store_true", help="output layer uses its exact local credit delta(onehot-pi)")
    ap.add_argument("--d-clean", action="store_true", help="learned derivative centred on a noise-free (shadow) output")
    ap.add_argument("--tag", default="")
    ap.add_argument("--headless", action="store_true", help="no policy: motor competition group, action = winner")
    ap.add_argument("--prior", type=float, default=0.0, help="trials of prior trust (equal-scale weight) in adapt mode")
    ap.add_argument("--adapt", type=float, default=0.0, help="rate of the per-neuron reliability weighting (0 = off)")
    ap.add_argument("--t-rate", type=float, default=0.001, help="learning rate of transports / sensitivities")
    ap.add_argument("--xnoise", type=float, default=0.3, help="sensory noise sd on the cluster sample")
    ap.add_argument("--repeat", type=int, default=1, help="each state persists this many trials")
    ap.add_argument("--mrate", type=float, default=0.01, help="running-mean rate for shadow-free centring")
    ap.add_argument("--du-plain", action="store_true", help="sensitivity learning without the global shadow pass")
    ap.add_argument("--homeo", type=float, default=0.0, help="homeostatic bias rate for competition units")
    ap.add_argument("--rel-noise", action="store_true", help="per-unit noise sd = s * running sd of own drive")
    ap.add_argument("--motor-credit", default="pert", choices=["pert", "flip"])
    ap.add_argument("--save", default=None, help="npz to store per-seed curves")
    ap.add_argument("--rules", nargs="+", default=["perturbation", "fa_nod", "fa_d", "fa_ld", "tr_ld", "avgJ",
                                                   "m_fa_nod", "m_fa_d", "m_fa_ld", "m_tr_ld", "m_avgJ", "backprop"])
    a = ap.parse_args()
    print(f"arch {ARCH} kappa {a.kappa} eta {a.eta}; mean reward per 2000-trial block (chance 0.25), mean of seeds")
    for rl in a.rules:
        t0 = time.time()
        try:
            res = [run(rl, sd, a.trials, eta=a.eta, kappa=a.kappa, out_exact=a.out_exact, d_clean=a.d_clean, headless=a.headless, motor_credit=a.motor_credit, rel_noise=a.rel_noise, homeo=a.homeo, du_plain=a.du_plain, repeat=a.repeat, mrate=a.mrate, xnoise=a.xnoise, t_rate=a.t_rate, adapt=a.adapt, prior=a.prior) for sd in range(a.seeds)]
            cv = np.array([r_[0] for r_ in res]); cs = np.nanmean([r_[1] for r_ in res], 0)
        except FloatingPointError as e:
            print("DIVERGED:", e, flush=True); continue
        if a.save:
            np.save(f"{a.save}_{rl}{a.tag}.npy", cv)
        mc = cv.mean(0); n = len(mc); marks = [0, n // 8, n // 4, n // 2, n - 1]
        print(f"{rl:13s}" + "".join(f"  b{mk + 1:<2d} {mc[mk]:.3f}" for mk in marks)
              + f"   mean {mc.mean():.3f} (seed sd {cv.mean(1).std():.3f})  {time.time() - t0:.0f}s", flush=True)
        if np.isfinite(cs).any():
            print(f"{'':13s}  feedback cos vs exact (layers 1..{cs.shape[1]}): first block "
                  + " ".join(f"{v:+.2f}" for v in cs[0]) + " | last quarter " + " ".join(f"{v:+.2f}" for v in cs[1]),
                  flush=True)


if __name__ == "__main__":
    main()
