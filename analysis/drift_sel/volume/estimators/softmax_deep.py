"""Phase 5: a general two-stage architecture. Input -> hidden H (80 E, no action labels, noise) -> motor (80 E, each
with an action label = its vote, noise) with shared inhibition on the motor stage (fixed) and weak random motor E -> E
(frozen, as the untrained network). Action = the motor group with the most spikes. Plastic: W_in (input -> H) and
W_hm (H -> motor, signed).

Motor credit: the action group's mean perturbation credit, C_m = mean over m's group of delta * sum_t xi (the vote
pathway's feedback, the best signal in phase 4).
Hidden credit variants:
  own        : delta * sum_t xi_h (the hidden neuron's own perturbation)
  route_sym  : psi_h * sum_m W_hm[m, h] C_m      (routed back through the true forward weights)
  route_fa   : psi_h * sum_m B[m, h] C_m          (fixed random feedback)
  route_link : psi_h * sum_m S[m, h] C_m          (links learned reward-free from spikes: S regresses the motor neuron's
                                                   input on the hidden neuron's previous-step spike, per step, NLMS)
  route_*+own: both, each RMS-balanced
psi_h = the hidden neuron's mean pseudo-derivative over the trial. 16 learners, 8-context bandit.
"""
import sys, os, numpy as np

K, NE, NI = 4, 20, 20
NH = int(os.environ.get("NH", "80"))
NM = K * NE
ALPHA, THETA, REFRAC, T, T0 = float(np.exp(-0.1)), 1.0, 2, 60, 20
D, M = 20, 16
NCTX = int(os.environ.get("NCTX", "8"))
TASK = os.environ.get("TASK", "bandit")
label = np.repeat(np.arange(K), NE)


def run(variant, trials=3000, seed=0, eta=0.003, b_h=0.085, b_m=0.07, gin=0.02, sigma=0.12, link_eta=0.01, eta_hm=1e-4, win0=0.05, h_len=0.0, h_rho=0.0, sig_h=None):
    sig_h = sigma if sig_h is None else sig_h
    rng = np.random.default_rng(seed); task = np.random.default_rng(100 + seed)
    pats = (task.random((NCTX, D)) < 0.3).astype(float); correct = task.integers(0, K, NCTX)
    Win = win0 * rng.standard_normal((M, NH, D)) / np.sqrt(D)
    Whm = 0.02 * rng.standard_normal((M, NM, NH)) / np.sqrt(NH) + 0.002     # [motor, hidden]
    B = rng.standard_normal((M, NM, NH)) / np.sqrt(NH)
    S = np.zeros((M, NM, NH))
    Wmm = rng.uniform(0, 2 * 0.06 * (NE - 1) / (NM - 1), (M, NM, NM)) * (~np.eye(NM, dtype=bool))
    w_ei, w_ie = 0.04, -0.12
    ring = np.arange(NH); dd = np.abs(ring[:, None] - ring[None, :]); dd = np.minimum(dd, NH - dd)
    Kg = np.exp(-dd ** 2 / (2 * max(h_len, 1e-6) ** 2))
    Ksm = Kg / np.sqrt((Kg ** 2).sum(1, keepdims=True))        # smoothing with unit output variance
    Kpool = Kg / Kg.sum(1, keepdims=True)                      # spatial averaging
    rbar = np.full(M, 0.25); ms = {k: np.full(M, 1e-12) for k in ("in", "hm", "rt", "ow")}
    rew = []
    for t in range(trials):
        if TASK == "select":   # cue (4) + 4 location slots x item one-hot (16); goal: the location of the cued item
            cue = rng.integers(0, K, M); perm = np.argsort(rng.random((M, K)), 1)      # perm[m, loc] = item
            u = np.concatenate([np.eye(K)[cue], np.eye(K)[perm].reshape(M, K * K)], 1)
            target = np.argmax(perm == cue[:, None], 1)
        else:
            c = rng.integers(0, NCTX, M); u = pats[c]; target = correct[c]
        dh = b_h + gin * np.einsum("mhd,md->mh", Win, u)
        Vh = np.zeros((M, NH)); rh = np.zeros((M, NH)); zh = np.zeros((M, NH))
        Vm = np.zeros((M, NM)); rm = np.zeros((M, NM)); zm = np.zeros((M, NM)); Vi = np.zeros((M, NI)); zi = np.zeros((M, NI))
        ri = np.zeros((M, NI))
        cnt_m = np.zeros((M, NM)); cnt_h = np.zeros((M, NH)); xs_m = np.zeros((M, NM)); xs_h = np.zeros((M, NH))
        psi_h = np.zeros((M, NH)); reg_xy = np.zeros((M, NM, NH)); reg_xx = np.zeros((M, NH))
        inbar = np.zeros((M, NM)); zhbar = np.zeros((M, NH)); first = True
        for s_ in range(T):
            xh = rng.standard_normal((M, NH))
            if h_rho > 0:                    # spatially smooth noise field on the hidden ring (unit variance per neuron)
                xh = np.sqrt(h_rho) * (rng.standard_normal((M, NH)) @ Ksm.T) + np.sqrt(1 - h_rho) * xh
            xm = rng.standard_normal((M, NM)); xi_ = rng.standard_normal((M, NI))
            xs_h += xh; xs_m += xm
            psi_h += 0.3 * np.maximum(0.0, 1.0 - np.abs(Vh - THETA) / THETA) / THETA
            in_m = np.einsum("mkh,mh->mk", Whm, zh)                          # motor input from H (prev step spikes)
            if variant.startswith("route_link"):                              # per-step regression of in_m on zh
                if not first:
                    zc = zh - zhbar
                    reg_xy += (in_m - inbar)[:, :, None] * zc[:, None, :]; reg_xx += zc * zc
                inbar += 0.1 * (in_m - inbar); zhbar += 0.1 * (zh - zhbar); first = False
            Vh = ALPHA * Vh + dh + sig_h * xh
            Vh = np.where(rh > 0, ALPHA * Vh, Vh)
            Vm = (ALPHA * Vm + b_m + in_m + np.einsum("mij,mj->mi", Wmm, zm) + w_ie * zi.sum(1, keepdims=True)
                  + sigma * xm)
            Vm = np.where(rm > 0, ALPHA * Vm, Vm)
            Vi = ALPHA * Vi + w_ei * zm.sum(1, keepdims=True) + sigma * xi_
            Vi = np.where(ri > 0, ALPHA * Vi, Vi)
            zh = ((Vh >= THETA) & (rh <= 0)).astype(float); Vh -= THETA * zh; rh = np.where(zh > 0, REFRAC, rh - 1)
            zm = ((Vm >= THETA) & (rm <= 0)).astype(float); Vm -= THETA * zm; rm = np.where(zm > 0, REFRAC, rm - 1)
            zi = ((Vi >= THETA) & (ri <= 0)).astype(float); Vi -= THETA * zi; ri = np.where(zi > 0, REFRAC, ri - 1)
            cnt_h += zh
            if s_ >= T0:
                cnt_m += zm
        votes = cnt_m.reshape(M, K, NE).sum(2)
        a = np.argmax(votes + rng.random((M, K)) * 1e-3, 1)
        r = (a == target).astype(float); delta = r - rbar; rbar += 0.02 * (r - rbar); rew.append(r)
        psi_h /= T
        Cm = np.repeat((delta[:, None] * xs_m).reshape(M, K, NE).mean(2), NE, axis=1)   # vote-pathway credit
        own_h = delta[:, None] * xs_h
        if variant.startswith("route_link"):
            S += link_eta * (reg_xy - S * reg_xx[:, None, :]) / (reg_xx[:, None, :] + reg_xx.mean(1)[:, None, None] + 1e-9)
        F = {"route_sym": Whm, "route_fa": B, "route_link": S}
        key = variant.replace("+own", "")
        def rms(x):
            return x / (np.sqrt((x ** 2).mean(1, keepdims=True)) + 1e-12)
        if key == "pool_space":                # credit averaged over spatial neighbours (same kernel)
            Ch = own_h @ Kpool.T
        elif key in F:
            Ch = psi_h * np.einsum("mkh,mk->mh", F[key], Cm)
            if variant.endswith("+own"):
                Ch = rms(Ch) + rms(own_h)
        else:
            Ch = own_h
        upd = [("hm", Whm, Cm[:, :, None] * cnt_h[:, None, :])]
        if variant != "frozen_h":
            upd.append(("in", Win, Ch[:, :, None] * u[:, None, :]))
        for name, W_, G in upd:
            ms[name] = 0.99 * ms[name] + 0.01 * (G ** 2).mean((1, 2))
            W_ += (eta if name == "in" else eta_hm) * G / (np.sqrt(ms[name] / (1 - 0.99 ** (t + 1)))[:, None, None] + 1e-12)
        if t == trials - 1 and variant.startswith("route_link"):
            corr = np.mean([np.corrcoef(S[i].ravel(), Whm[i].ravel())[0, 1] for i in range(M)])
            run.link_corr = corr
    R = np.array(rew).reshape(-1, 250, M).mean(1)
    run.rates = (cnt_h.mean() / T * 1000, cnt_m.mean() / (T - T0) * 1000)
    pref = Whm.reshape(M, K, NE, NH).sum(2)                    # (M, K, NH): each hidden neuron's drive to each action
    pref = pref - pref.mean(1, keepdims=True); pref = pref - pref.mean(2, keepdims=True)   # specific preferences
    pn = pref / (np.linalg.norm(pref, axis=1, keepdims=True) + 1e-12)
    near = np.mean([np.sum(pn[:, :, (ring + d) % NH] * pn, 1).mean() for d in (1, 2)])
    far = np.mean([np.sum(pn[:, :, (ring + d) % NH] * pn, 1).mean() for d in range(NH // 4, NH // 2)])
    run.spec = (near, far)
    return R


if __name__ == "__main__":
    for v in sys.argv[1:] or ["own", "route_sym", "route_fa", "route_link", "route_sym+own", "route_link+own"]:
        R = run(v, trials=int(os.environ.get("TR", "3000")), seed=int(os.environ.get("SEED", "0")), eta=float(os.environ.get("ETA", "0.003")), eta_hm=float(os.environ.get("EHM", "1e-4")),
                win0=float(os.environ.get("WIN0", "0.05")), h_len=float(os.environ.get("HL", "0")),
                h_rho=float(os.environ.get("HRHO", "0")),
                sig_h=float(os.environ["SIGH"]) if "SIGH" in os.environ else None)
        extra = f"  link corr with W_hm {run.link_corr:+.2f}" if v.startswith("route_link") else ""
        print(f"{v:15s} reward/250: " + " ".join(f"{x:.2f}" for x in R.mean(1))
              + f"   Hz H {run.rates[0]:.0f} motor {run.rates[1]:.0f}{extra}"
              + f"  | neighbour pref cos {run.spec[0]:+.2f} vs far {run.spec[1]:+.2f}", flush=True)
