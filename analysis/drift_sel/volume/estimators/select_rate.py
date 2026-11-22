"""Rate version of the two-stage credit rules on the selection task (separates the rules from the spiking setup).
u (20) -> hidden h = relu(W1 u + b1 + s xi_h) (H) -> motor m = W2 h + s xi_m (80, 4 vote groups of 20); action = argmax
of the group sums. Motor credit Cm = group mean of delta xi_m / s. Hidden credit: own (delta xi_h / s gated), route_sym
(W2^T Cm gated by relu'), route_fa (B^T Cm), frozen. Reference: backprop REINFORCE with a softmax head (select_bp)."""
import numpy as np, sys, os
EASY = os.environ.get('EASY') == '1'
BETA = float(os.environ.get('BETA', '1.0'))
SGD = os.environ.get('SGD', '1') == '1'
SH = float(os.environ.get('SH', '0.5'))
LINK = float(os.environ.get('LINK', '0.01'))
K, NE = 4, 20; NM = K * NE
def run(rule, H=80, trials=20000, lr1=float(os.environ.get('LR', '1e-3')), lr2=float(os.environ.get('LR', '1e-3')), s=float(os.environ.get('S', '0.5')), seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((H, 20)) / np.sqrt(20); b1 = np.zeros(H); W2 = rng.standard_normal((NM, H)) / np.sqrt(H) * 0.3
    B = rng.standard_normal((NM, H)) / np.sqrt(H)
    Sl = np.zeros((NM, H)); hbar = np.zeros(H); mbar = np.zeros(NM)
    base = 0.25; rew = []; ms1 = ms2 = 1e-12
    for t in range(trials):
        cue = rng.integers(K); perm = rng.permutation(K)
        u = np.concatenate([np.eye(K)[cue], np.eye(K)[perm].ravel()]); target = int(np.argmax(perm == cue)) if not EASY else int(cue)
        xh = rng.standard_normal(H); a1 = W1 @ u + b1 + SH * xh; h = np.maximum(a1, 0)
        xm = rng.standard_normal(NM); m = W2 @ h + s * xm
        if rule.startswith("bp"):              # softmax over group sums (noise-free motor), exact REINFORCE credit
            zg = (W2 @ h).reshape(K, NE).sum(1) * BETA; pg = np.exp(zg - zg.max()); pg /= pg.sum()
            a = int(rng.choice(K, p=pg))
        else:
            a = int(np.argmax(m.reshape(K, NE).sum(1)))
        r = float(a == target); d = r - base; base += 0.01 * (r - base); rew.append(r)
        if rule.startswith("bp"):
            Cm = np.repeat(BETA * d * (np.eye(K)[a] - pg), NE)
        else:
            Cm = np.repeat((d * xm / s).reshape(K, NE).mean(1), NE)
        g = (a1 > 0).astype(float)
        if rule == "own": Ch = d * xh / SH * g
        elif rule in ("route_sym", "bp"): Ch = (W2.T @ Cm) * g
        elif rule == "route_fa": Ch = (B.T @ Cm) * g
        elif rule == "route_link": Ch = (Sl.T @ Cm) * g
        else: Ch = np.zeros(H)
        hc = h - hbar; mc = (W2 @ h + s * xm) - mbar; hbar += 0.01 * (h - hbar); mbar += 0.01 * ((W2 @ h + s * xm) - mbar)
        if rule == "route_link":           # links regress the motor drive on the hidden activity (reward-free, NLMS)
            Sl += LINK * np.outer(mc - Sl @ hc, hc) / (hc @ hc + 1e-6)
        G2 = np.outer(Cm, h); G1 = np.outer(Ch, u)
        ms2 = 0.99 * ms2 + 0.01 * np.mean(G2 ** 2); ms1 = 0.99 * ms1 + 0.01 * np.mean(G1 ** 2)
        bc = 1 - 0.99 ** (t + 1)
        W2 += lr2 * (G2 if SGD else G2 / (np.sqrt(ms2 / bc) + 1e-12))
        if rule not in ("frozen", "bp_frozen"): W1 += lr1 * (G1 if SGD else G1 / (np.sqrt(ms1 / bc) + 1e-12)); b1 += lr1 * (Ch if SGD else Ch / (np.sqrt(ms1 / bc) + 1e-12))
    if rule == "route_link":
        print(f"  link corr with W2 {np.corrcoef(Sl.ravel(), W2.ravel())[0, 1]:+.2f}", flush=True)
    return np.array(rew).reshape(-1, 2000).mean(1)
for rule in sys.argv[1:] or ["frozen", "own", "route_sym", "route_fa"]:
    print(f"{rule:10s} per 2000 trials: " + " ".join(f"{x:.2f}" for x in run(rule)), flush=True)
