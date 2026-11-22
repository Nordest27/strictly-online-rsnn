"""Pick task (flexible softmax): S locations, 2..S of them hold an item (8 types); a cue (4) says what is wanted; the
reward is V[cue, type of the picked item] (random table in [0, 1]); an empty location gives 0. Each location has its own
column: hidden h_s = relu(W1_s [item one-hot, present, cue] + b_s + SH xi) (H per column) -> motor group m_s = W2_s h_s
+ s xi_m (NE units); the action = the location whose group sum is largest (the groups compete; only what is present
should win). Rate network; plain SGD.

Motor credit: group mean of delta xi_m / s (the vote pathway). Hidden credit:
  frozen     : none
  route_sym  : relu' * W2_s^T Cm_s     (back through the true weights of its own column)
  route_link : relu' * S_s^T Cm_s      (links learned reward-free: regress the motor drive on the hidden activity)
  route_fa   : relu' * B_s^T Cm_s
  bp         : exact REINFORCE through a softmax over the group sums (reference)
SHARE=1: the columns share their weights (replicated circuits): the updates are averaged over locations.
Reported per block of 2000 trials: mean reward / mean best available, and P(best picked).
"""
import os, sys, numpy as np

S_LOC, NTYPE, NCUE, H, NE = 6, 8, 4, int(os.environ.get("H", "16")), 10
DIN = NTYPE + 1 + NCUE
SH, SM = float(os.environ.get("SH", "0.05")), float(os.environ.get("SM", "0.5"))
SHARE = os.environ.get("SHARE", "0") == "1"


def run(rule, trials=30000, lr=float(os.environ.get("LR", "0.003")), link=0.03, seed=0, beta=1.0):
    rng = np.random.default_rng(seed); task = np.random.default_rng(500 + seed)
    V = task.random((NCUE, NTYPE))
    W1 = rng.standard_normal((S_LOC, H, DIN)) / np.sqrt(DIN); b1 = np.zeros((S_LOC, H))
    W2 = 0.3 * rng.standard_normal((S_LOC, NE, H)) / np.sqrt(H)
    if SHARE:
        W1[:] = W1[0]; W2[:] = W2[0]
    B = rng.standard_normal((S_LOC, NE, H)) / np.sqrt(H)
    Sl = np.zeros((S_LOC, NE, H)); hbar = np.zeros((S_LOC, H)); mbar = np.zeros((S_LOC, NE))
    base = 0.5; rr, best, hit = [], [], []
    for t in range(trials):
        cue = rng.integers(NCUE); n = rng.integers(2, S_LOC + 1)
        locs = rng.choice(S_LOC, n, replace=False); types = rng.choice(NTYPE, n, replace=False)
        x = np.zeros((S_LOC, DIN)); x[:, NTYPE + 1 + cue] = 1.0
        x[locs, types] = 1.0; x[locs, NTYPE] = 1.0
        val = np.zeros(S_LOC); val[locs] = V[cue, types]
        a1 = np.einsum("shd,sd->sh", W1, x) + b1 + SH * rng.standard_normal((S_LOC, H)); h = np.maximum(a1, 0)
        xm = rng.standard_normal((S_LOC, NE)); drive = np.einsum("sjh,sh->sj", W2, h); m = drive + SM * xm
        if rule == "bp":
            z = beta * drive.sum(1); p = np.exp(z - z.max()); p /= p.sum(); a = int(rng.choice(S_LOC, p=p))
        else:
            a = int(np.argmax(m.sum(1)))
        r = val[a]; d = r - base; base += 0.01 * (r - base)
        rr.append(r); best.append(val.max()); hit.append(float(val[a] == val.max()))
        if rule == "bp":
            Cm = np.repeat((beta * d * (np.eye(S_LOC)[a] - p))[:, None], NE, 1)
        else:
            Cm = np.repeat((d * xm / SM).mean(1, keepdims=True), NE, 1)
        g = (a1 > 0).astype(float)
        F = {"route_sym": W2, "bp": W2, "route_fa": B, "route_link": Sl}.get(rule)
        Ch = np.einsum("sjh,sj->sh", F, Cm) * g if F is not None else np.zeros((S_LOC, H))
        if rule == "route_link":
            hc = h - hbar; mc = m - mbar; hbar += 0.01 * (h - hbar); mbar += 0.01 * (m - mbar)
            Sl += link * np.einsum("sj,sh->sjh", mc - np.einsum("sjh,sh->sj", Sl, hc), hc) / ((hc ** 2).sum(1)[:, None, None] + 1e-6)
        G2 = np.einsum("sj,sh->sjh", Cm, h); G1 = np.einsum("sh,sd->shd", Ch, x)
        if SHARE:
            G2[:] = G2.mean(0); G1[:] = G1.mean(0); Ch = np.repeat(Ch.mean(0, keepdims=True), S_LOC, 0)
        W2 += lr * G2
        if rule != "frozen":
            W1 += lr * G1; b1 += lr * Ch
    rr, best, hit = (np.array(v).reshape(-1, 2000).mean(1) for v in (rr, best, hit))
    if rule == "route_link":
        run.link = np.corrcoef(Sl.ravel(), W2.ravel())[0, 1]
    return rr / best, hit


if __name__ == "__main__":
    for rule in sys.argv[1:] or ["frozen", "route_sym", "route_link", "route_fa", "bp"]:
        q, hit = run(rule)
        extra = f"  link corr {run.link:+.2f}" if rule == "route_link" else ""
        print(f"{rule:10s} reward/best per 2000: " + " ".join(f"{v:.2f}" for v in q[::2])
              + "  | P(best): " + " ".join(f"{v:.2f}" for v in hit[::2]) + extra, flush=True)
