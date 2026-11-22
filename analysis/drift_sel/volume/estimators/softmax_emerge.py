"""Phase 4: can action competition emerge, with noise credit relayed through the network's own excitation?

Same neurons as the teacher (softmax_net): 80 E neurons, each with a fixed action label (which action its spikes vote
for: 20 per action), 20 shared I neurons (all E -> I, all I -> E, fixed). No pools in the wiring: E -> E weights start
weak, random, non-negative, and are plastic (clipped to [0, W_MAX]). Input weights W_in (E x D) plastic.
Contextual bandit: 8 contexts, one correct action each; 16 independent learners in parallel.

Credit: own_i = delta * sum_t xi_i(t) (node perturbation). Relay through the network's own E -> E weights:
  relayed_i = own_i + gamma * sum_j W_ij own_j / sum_j W_ij   (each neuron also hears its excitatory partners' credit)
Updates (per trial, per-learner RMS-normalised per weight set):
  W_in[i]   += eta * relayed_i * u
  W_ee[i,j] += eta_ee * (credit term: relayed_i * zc_j) + hebb * (zc_i zc_j - mean) [variants]
zc = per-neuron spike count in the window, centred by its running mean.

Variants:
  own          : no relay; W_ee frozen random
  relay_frozen : relay through the frozen random W_ee
  relay_credit : relay; W_ee learns from the relayed credit
  relay_hebb   : relay; W_ee learns from centred co-activity (Hebbian)
  relay_both   : relay; W_ee learns from both
  teacher      : the built-in pools (within-action W_ee = 0.06, frozen) with relay (the reference)
Reported: reward per 250-trial block; structure index = mean within-action W_ee / mean cross-action W_ee; and the
cosine of the relayed credit with the true pool-level credit delta * (1[a = k] - pi_k), pi from the learner's own
recent choices in that context.
"""
import sys, numpy as np, softmax_net as S

K, NE, NI = S.K, S.N_E, S.N_I
N, NEt = S.N, S.K * S.N_E
D, NCTX, M = 20, 8, 16
ALPHA, THETA, REFRAC, T, T0 = S.ALPHA, S.THETA, S.REFRAC, S.T, S.T0
label = np.repeat(np.arange(K), NE)
same = (label[:, None] == label[None, :]) & ~np.eye(NEt, dtype=bool)
W_MAX = 0.12
S_TOT = 0.06 * (NE - 1)           # the teacher's total incoming E weight per neuron


def sim(Wee, drive, rng, w_ei=0.04, w_ie=-0.12, sigma=0.12, rho=0.0, match=False):
    """Batched over learners: Wee (M, NEt, NEt) [post, pre], drive (M, N). Returns action, per-neuron E counts, xi sums."""
    Mm = drive.shape[0]
    V = np.zeros((Mm, N)); ref = np.zeros((Mm, N)); z = np.zeros((Mm, N))
    cnt = np.zeros((Mm, NEt)); xs = np.zeros((Mm, NEt))
    for t in range(T):
        xi = rng.standard_normal((Mm, N))
        if rho > 0:                          # noise shared within each action group
            g = rng.standard_normal((Mm, K))
            xi[:, :NEt] = np.sqrt(rho) * g[:, label] + np.sqrt(1 - rho) * xi[:, :NEt]
            if match:                        # same variance of each group's summed noise as independent noise
                xi[:, :NEt] /= np.sqrt((1 - rho) + NE * rho)
        xs += xi[:, :NEt]
        zE, zI = z[:, :NEt], z[:, NEt:]
        rec = np.zeros((Mm, N))
        rec[:, :NEt] = np.einsum("mij,mj->mi", Wee, zE) + w_ie * zI.sum(1, keepdims=True)
        rec[:, NEt:] = w_ei * zE.sum(1, keepdims=True)
        V = ALPHA * V + drive + rec + sigma * xi
        V = np.where(ref > 0, ALPHA * V, V)
        z = ((V >= THETA) & (ref <= 0)).astype(float)
        V = V - THETA * z
        ref = np.where(z > 0, REFRAC, ref - 1)
        if t >= T0:
            cnt += z[:, :NEt]
    votes = cnt.reshape(Mm, K, NE).sum(2)
    a = np.argmax(votes + rng.random((Mm, K)) * 1e-3, 1)
    return a, cnt, xs


def run(variant, trials=3000, seed=0, eta=0.003, eta_ee=0.002, gamma=1.0, b=0.085, gin=0.02, scaling=True, relay_power=1.0, rho=0.0, match=False):
    rng = np.random.default_rng(seed); task = np.random.default_rng(100 + seed)
    pats = (task.random((NCTX, D)) < 0.3).astype(float); correct = task.integers(0, K, NCTX)
    Win = 0.05 * rng.standard_normal((M, NEt, D)) / np.sqrt(D)
    if variant == "teacher":
        Wee = np.where(same, 0.06, 0.0)[None].repeat(M, 0)
    else:                                   # same total recurrent drive as the teacher, spread over everyone
        Wee = rng.uniform(0, 2 * 0.06 * (NE - 1) / (NEt - 1), (M, NEt, NEt)) * (~np.eye(NEt, dtype=bool))
    relay = variant not in ("own", "hebb_only")
    learn_credit = variant in ("relay_credit", "relay_both")
    learn_hebb = variant in ("relay_hebb", "relay_both", "hebb_only")
    rbar = np.full(M, 0.25); zbar = np.zeros((M, NEt)); ms_in = np.full(M, 1e-12); ms_ee = np.full(M, 1e-12)
    pi_hat = np.full((M, NCTX, K), 0.25)
    rew, struct, cosr, rates = [], [], [], []
    for t in range(trials):
        c = rng.integers(0, NCTX, M); u = pats[c]
        drive = np.zeros((M, N)); drive[:, :NEt] = b + gin * np.einsum("mid,md->mi", Win, u)
        a, cnt, xs = sim(Wee, drive, rng, rho=rho, match=match)
        rates.append(cnt.mean() / (T - T0) * 1000)
        r = (a == correct[c]).astype(float); delta = r - rbar; rbar += 0.02 * (r - rbar); rew.append(r)
        own = delta[:, None] * xs
        if variant == "vote_only":              # only the action group's mean credit (the vote pathway's feedback)
            cr = np.repeat(own.reshape(M, K, NE).mean(2), NE, axis=1)
        elif variant == "relay_vote":             # back through the vote pathway: the action group's mean credit
            cr = own + gamma * np.repeat(own.reshape(M, K, NE).mean(2), NE, axis=1)
        elif relay:
            Rw = Wee ** relay_power                         # steep relay synapse: strong partners dominate
            cr = own + gamma * np.einsum("mij,mj->mi", Rw, own) / (Rw.sum(2) + 1e-12)
        else:
            cr = own
        # quality: cosine with the true pool-level credit delta (1[a = k] - pi_k)
        pk = pi_hat[np.arange(M), c]
        true = (delta[:, None] * (np.eye(K)[a] - pk))[:, label]
        cosr.append(np.mean(np.sum(cr * true, 1) / (np.linalg.norm(cr, axis=1) * np.linalg.norm(true, axis=1) + 1e-12)))
        pi_hat[np.arange(M), c] += 0.05 * (np.eye(K)[a] - pk)
        G = cr[:, :, None] * u[:, None, :]
        ms_in = 0.99 * ms_in + 0.01 * (G ** 2).mean((1, 2))
        Win += eta * G / (np.sqrt(ms_in / (1 - 0.99 ** (t + 1)))[:, None, None] + 1e-12)
        zc = cnt - zbar; zbar += 0.02 * (cnt - zbar)
        if learn_credit or learn_hebb:
            Gee = np.zeros_like(Wee)
            if learn_credit:
                Gee += cr[:, :, None] * zc[:, None, :]
            if learn_hebb:
                h = zc[:, :, None] * zc[:, None, :]
                Gee += h - h.mean((1, 2), keepdims=True)
            Gee *= ~np.eye(NEt, dtype=bool)
            ms_ee = 0.99 * ms_ee + 0.01 * (Gee ** 2).mean((1, 2))
            Wee = np.clip(Wee + eta_ee * Gee / (np.sqrt(ms_ee / (1 - 0.99 ** (t + 1)))[:, None, None] + 1e-12),
                          0.0, W_MAX)
            if scaling:                     # synaptic scaling: each neuron keeps its total incoming E weight
                Wee *= S_TOT / (Wee.sum(2, keepdims=True) + 1e-12)
        if t % 250 == 249:
            struct.append(np.mean(Wee[:, same].mean(1) / (Wee[:, ~same & ~np.eye(NEt, dtype=bool)].mean(1) + 1e-12)))
    R = np.array(rew).reshape(-1, 250, M).mean(1)
    C = np.array(cosr).reshape(-1, 250).mean(1)
    return R, np.array(struct), C, np.array(rates).reshape(-1, 250).mean(1)


if __name__ == "__main__":
    import os
    EETA = float(os.environ.get("EETA", "0.002")); TR = int(os.environ.get("TR", "3000"))
    RP = float(os.environ.get("RP", "1.0"))
    variants = sys.argv[1:] or ["own", "relay_frozen", "relay_credit", "relay_hebb", "relay_both", "teacher"]
    for v in variants:
        R, st, C, hz = run(v, eta_ee=EETA, trials=TR, relay_power=RP, rho=float(os.environ.get("RHO", "0")),
                           match=os.environ.get("MATCH", "0") == "1")
        m = R.mean(1)
        print(f"{v:13s} reward/250: " + " ".join(f"{x:.2f}" for x in m), flush=True)
        print(f"{'':13s} structure (within/cross W_ee): " + " ".join(f"{x:.2f}" for x in st[::2])
              + " | E Hz: " + " ".join(f"{x:.0f}" for x in hz[::2]), flush=True)
