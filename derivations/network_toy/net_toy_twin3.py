"""Twin (antithetic) decomposition of the weight-dist score in the grid toy.

Three staggered twins: twin j is re-synced to the real network every W=30 sim steps (offset 10j)
and from then on gets the same weights and input spikes but mirrored noise (-n). The score at each
step is decomposed with the twin that has run 20-29 steps since its sync, so the noise trace
(decay 0.905/step) keeps < 13% of any un-mirrored pre-sync noise. With s the real score and s' the
twin's score at each step:
    predictable part  P = (s + s') / 2   (survives flipping this step's noise: set by the state)
    noise part        Q = (s - s') / 2   (flips with the noise: the noise-reward correlation)
The twin is a diagnostic; behaviour and delta come from the real network only.
Variants: full (s, code sign), pred (P, code sign), noise (Q, code sign), noise_rf (Q, REINFORCE sign),
plus everything from net_toy_grid.py.

Base toy: Snake-like CPU toy of the live weight-dist rule: 2-D grid forager with an egocentric view.

Same rule and readouts as net_toy.py (see its docstring). What changes, to match snake.py:
- Task: GxG board, food +1, leaving the board = death (-1, respawn), 4 absolute actions with a
  softmax policy, decision every K sim steps. Unbounded egocentric view (VR x VR) with two
  channels: food, wall (cells outside the board).
- Connectivity: toroidal-Gaussian local fan-in (input -> hidden, hidden -> hidden, hidden -> fields),
  hidden E and I sheets, fields on their own sheet, as in EILayer / connect_to_field.
- Weights as make_connectivity: mean = sign * mean_scale / sqrt(fan_in) (x3 for inhibitory),
  sd = sd_scale / sqrt(fan_in), mean_scale 0.1, sd_scale 0.05. E/I signs are set at init only:
  snake.py never runs DalePrune/DaleRewire, so weights may change sign during training.
- Readouts: dense field -> head weights, N(0, 1/sqrt(n_actions)).

Variants: full, drift, fluct, frozen (readouts fixed), reservoir (hidden fixed), and
fire120 (hidden learns only the firing-rate regulariser with f_target 120 Hz; your baseline).
"""
import argparse, time
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--variants", default="full,pred,noise,noise_rf")
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--env-steps", type=int, default=20000)
p.add_argument("--K", type=int, default=10)
p.add_argument("--G", type=int, default=5, help="board size")
p.add_argument("--VR", type=int, default=5, help="view size (odd)")
p.add_argument("--hE", type=int, default=6, help="E sheet side")
p.add_argument("--hI", type=int, default=4, help="I sheet side")
p.add_argument("--hF", type=int, default=4, help="field sheet side (each field)")
p.add_argument("--fan-in", type=int, default=12, help="desired fan-in per projection")
p.add_argument("--sig-in", type=float, default=0.2)
p.add_argument("--sig-h", type=float, default=0.2)
p.add_argument("--mean-scale", type=float, default=0.1)
p.add_argument("--sd-scale", type=float, default=0.05)
p.add_argument("--in-rate", type=float, default=0.5, help="spike prob per step of an active pixel")
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--lr-out", type=float, default=None)
p.add_argument("--log-sig", type=float, default=0.0)
p.add_argument("--beta-score", type=float, default=0.99)
p.add_argument("--creg", type=float, default=1e-4)
p.add_argument("--regs", type=int, default=1)
p.add_argument("--sign", type=float, default=-1.0, help="-1 = code sign (descend on DeltaG)")
p.add_argument("--block", type=int, default=2000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--save", default="net_toy_twin3_last.npz")
p.add_argument("--dump", default=None, help="save per-step samples of P and pre/post state (batch element 0)")
p.add_argument("--dump-start", type=int, default=0, help="sim step to start dumping")
p.add_argument("--dump-every", type=int, default=3)
p.add_argument("--dump-n", type=int, default=3000)
p.add_argument("--twin-sign", type=float, default=-1.0, help="twin noise = twin_sign * n (+1: sanity check, Q must be 0)")
a = p.parse_args()

rng = np.random.default_rng(a.seed)
f32 = np.float32
VARS = a.variants.split(",")
B = len(VARS) * a.seeds
var_of = np.repeat(np.arange(len(VARS)), a.seeds)
K, G, VR = a.K, a.G, a.VR
NC = 2                                          # channels: food, wall
NIN = VR * VR * NC
NE, NI, NF = a.hE ** 2, a.hI ** 2, a.hF ** 2
N = NE + NI
PRE, POST = NIN + N, N + 2 * NF                 # [inp, E, I] -> [E, I, pf, vf]
NA = 4

v_th, n_ref, beta_a = 0.61, 3, 0.0174
alpha, rho = np.exp(-1 / 10), np.exp(-1 / 300)
alpha_o = np.exp(-1 / 10)
gamma = 0.5 ** (1 / K); lam = 0.8 ** (1 / K); rdecay = 0.1 ** (1 / K)
SIG = np.exp(a.log_sig - 5)
b1, b2, eps_opt, l2 = 0.99, 0.99999, 1e-8, 1e-5
alpha_fav = np.exp(-1 / 500)


# ---- sheet coordinates in [0,1)^2 ----
def sheet(side, ch=1):
    ys, xs = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    c = np.stack([ys.ravel(), xs.ravel()], 1) / side + 0.5 / side
    return np.repeat(c, ch, axis=0)             # channel-major within a cell
cin, cE, cI, cF = sheet(VR, NC), sheet(a.hE), sheet(a.hI), sheet(a.hF)


def toroidal(src, dst, fan_in, sigma, r):
    d = np.abs(src[:, None, :] - dst[None, :, :]); d = np.minimum(d, 1 - d)
    g = np.exp(-(d ** 2).sum(-1) / (2 * sigma ** 2))
    pmax = fan_in / g.sum(0).mean()
    return (r.random(g.shape) < np.minimum(1.0, pmax * g)).astype(f32)


def build(seed):
    r = np.random.default_rng(1000 + seed)
    M = np.zeros((PRE, POST), f32); W = np.zeros((PRE, POST), f32)
    rows = {"in": slice(0, NIN), "E": slice(NIN, NIN + NE), "I": slice(NIN + NE, PRE)}
    cols = {"E": slice(0, NE), "I": slice(NE, N), "pf": slice(N, N + NF), "vf": slice(N + NF, POST)}
    csrc = {"in": cin, "E": cE, "I": cI}
    cdst = {"E": cE, "I": cI, "pf": cF, "vf": cF}
    sgn = {"in": 1, "E": 1, "I": -1}
    proj = [("in", "E", a.sig_in), ("in", "I", a.sig_in),
            ("E", "E", a.sig_h), ("E", "I", a.sig_h), ("I", "E", a.sig_h), ("I", "I", a.sig_h),
            ("E", "pf", a.sig_h), ("I", "pf", a.sig_h), ("E", "vf", a.sig_h), ("I", "vf", a.sig_h)]
    for s, d, sg in proj:
        m = toroidal(csrc[s], cdst[d], a.fan_in, sg, r)
        if s == d:
            np.fill_diagonal(m, 0)
        ms = a.mean_scale * (3 if sgn[s] < 0 else 1)
        w = sgn[s] * ms / np.sqrt(a.fan_in) + a.sd_scale / np.sqrt(a.fan_in) * r.normal(0, 1, m.shape)
        M[rows[s], cols[d]] = m; W[rows[s], cols[d]] = w * m
    return M, W


Ms, Ws = zip(*[build(s) for s in range(a.seeds)])
M = np.stack([Ms[i % a.seeds] for i in range(B)])
W = np.stack([Ws[i % a.seeds] for i in range(B)]).astype(f32)
Winit = W.copy()
Wp = np.stack([np.random.default_rng(2000 + i % a.seeds).normal(0, 1 / np.sqrt(NA), (NF, NA)) for i in range(B)]).astype(f32)
Wv = np.stack([np.random.default_rng(3000 + i % a.seeds).normal(0, 1 / np.sqrt(NA), (NF, 1)) for i in range(B)]).astype(f32)
Wp0, Wv0 = Wp.copy(), Wv.copy()

c_score = np.array([{"drift": 0.0, "fire120": 0.0, "pred": 0.0, "noise": 0.0, "noise_rf": 0.0, "both": 0.0, "both_ema": 0.0, "pred_ema": 0.0}.get(v, 1.0) for v in VARS], f32)[var_of][:, None, None]
c_P = np.array([1.0 if v in ("pred", "both") else 0.0 for v in VARS], f32)[var_of][:, None, None]
# "both": drift with the code sign + noise part with the REINFORCE sign = code sign on (P - Q)
c_Q = np.array([{"noise": 1.0, "noise_rf": 1.0, "both": -1.0}.get(v, 0.0) for v in VARS], f32)[var_of][:, None, None]
c_bar = np.array([{"both_ema": 2.0, "pred_ema": 1.0}.get(v, 0.0) for v in VARS], f32)[var_of][:, None]
use_L = np.array([1.0 if v in ("both_ema", "pred_ema") else 0.0 for v in VARS], f32)[var_of][:, None, None]
sgn_h = np.array([1.0 if v == "noise_rf" else a.sign for v in VARS], f32)[var_of][:, None, None]
c_mean = np.array([{"drift": 1.0, "fluct": -1.0}.get(v, 0.0) for v in VARS], f32)[var_of][:, None, None]
c_td = np.array([0.0 if v == "fire120" else 1.0 for v in VARS], f32)[var_of][:, None, None]
lr_h = np.array([0.0 if v == "reservoir" else a.lr for v in VARS], f32)[var_of][:, None, None]
lr_o = np.array([0.0 if v == "frozen" else (a.lr_out or a.lr) for v in VARS], f32)[var_of][:, None, None]
f_target = np.array([120.0 if v == "fire120" else 10.0 for v in VARS], f32)[var_of][:, None] / 1000

z = lambda *s: np.zeros(s, f32)
v, ad, zs, ref = z(B, POST), z(B, POST), z(B, POST), np.zeros((B, POST), np.int32)
zin = z(B, NIN); zbar = z(B, PRE)
xi, epsA, efilt, rln, smean = (z(B, PRE, POST) for _ in range(5))
m_h, s_h = z(B, PRE, POST), z(B, PRE, POST)
# twin state + diagnostic traces of the two parts (lambda-accumulated like RLNoiseTrace)
NT, WIN = 3, 30
v2, ad2, zs2, ref2 = z(NT, B, POST), z(NT, B, POST), z(NT, B, POST), np.zeros((NT, B, POST), np.int32)
xi2, epsA2, zbar2 = z(NT, B, PRE, POST), z(NT, B, PRE, POST), z(NT, B, PRE)
rlP, rlQ = z(B, PRE, POST), z(B, PRE, POST)
psibar, psibar2 = z(B, POST), z(NT, B, POST)
rlPL, rlQL = z(B, PRE, POST), z(B, PRE, POST)        # twin split of the local-gate score (gate psi - 2 psibar)
dg = None
fav, favtr = z(B, POST), z(B, POST)
vp, vv, vprev, r = z(B, NA), z(B, 1), z(B, 1), z(B, 1)
zbar_pf, zbar_vf = z(B, NF), z(B, NF)
rlp, rlv = z(B, NF, NA), z(B, NF, 1)
m_p, s_p, m_v, s_v = z(B, NF, NA), z(B, NF, NA), z(B, NF, 1), z(B, NF, 1)
PG = z(B, NA)
step_count = 0

# ---- environment ----
pos = rng.integers(0, G, (B, 2)); food = rng.integers(0, G, (B, 2))
DIRS = np.array([[0, -1], [-1, 0], [0, 1], [1, 0]])   # left, up, right, down  (dy, dx)
half = VR // 2
off = np.stack(np.meshgrid(np.arange(-half, half + 1), np.arange(-half, half + 1), indexing="ij"), -1).reshape(-1, 2)


def respawn_food(mask):
    global food
    nf = rng.integers(0, G, (B, 2))
    same = (nf == pos).all(1)
    nf[same] = (nf[same] + [0, 1]) % G
    food = np.where(mask[:, None], nf, food)


respawn_food(np.ones(B, bool))


def obs_rates():
    cells = pos[:, None, :] + off[None]                               # (B, VR*VR, 2)
    outside = ((cells < 0) | (cells >= G)).any(-1)
    isfood = (cells == food[:, None, :]).all(-1)
    o = np.stack([isfood, outside], -1).reshape(B, NIN).astype(f32)   # channel-major per cell
    return o * a.in_rate


def adabelief(Wt, m, s, g, lr, init, t):
    m *= b1; m += (1 - b1) * g
    d = g - m; s *= b2; s += (1 - b2) * d * d + eps_opt
    mh = m / (1 - b1 ** t); sh = s / (1 - b2 ** t)
    Wt -= lr * (mh / (np.sqrt(sh) + eps_opt)) + lr * l2 * (Wt - init)


rates = obs_rates()
keys = ("rew", "food", "death", "steps", "spk", "spkf", "psi_open", "PP", "QQ", "PQ", "LL_P", "LL_Q", "PL_P", "QL_Q")
stats = {k: np.zeros(B) for k in keys}
hist = []
zb2 = z(B, PRE)                                       # alpha^2-filtered presynaptic spikes
tsl = np.full((B, POST), 99, np.int32)               # sim steps since last postsynaptic spike
D = {k: [] for k in ("P", "Q", "pre", "zb2", "zbar", "z", "refr", "ref", "vA", "psi", "tsl", "delta")}
t0 = time.time()
for es in range(a.env_steps):
    for k in range(K):
        step_count += 1
        for j in range(NT):
            if (step_count - 1 - 10 * j) % WIN == 0:
                v2[j], ad2[j], zs2[j], ref2[j] = v, ad, zs, ref
                xi2[j], epsA2[j], zbar2[j] = xi, epsA, zbar
                psibar2[j] = psibar
        jt = [j for j in range(NT) if (step_count - 1 - 10 * j) % WIN >= 20][0]   # twin used this step
        zpre = np.concatenate([zin, zs[:, :N]], 1)
        nz = rng.standard_normal((B, POST)).astype(f32)
        # ---- twins: same weights and input spikes, mirrored noise ----
        zpre2 = np.concatenate([np.broadcast_to(zin, (NT, B, NIN)), zs2[:, :, :N]], 2)
        app2 = SIG * (a.twin_sign * nz)[None, :, None, :] * zpre2[:, :, :, None] * M[None]
        I2 = np.einsum("tbi,bij->tbj", zpre2, W) + app2.sum(2)
        xi2 += app2
        ad2 = rho * ad2 + zs2; A2 = v_th + beta_a * ad2
        v2 = alpha * v2 + I2 - v_th * zs2
        ref2 = np.maximum(ref2 - 1, 0); refr2 = ref2 > 0
        zs2 = ((v2 >= A2) & ~refr2).astype(f32); ref2 = np.where(zs2 > 0, n_ref, ref2)
        psi2 = np.where(refr2, 0.0, (0.3 / v_th) * np.maximum(0, 1 - np.abs(v2 - A2) / v_th)).astype(f32)
        zbar2 = alpha * zbar2 + zpre2
        pre2 = zbar2[jt][:, :, None] - beta_a * epsA2[jt]
        psibar2 = 0.99 * psibar2 + 0.01 * psi2
        e2 = psi2[:, :, None, :] * (zbar2[:, :, :, None] - beta_a * epsA2)
        epsA2 = psi2[:, :, None, :] * zbar2[:, :, :, None] + (rho - psi2[:, :, None, :] * beta_a) * epsA2
        score2 = e2[jt] * xi2[jt] / (SIG * SIG + 1e-6)
        scoreL2 = (psi2[jt] - 2 * psibar2[jt])[:, None, :] * pre2 * xi2[jt] / (SIG * SIG + 1e-6)
        xi2 *= alpha
        applied = SIG * nz[:, None, :] * zpre[:, :, None] * M
        I = np.einsum("bi,bij->bj", zpre, W) + applied.sum(1)
        xi += applied
        ad = rho * ad + zs
        A = v_th + beta_a * ad
        v = alpha * v + I - v_th * zs
        ref = np.maximum(ref - 1, 0); refr = ref > 0
        zs = ((v >= A) & ~refr).astype(f32); ref = np.where(zs > 0, n_ref, ref)
        psi = np.where(refr, 0.0, (0.3 / v_th) * np.maximum(0, 1 - np.abs(v - A) / v_th)).astype(f32)
        zpf, zvf = zs[:, N:N + NF], zs[:, N + NF:]
        vp = alpha_o * vp + np.einsum("bi,bik->bk", zpf, Wp)
        vv = alpha_o * vv + np.einsum("bi,bik->bk", zvf, Wv)
        zbar_pf = alpha * zbar_pf + zpf; zbar_vf = alpha * zbar_vf + zvf
        delta = gamma * vv + r - vprev; vprev = vv.copy(); r *= rdecay
        zbar = alpha * zbar + zpre
        pre_term = zbar[:, :, None] - beta_a * epsA
        e = psi[:, None, :] * pre_term
        epsA = psi[:, None, :] * zbar[:, :, None] + (rho - psi[:, None, :] * beta_a) * epsA
        efilt = alpha * efilt + e
        score = e * xi / (SIG * SIG + 1e-6)
        Pt, Qt = 0.5 * (score + score2), 0.5 * (score - score2)
        zb2 = alpha * alpha * zb2 + zpre
        tsl = np.where(zs > 0, 0, np.minimum(tsl + 1, 99))
        if a.dump and step_count >= a.dump_start and (step_count - a.dump_start) % a.dump_every == 0 and len(D["P"]) < a.dump_n:
            for kk, vv_ in (("P", Pt[0]), ("Q", Qt[0]), ("pre", pre_term[0]), ("zb2", zb2[0]), ("zbar", zbar[0]),
                            ("z", zs[0]), ("refr", refr[0]), ("ref", ref[0]), ("vA", (v - A)[0]), ("psi", psi[0]),
                            ("tsl", tsl[0]), ("delta", delta[0])):
                D[kk].append(np.asarray(vv_, np.float16 if kk in ("P", "Q", "pre") else np.float32))
        psibar = 0.99 * psibar + 0.01 * psi
        scoreL = (psi - 2 * psibar)[:, None, :] * pre_term * xi / (SIG * SIG + 1e-6)
        PLt, QLt = 0.5 * (scoreL + scoreL2), 0.5 * (scoreL - scoreL2)
        local_in = (psi - c_bar * psibar)[:, None, :] * pre_term * xi / (SIG * SIG + 1e-6)
        d3 = delta[:, :, None]
        gP, gQ = d3 * rlP * M, d3 * rlQ * M                  # each part's TD update (before trace update)
        gPL, gQL = d3 * rlPL * M, d3 * rlQL * M
        dg_PP = (gP * gP).sum((1, 2)); dg_QQ = (gQ * gQ).sum((1, 2)); dg_PQ = (gP * gQ).sum((1, 2))
        dg_LP = (gPL * gPL).sum((1, 2)); dg_LQ = (gQL * gQL).sum((1, 2))
        dg_PLP = (gPL * gP).sum((1, 2)); dg_QLQ = (gQL * gQ).sum((1, 2))
        rlP = lam * rlP + Pt; rlQ = lam * rlQ + Qt
        rlPL = lam * rlPL + PLt; rlQL = lam * rlQL + QLt
        g = c_td * delta[:, :, None] * rln
        if a.regs:
            fav = fav * alpha_fav + (1 - alpha_fav) * zs
            fire_reg = fav - f_target
            favtr = lam * favtr + (1 - lam) * np.abs(fire_reg)
            fire_rr = favtr - np.abs(fire_reg)
            vreg = np.maximum(0, v - A) + np.maximum(0, -v - A)
            dV = v_th * zs
            g = g + (-0.01 * dV - a.creg * vreg + a.creg * fire_rr)[:, None, :] * rln * c_td \
                  + a.creg * fire_reg[:, None, :] * efilt
        smean = a.beta_score * smean + (1 - a.beta_score) * score
        rln = lam * rln + c_score * score + c_mean * smean + c_P * Pt + c_Q * Qt + use_L * local_in
        xi *= alpha
        adabelief(W, m_h, s_h, -sgn_h * g * M, lr_h, Winit, step_count)
        gp = delta[:, :, None] * rlp
        rlp = lam * rlp + zbar_pf[:, :, None] * PG[:, None, :]
        gv = rlv * delta[:, :, None]
        rlv = gamma * lam * rlv - zbar_vf[:, :, None]
        adabelief(Wp, m_p, s_p, gp, lr_o, Wp0, step_count)
        adabelief(Wv, m_v, s_v, gv, lr_o, Wv0, step_count)
        PG[:] = 0
        stats["spk"] += zs[:, :N].mean(1); stats["spkf"] += zs[:, N:].mean(1)
        stats["psi_open"] += (psi[:, :N] > 0).mean(1)
        stats["PP"] += dg_PP; stats["QQ"] += dg_QQ; stats["PQ"] += dg_PQ
        stats["LL_P"] += dg_LP; stats["LL_Q"] += dg_LQ; stats["PL_P"] += dg_PLP; stats["QL_Q"] += dg_QLQ
        if k == K - 1:
            lg = vp - vp.max(1, keepdims=True); pr = np.exp(lg); pr /= pr.sum(1, keepdims=True)
            act = (rng.random(B)[:, None] > np.cumsum(pr, 1)).sum(1).clip(0, NA - 1)
            PG = pr.copy(); PG[np.arange(B), act] -= 1
            pos = pos + DIRS[act]
            dead = ((pos < 0) | (pos >= G)).any(1)
            ate = ~dead & (pos == food).all(1)
            rew = ate.astype(f32) - dead.astype(f32)
            pos = np.where(dead[:, None], rng.integers(0, G, (B, 2)), pos)
            respawn_food(ate | dead | (pos == food).all(1))
            r = np.where(rew[:, None] != 0, rew[:, None], r)
            stats["rew"] += rew; stats["food"] += ate; stats["death"] += dead; stats["steps"] += 1
            rates = obs_rates()
        zin = (rng.random((B, NIN)) < rates).astype(f32)
    if (es + 1) % a.block == 0:
        st = stats["steps"]; rate = stats["rew"] / st
        line = f"[{es+1:6d}, {time.time()-t0:4.0f}s] "
        for vi, vn in enumerate(VARS):
            sel = var_of == vi
            hz = 1000 * stats["spk"][sel].mean() / (st[sel].mean() * K)
            hzf = 1000 * stats["spkf"][sel].mean() / (st[sel].mean() * K)
            fd = stats["food"][sel].mean() / st[sel].mean(); dd = stats["death"][sel].mean() / st[sel].mean()
            pp, qq, pq = stats["PP"][sel].sum(), stats["QQ"][sel].sum(), stats["PQ"][sel].sum()
            line += (f"{vn}: {rate[sel].mean():+.3f}±{rate[sel].std():.3f} d{dd:.2f} ({hz:3.0f}/{hzf:3.0f}Hz)"
                     f" cos(P_loc,P)={stats['PL_P'][sel].sum() / np.sqrt(stats['LL_P'][sel].sum() * pp):+.2f}"
                     f" cos(Q_loc,Q)={stats['QL_Q'][sel].sum() / np.sqrt(stats['LL_Q'][sel].sum() * qq):+.2f}"
                     f" |Q_loc|/|Q|={np.sqrt(stats['LL_Q'][sel].sum() / qq):.2f}  ")
        print(line, flush=True)
        hist.append(rate.copy())
        stats = {k: np.zeros(B) for k in keys}

if a.dump:
    np.savez(a.dump, M=M[0], N=N, NIN=NIN, NF=NF, **{k: np.array(v_) for k, v_ in D.items()})
np.savez(a.save, variants=VARS, var_of=var_of, hist=np.array(hist), args=str(vars(a)))
