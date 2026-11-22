"""Snake-like CPU toy of the live weight-dist rule: 2-D grid forager with an egocentric view.

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

Local estimates of "both" (drift with the code sign + noise part with the REINFORCE sign), all with
the code sign and the same traces; only the gate multiplying pre * NoiseTrace / SynSig^2 changes:
  both_v0   gate = psi - 2 psi0, psi0 = psi(V - eta) with eta = sum_i NoiseTrace_ij (the noise the
            neuron has received), zero while the real neuron is refractory
  both_v0n  same, psi0 ignores refractoriness
  both_ema  gate = psi - 2 psibar, psibar = per-neuron EMA of psi (decay 0.99 per step)
  pred_v0   gate = psi - psi0 (local estimate of the predictable part only)

Noise-free Hebbian (Clopath-like) rules, input to RLNoiseTrace = h(post state) * pre_term * zb2_i,
h from the table measured with hebb_check.py (time since last post spike / V - A bins):
  hebb_tab     table rule only (no noise term at all)
  hebb_tab_rf  table rule + local REINFORCE term -psibar * pre_term * NoiseTrace / SynSig^2
  pre_only     constant h (control: does the table's shape matter?)

Clopath et al. (2010) voltage-based rule, mapped to the ALIF toy (theta- = rest = 0, LTP at post spikes):
  C_ij = A_ltp * xbar_i * z_j * [ubar+_j - theta]_+  -  A_ltd * z_i * [ubar-_j - theta]_+
  ubar- / ubar+ = post voltage low-passed (tau 10 / 7), xbar = pre spikes low-passed (tau 15).
  clopath_unsup  hidden weights follow C directly (unsupervised), readouts learn by TD
  clopath_rl     three-factor: RLNoiseTrace input = -C, so dW ~ +delta * lambda-trace(C)
  clopath_rl_rf  clopath_rl + local REINFORCE term -psibar * pre_term * NoiseTrace / SynSig^2

Noise-free gate estimates G for the "both" gate psi - 2 G:
  be90 / be99 / be999   G = EMA of psi with decay 0.9 / 0.99 / 0.999   (be99 == both_ema)
  blf    leak-free two-trace G = a*EMA_0.99(psi) - b*EMA_alpha(psi), a - b = 1, weights chosen so
         that G has zero overlap with the noise trace's memory (sum_k K(k) alpha^k = 0)
  breg   regression-corrected G = psi - bhat*(eta - mean eta), bhat = cov(psi, eta)/var(eta) online
         (decay 0.99), eta = noise in the neuron's membrane = sum_i NoiseTrace_ij
"""
import argparse, time
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--variants", default="full,reservoir")
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
p.add_argument("--save", default="net_toy_grid_last.npz")
p.add_argument("--hidden-delta", default="real", choices=["real", "shuffle", "const", "negconst", "randsign"],
               help="TD error used by the HIDDEN update only: real | shuffle (random past delta) | const (running mean |delta|)")
p.add_argument("--psi-w", type=float, default=0.61, help="psi half-width (default v_th); peak height kept at 0.3/v_th")
p.add_argument("--save-weights", default=None, help="save W, M, Wp, Wv of batch element 0 at the end")
p.add_argument("--clp-theta", type=float, default=0.0)
p.add_argument("--a-ltp", type=float, default=1.0)
p.add_argument("--a-ltd", type=float, default=1.0)
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

HTAB = np.array([-0.550, -1.183, -0.489, 0.149, 0.019, 0.507, 1.195, 0.585, 0.0], np.float32)
c_tab = np.array([1.0 if v in ("hebb_tab", "hebb_tab_rf") else 0.0 for v in VARS], np.float32)
c_const = np.array([0.32 if v == "pre_only" else 0.0 for v in VARS], np.float32)
c_clp_rl = np.array([1.0 if v in ("clopath_rl", "clopath_rl_rf") else 0.0 for v in VARS], np.float32)
c_clp_un = np.array([1.0 if v == "clopath_unsup" else 0.0 for v in VARS], np.float32)
GTAB = {"be90": (0, 1, -2), "be99": (1, 1, -2), "be999": (2, 1, -2), "blf": (3, 1, -2), "breg": (4, 1, -2),
        "drift_ema": (1, 1, -1), "grad_ema": (1, 0, -1), "drift_reg": (4, 1, -1), "grad_reg": (4, 0, -1),
        "ind_be99": (1, 1, -2), "det_rf": (1, 0, -1), "split0": (1, 0, -1), "split1": (1, 0, -1), "split4": (1, 0, -1)}
GVARS = {k: v[0] for k, v in GTAB.items()}
gsel = np.array([GVARS.get(v, -1) for v in VARS])[var_of]                      # (B,)
gpsi = np.array([GTAB.get(v, (0, 0, 0))[1] for v in VARS], np.float32)[var_of][:, None]
gG = np.array([GTAB.get(v, (0, 0, 0))[2] for v in VARS], np.float32)[var_of][:, None]
# unbiased: exact likelihood-ratio score (fresh noise x presynaptic spike, no gate, no filter)
c_unb = np.array([1.0 if v == "unbiased" else 0.0 for v in VARS], np.float32)[var_of][:, None, None]
# ind_*: independent noise per synapse instead of shared per postsynaptic neuron
is_ind = np.array([v.startswith("ind_") for v in VARS])[var_of]
# split*: the drift part (psi - psibar) gets its own trace with decay lam_P (0, lambda, lambda^(1/4)); the gradient
# part (-psibar) stays in RLNoiseTrace (gate table above gives psi coef 0, G coef -1 for these)
# det_*: noise-free drift from a deterministic second-order surrogate (verify_det.py), scaled by the measured
# slope 3.8 of E[s] on it; det_rf adds the gradient part -psibar*pre*NT (REINFORCE sign), det_only does not
c_det = np.array([3.8 if v in ("det_rf", "det_only") else 0.0 for v in VARS], np.float32)[var_of][:, None, None]
KMAX_DET = 10
lamP_of = {"split0": 0.0, "split1": 0.8 ** (1 / a.K), "split4": 0.8 ** (1 / (4 * a.K))}
c_split = np.array([1.0 if v in lamP_of else 0.0 for v in VARS], np.float32)[var_of][:, None, None]
lamP = np.array([lamP_of.get(v, 0.0) for v in VARS], np.float32)[var_of][:, None, None]
f_ = lambda b_: (1 - b_) / (1 - alpha * b_)
lf_a = f_(alpha) / (f_(alpha) - f_(0.99)); lf_b = f_(0.99) / (f_(alpha) - f_(0.99))
GATES = {"hebb_tab_rf": (0, 0, 0, -1), "clopath_rl_rf": (0, 0, 0, -1), "both_v0": (1, -2, 0, 0), "both_v0n": (1, 0, -2, 0), "both_ema": (1, 0, 0, -2), "pred_v0": (1, -1, 0, 0)}
c_score = np.array([{"drift": 0.0, "fire120": 0.0, "hebb_tab": 0.0, "pre_only": 0.0, "clopath_rl": 0.0, "clopath_unsup": 0.0, "det_rf": 0.0, "det_only": 0.0, **{g: 0.0 for g in GVARS}, **{g: 0.0 for g in GATES}, "unbiased": 0.0}.get(v, 1.0) for v in VARS], f32)[var_of][:, None, None]
gc = np.array([GATES.get(v, (0, 0, 0, 0)) for v in VARS], f32)[var_of]          # (B, 4)
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
fav, favtr = z(B, POST), z(B, POST)
psibar = z(B, POST)
zb2 = z(B, PRE); tsl = np.full((B, POST), 99, np.int32)
ubm, ubp, xbp = z(B, POST), z(B, POST), z(B, PRE)
rlP_h = z(B, PRE, POST)
H_psipi = np.zeros((KMAX_DET, B, PRE, POST), f32); H_z = np.zeros((KMAX_DET, B, POST), f32)
dbuf = np.zeros((20000, B, 1), f32); absd = z(B, 1); stats_dsum = np.zeros(B)
e90, e99, e999, ea = z(B, POST), z(B, POST), z(B, POST), z(B, POST)
r_mp, r_me, r_cpe, r_vee = z(B, POST), z(B, POST), z(B, POST), z(B, POST)
a_m, a_p, a_x = np.exp(-1 / 10), np.exp(-1 / 7), np.exp(-1 / 15)
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
keys = ("rew", "food", "death", "steps", "spk", "spkf", "psi_open")
stats = {k: np.zeros(B) for k in keys}
hist = []
t0 = time.time()
for es in range(a.env_steps):
    for k in range(K):
        step_count += 1
        zpre = np.concatenate([zin, zs[:, :N]], 1)
        nz = rng.standard_normal((B, POST)).astype(f32)
        if is_ind.any():
            nsyn = np.where(is_ind[:, None, None], rng.standard_normal((B, PRE, POST)).astype(f32), nz[:, None, :])
            applied = SIG * nsyn * zpre[:, :, None] * M
        else:
            applied = SIG * nz[:, None, :] * zpre[:, :, None] * M
        I = np.einsum("bi,bij->bj", zpre, W) + applied.sum(1)
        xi += applied
        ad = rho * ad + zs
        A = v_th + beta_a * ad
        v = alpha * v + I - v_th * zs
        ref = np.maximum(ref - 1, 0); refr = ref > 0
        zs = ((v >= A) & ~refr).astype(f32); ref = np.where(zs > 0, n_ref, ref)
        psi = np.where(refr, 0.0, (0.3 / v_th) * np.maximum(0, 1 - np.abs(v - A) / a.psi_w)).astype(f32)
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
        eta = (xi * M).sum(1)                                           # noise received by each neuron
        d0 = np.maximum(0, 1 - np.abs(v - eta - A) / v_th) * (0.3 / v_th)
        psi0r = np.where(refr, 0.0, d0).astype(f32); psi0n = d0.astype(f32)
        psibar = 0.99 * psibar + 0.01 * psi
        gate = (gc[:, 0:1] * psi + gc[:, 1:2] * psi0r + gc[:, 2:3] * psi0n + gc[:, 3:4] * psibar).astype(f32)
        local = gate[:, None, :] * pre_term * xi / (SIG * SIG + 1e-6)
        zb2 = alpha * alpha * zb2 + zpre
        tsl = np.where(zs > 0, 0, np.minimum(tsl + 1, 99))
        vA_ = v - A
        hb = np.select([tsl == 0, tsl == 1, tsl == 2, tsl == 3, vA_ < -1.2, vA_ < -0.6, vA_ < -0.3, vA_ < 0],
                       [0, 1, 2, 3, 4, 5, 6, 7], 8)
        hpost = HTAB[hb] * c_tab[var_of][:, None] + c_const[var_of][:, None]
        local = local + hpost[:, None, :] * pre_term * zb2[:, :, None]
        # ---- Clopath et al. 2010 ----
        ubm = a_m * ubm + (1 - a_m) * v; ubp = a_p * ubp + (1 - a_p) * v; xbp = a_x * xbp + (1 - a_x) * zpre
        C = (a.a_ltp * xbp[:, :, None] * (zs * np.maximum(ubp - a.clp_theta, 0))[:, None, :]
             - a.a_ltd * zpre[:, :, None] * np.maximum(ubm - a.clp_theta, 0)[:, None, :])
        local = local - c_clp_rl[var_of][:, None, None] * C
        # ---- noise-free gate estimates ----
        e90 = 0.9 * e90 + 0.1 * psi; e99 = 0.99 * e99 + 0.01 * psi; e999 = 0.999 * e999 + 0.001 * psi
        ea = alpha * ea + (1 - alpha) * psi
        r_mp = 0.99 * r_mp + 0.01 * psi; r_me = 0.99 * r_me + 0.01 * eta
        r_cpe = 0.99 * r_cpe + 0.01 * psi * eta; r_vee = 0.99 * r_vee + 0.01 * eta * eta
        bhat = (r_cpe - r_mp * r_me) / (r_vee - r_me * r_me + 1e-12)
        Gall = np.stack([e90, e99, e999, lf_a * e99 - lf_b * ea, psi - bhat * (eta - r_me)])   # (5, B, POST)
        if (gsel >= 0).any():
            Gest = Gall[np.maximum(gsel, 0), np.arange(B)] * (gsel >= 0)[:, None]
            local = local + ((gpsi * psi + gG * Gest) * (gsel >= 0)[:, None])[:, None, :] * pre_term * xi / (SIG * SIG + 1e-6)
        local = local + c_unb * applied / (SIG * SIG + 1e-6)
        if c_det.any():
            cps = 0.3 / v_th; xA = v - A
            psif = lambda x: cps * np.maximum(0, 1 - np.abs(x) / a.psi_w)
            slope_ = np.where(xA < 0, cps / a.psi_w, -cps / a.psi_w) * (np.abs(xA) < a.psi_w) * (~refr)
            det = pre_term * (slope_[:, None, :] * pre_term)
            acc = np.zeros((B, PRE, POST), f32)
            for dd in range(1, KMAX_DET + 1):
                k_ = (step_count - dd) % KMAX_DET
                spk = H_z[k_] > 0; sh = v_th * alpha ** dd
                if dd <= 2:
                    dpsi = -np.where(spk, psif(xA + sh), psi)
                else:
                    dpsi = (np.where(spk, psi, np.where(refr, 0.0, psif(xA - sh)))
                            - np.where(spk, np.where(refr, 0.0, psif(xA + sh)), psi))
                acc += dpsi[:, None, :] * H_psipi[k_]
            det = det + pre_term * acc
            H_psipi[step_count % KMAX_DET] = psi[:, None, :] * pre_term; H_z[step_count % KMAX_DET] = zs
            local = local + c_det * det
        # TD error seen by the hidden update (readouts always use the real delta)
        dbuf[step_count % len(dbuf)] = delta
        absd = 0.999 * absd + 0.001 * np.abs(delta)
        if a.hidden_delta == "shuffle":
            dh = dbuf[rng.integers(0, min(step_count, len(dbuf)), B), np.arange(B)]
        elif a.hidden_delta == "const":
            dh = absd
        elif a.hidden_delta == "negconst":
            dh = -absd
        elif a.hidden_delta == "randsign":
            dh = delta * np.where(rng.random((B, POST)) < 0.5, -1.0, 1.0).astype(f32)   # per neuron and step
        else:
            dh = delta
        stats_dsum += delta[:, 0]
        dmod = (dh[:, None, :] if dh.shape[1] == POST else dh[:, :, None])
        g = c_td * dmod * (rln + c_split * rlP_h)                       # DeltaG uses traces before update
        if c_split.any():
            Pd = ((psi - e99) * c_split[:, 0])[:, None, :] * pre_term * xi / (SIG * SIG + 1e-6)
            rlP_h = lamP * rlP_h + Pd
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
        rln = lam * rln + c_score * score + c_mean * smean + local
        xi *= alpha
        g = g - c_clp_un[var_of][:, None, None] * C        # unsupervised: descend on -C, i.e. dW ~ +C
        adabelief(W, m_h, s_h, -a.sign * g * M, lr_h, Winit, step_count)
        gp = delta[:, :, None] * rlp
        rlp = lam * rlp + zbar_pf[:, :, None] * PG[:, None, :]
        gv = rlv * delta[:, :, None]
        rlv = gamma * lam * rlv - zbar_vf[:, :, None]
        adabelief(Wp, m_p, s_p, gp, lr_o, Wp0, step_count)
        adabelief(Wv, m_v, s_v, gv, lr_o, Wv0, step_count)
        PG[:] = 0
        stats["spk"] += zs[:, :N].mean(1); stats["spkf"] += zs[:, N:].mean(1)
        stats["psi_open"] += (psi[:, :N] > 0).mean(1)
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
            line += f"{vn}: {rate[sel].mean():+.3f}±{rate[sel].std():.3f} f{fd:.2f}/d{dd:.2f} ({hz:3.0f}/{hzf:3.0f}Hz)  "
        line += f" mean_delta={stats_dsum.mean() / (st.mean() * K):+.5f}"; stats_dsum[:] = 0
        print(line, flush=True)
        hist.append(rate.copy())
        stats = {k: np.zeros(B) for k in keys}

if a.save_weights:
    np.savez(a.save_weights, W=W[0], M=M[0], Wp=Wp[0], Wv=Wv[0], NIN=NIN, N=N, NF=NF, rates=rates[0])
np.savez(a.save, variants=VARS, var_of=var_of, hist=np.array(hist), args=str(vars(a)))
