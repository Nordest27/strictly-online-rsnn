"""Core of the analyses of perturbation-based e-prop (paper, Appendices D-F).

A small Snake-like task and a recurrent ALIF network that follow the ml_genn implementation:

* GridForager: G x G board, food +1, leaving the board -1 (respawn), four actions, egocentric
  VR x VR view with a food and a wall channel, Poisson inputs; one action every K steps.
* EpropToyNetwork: inputs -> recurrent E/I ALIF core -> policy and value field layers -> leaky
  readouts. Toroidal-Gaussian fan-in, weights mean +-c/sqrt(K) (inhibitory x3), sd 0.05/sqrt(K),
  Dale sign at initialisation only. Shared noise per postsynaptic neuron delivered on presynaptic
  spikes, per-synapse noise trace, e-prop ALIF eligibility, pseudo-derivative gate psi.
* Learner: batch of independent networks (variants x seeds) trained online with AdaBelief on
  every step; hidden weights by the perturbation rule with a configurable gate, readouts by exact
  TD(lambda) actor-critic updates. The optimiser DESCENDS on DeltaG, as in the implementation.
* StaggeredTwins: antithetic twins that split the score into its drift P and noise part Q.

Notation (paper): score s = kappa * g * pi * xi / sigma^2, with pi = zbar - beta*epsA the presynaptic
factor of e-prop's eligibility, g a postsynaptic gate (g = psi gives e-prop's eligibility e = psi*pi)
and kappa = 1 - alpha^2. The update descends on sum_t delta_t p_t with p_t = lambda p_{t-1} + s_t.
"""
from dataclasses import dataclass, field, replace
import numpy as np

f32 = np.float32


@dataclass
class Config:
    # task
    board: int = 5
    view: int = 5
    K: int = 10                      # simulation steps per action
    in_rate: float = 0.5             # spike probability per step of an active pixel
    gamma_env: float = 0.5           # per action
    lam_env: float = 0.8             # per action
    reward_decay_env: float = 0.1    # per action
    # network
    hE: int = 6                      # E sheet side
    hI: int = 4                      # I sheet side
    hF: int = 4                      # field sheet side (each field)
    fan_in: int = 12
    sig_in: float = 0.2
    sig_h: float = 0.2
    mean_scale: float = 0.1
    sd_scale: float = 0.05
    # neurons (ml_genn defaults used in the experiments)
    v_th: float = 0.61
    tau_mem: float = 10.0
    tau_adapt: float = 300.0
    beta: float = 0.0174
    n_ref: int = 3
    psi_width: float = 0.61          # half-width of the gate; peak height 0.3 / v_th
    # perturbation and rule
    log_sigma: float = -5.0          # SynSig = exp(log_sigma)
    kappa: bool = True               # multiply the score by 1 - alpha^2
    psibar_decay: float = 0.99
    # optimiser and regularisers (as in the implementation)
    lr: float = 1e-4
    beta1: float = 0.99
    beta2: float = 0.99999
    eps: float = 1e-8
    l2_init: float = 1e-5
    regs: bool = True
    c_reg: float = 1e-4
    f_target_hz: float = 10.0
    tau_reg: float = 500.0

    # derived per-step constants
    @property
    def alpha(self): return float(np.exp(-1 / self.tau_mem))
    @property
    def rho(self): return float(np.exp(-1 / self.tau_adapt))
    @property
    def gamma(self): return self.gamma_env ** (1 / self.K)
    @property
    def lam(self): return self.lam_env ** (1 / self.K)
    @property
    def rdecay(self): return self.reward_decay_env ** (1 / self.K)
    @property
    def sigma(self): return float(np.exp(self.log_sigma))
    @property
    def kappa_val(self): return (1 - self.alpha ** 2) if self.kappa else 1.0


# ----------------------------------------------------------------------------------------------
# environment
# ----------------------------------------------------------------------------------------------
class GridForager:
    DIRS = np.array([[0, -1], [-1, 0], [0, 1], [1, 0]])     # left, up, right, down (dy, dx)

    def __init__(self, cfg: Config, B: int, rng: np.random.Generator):
        self.cfg, self.B, self.rng = cfg, B, rng
        h = cfg.view // 2
        self.off = np.stack(np.meshgrid(np.arange(-h, h + 1), np.arange(-h, h + 1), indexing="ij"), -1).reshape(-1, 2)
        self.pos = rng.integers(0, cfg.board, (B, 2))
        self.food = np.zeros((B, 2), int)
        self.swap_channels = False       # task switch: food and wall channels exchanged
        self._respawn_food(np.ones(B, bool))

    @property
    def n_inputs(self): return self.cfg.view ** 2 * 2

    def _respawn_food(self, mask):
        nf = self.rng.integers(0, self.cfg.board, (self.B, 2))
        same = (nf == self.pos).all(1)
        nf[same] = (nf[same] + [0, 1]) % self.cfg.board
        self.food = np.where(mask[:, None], nf, self.food)

    def rates(self):
        cells = self.pos[:, None, :] + self.off[None]
        outside = ((cells < 0) | (cells >= self.cfg.board)).any(-1)
        isfood = (cells == self.food[:, None, :]).all(-1)
        ch = [outside, isfood] if self.swap_channels else [isfood, outside]
        return np.stack(ch, -1).reshape(self.B, -1).astype(f32) * self.cfg.in_rate

    def step(self, act):
        self.pos = self.pos + self.DIRS[act]
        dead = ((self.pos < 0) | (self.pos >= self.cfg.board)).any(1)
        ate = ~dead & (self.pos == self.food).all(1)
        self.pos = np.where(dead[:, None], self.rng.integers(0, self.cfg.board, (self.B, 2)), self.pos)
        self._respawn_food(ate | dead | (self.pos == self.food).all(1))
        return ate.astype(f32) - dead.astype(f32), ate, dead


# ----------------------------------------------------------------------------------------------
# connectivity
# ----------------------------------------------------------------------------------------------
def _sheet(side, ch=1):
    ys, xs = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    c = np.stack([ys.ravel(), xs.ravel()], 1) / side + 0.5 / side
    return np.repeat(c, ch, axis=0)


def _toroidal(src, dst, fan_in, sigma, r):
    d = np.abs(src[:, None, :] - dst[None, :, :]); d = np.minimum(d, 1 - d)
    g = np.exp(-(d ** 2).sum(-1) / (2 * sigma ** 2))
    return (r.random(g.shape) < np.minimum(1.0, fan_in / g.sum(0).mean() * g)).astype(f32)


@dataclass
class Layout:
    NIN: int
    NE: int
    NI: int
    NF: int
    @property
    def N(self): return self.NE + self.NI                      # recurrent core
    @property
    def PRE(self): return self.NIN + self.N                    # [inputs, E, I]
    @property
    def POST(self): return self.N + 2 * self.NF                # [E, I, policy field, value field]
    @property
    def pf(self): return slice(self.N, self.N + self.NF)
    @property
    def vf(self): return slice(self.N + self.NF, self.POST)


def build_weights(cfg: Config, lay: Layout, seed: int):
    """Mask M (PRE x POST) and initial weights W for one seed."""
    r = np.random.default_rng(1000 + seed)
    cin, cE, cI, cF = _sheet(cfg.view, 2), _sheet(cfg.hE), _sheet(cfg.hI), _sheet(cfg.hF)
    rows = {"in": slice(0, lay.NIN), "E": slice(lay.NIN, lay.NIN + lay.NE), "I": slice(lay.NIN + lay.NE, lay.PRE)}
    cols = {"E": slice(0, lay.NE), "I": slice(lay.NE, lay.N), "pf": lay.pf, "vf": lay.vf}
    csrc = {"in": cin, "E": cE, "I": cI}
    cdst = {"E": cE, "I": cI, "pf": cF, "vf": cF}
    sgn = {"in": 1, "E": 1, "I": -1}
    M = np.zeros((lay.PRE, lay.POST), f32); W = np.zeros((lay.PRE, lay.POST), f32)
    for s, d in [("in", "E"), ("in", "I"), ("E", "E"), ("E", "I"), ("I", "E"), ("I", "I"),
                 ("E", "pf"), ("I", "pf"), ("E", "vf"), ("I", "vf")]:
        sg = cfg.sig_in if s == "in" else cfg.sig_h
        m = _toroidal(csrc[s], cdst[d], cfg.fan_in, sg, r)
        if s == d:
            np.fill_diagonal(m, 0)
        ms = cfg.mean_scale * (3 if sgn[s] < 0 else 1)
        w = sgn[s] * ms / np.sqrt(cfg.fan_in) + cfg.sd_scale / np.sqrt(cfg.fan_in) * r.normal(0, 1, m.shape)
        M[rows[s], cols[d]] = m; W[rows[s], cols[d]] = w * m
    Wp = np.random.default_rng(2000 + seed).normal(0, 0.5, (lay.NF, 4)).astype(f32)
    Wv = np.random.default_rng(3000 + seed).normal(0, 0.5, (lay.NF, 1)).astype(f32)
    return M, W, Wp, Wv


# ----------------------------------------------------------------------------------------------
# network dynamics (shared by the learner, the twins and the snapshot rollouts)
# ----------------------------------------------------------------------------------------------
class NeuronState:
    """State of a batch of networks (leading dimension B)."""
    FIELDS = ("v", "a", "z", "ref", "zbar", "epsA", "xi", "psibar")

    def __init__(self, B, lay: Layout):
        P, Q = lay.PRE, lay.POST
        self.v, self.a, self.z = (np.zeros((B, Q), f32) for _ in range(3))
        self.ref = np.zeros((B, Q), np.int32)
        self.zbar = np.zeros((B, P), f32)
        self.epsA, self.xi = np.zeros((B, P, Q), f32), np.zeros((B, P, Q), f32)
        self.psibar = np.zeros((B, Q), f32)

    def copy_from(self, other, idx=slice(None)):
        for f in self.FIELDS:
            getattr(self, f)[idx] = getattr(other, f)
        return self


def gate_psi(cfg: Config, v, A, refr):
    return np.where(refr, 0.0, (0.3 / cfg.v_th) * np.maximum(0, 1 - np.abs(v - A) / cfg.psi_width)).astype(f32)


def advance(cfg: Config, lay: Layout, st: NeuronState, W, M, zin, nz, dI=None):
    """One simulation step, in the order of the ml_genn/GeNN implementation:
    integrate -> psi (pre-reset voltage, refractory state after decrement) -> threshold -> reset,
    then e-prop eligibility and noise trace. W: (B,PRE,POST) or (PRE,POST). nz: (B,POST) noise.
    dI: optional extra input current (B,POST). Returns the quantities the rules need; the noise
    trace returned is the one used by this step's score (it is decayed afterwards)."""
    alpha, rho = cfg.alpha, cfg.rho
    zpre = np.concatenate([zin, st.z[:, :lay.N]], 1)                       # spikes of the previous step
    applied = cfg.sigma * nz[:, None, :] * zpre[:, :, None] * M            # delivered noise per synapse
    I = (np.einsum("bi,bij->bj", zpre, W) if W.ndim == 3 else zpre @ W) + applied.sum(1)
    if dI is not None:
        I = I + dI
    st.xi += applied
    st.a = rho * st.a + st.z
    A = cfg.v_th + cfg.beta * st.a
    st.v = alpha * st.v + I - cfg.v_th * st.z                              # relative reset of last spike
    st.ref = np.maximum(st.ref - 1, 0); refr = st.ref > 0
    z_new = ((st.v >= A) & ~refr).astype(f32)
    st.ref = np.where(z_new > 0, cfg.n_ref, st.ref)
    psi = gate_psi(cfg, st.v, A, refr)
    st.zbar = alpha * st.zbar + zpre
    pi = st.zbar[:, :, None] - cfg.beta * st.epsA                          # presynaptic factor (old epsA)
    st.epsA = psi[:, None, :] * st.zbar[:, :, None] + (rho - psi[:, None, :] * cfg.beta) * st.epsA
    st.psibar = cfg.psibar_decay * st.psibar + (1 - cfg.psibar_decay) * psi
    xi = st.xi.copy()
    st.xi *= alpha
    st.z = z_new
    return dict(zpre=zpre, psi=psi, pi=pi, xi=xi, A=A, refr=refr, applied=applied)


def score(cfg: Config, gate, pi, xi):
    """s = kappa * g * pi * xi / sigma^2 for a postsynaptic gate g (B,POST)."""
    return cfg.kappa_val * gate[:, None, :] * pi * xi / (cfg.sigma ** 2 + 1e-6)


def adabelief(Wt, m, s, g, lr, init, t, cfg: Config, l2=None):
    m *= cfg.beta1; m += (1 - cfg.beta1) * g
    d = g - m; s *= cfg.beta2; s += (1 - cfg.beta2) * d * d + cfg.eps
    mh = m / (1 - cfg.beta1 ** t); sh = s / (1 - cfg.beta2 ** t)
    Wt -= lr * (mh / (np.sqrt(sh) + cfg.eps)) + lr * (cfg.l2_init if l2 is None else l2) * (Wt - init)


# ----------------------------------------------------------------------------------------------
# antithetic twins
# ----------------------------------------------------------------------------------------------
class StaggeredTwins:
    """n twins, re-synchronised to the real network every `window` steps with offsets window/n; each
    receives the same weights and input spikes but mirrored noise (sign = -1; sign = +1 reproduces the
    network exactly, used as a test). The score at each step is decomposed with the twin that has run
    for at least `use_from` steps, so that the noise trace keeps < alpha^use_from of unmirrored noise."""

    def __init__(self, cfg, lay, B, n=3, window=30, use_from=20, sign=-1.0):
        assert use_from >= window - window // n
        self.cfg, self.lay, self.n, self.window, self.use_from, self.sign = cfg, lay, n, window, use_from, sign
        self.states = [NeuronState(B, lay) for _ in range(n)]

    def step(self, t, main: NeuronState, W, M, zin, nz):
        """t: 0-based step counter. Call BEFORE advancing the main network at step t (the twins are
        synchronised to the main state at the start of the step). Returns the selected twin's output."""
        offs = [j * self.window // self.n for j in range(self.n)]
        for j, st in enumerate(self.states):
            if (t - offs[j]) % self.window == 0:
                st.copy_from(main)
        outs = [advance(self.cfg, self.lay, st, W, M, zin, self.sign * nz) for st in self.states]
        jt = [j for j in range(self.n) if (t - offs[j]) % self.window >= self.use_from][0]
        return outs[jt], self.states[jt]


# ----------------------------------------------------------------------------------------------
# learner
# ----------------------------------------------------------------------------------------------
@dataclass
class Variant:
    """How the hidden layer learns. RLNoiseTrace input = gate-based score, or a twin-based part.
    gate = c_psi * psi + c_bar * psibar.  twin_P / twin_Q: coefficients on the twin-estimated drift P
    and noise part Q (then the gate is not used).  hidden_lr / readout_lr: 0 freezes that part."""
    name: str
    c_psi: float = 1.0
    c_bar: float = 0.0
    twin_P: float = 0.0
    twin_Q: float = 0.0
    hidden_lr: bool = True
    readout_lr: bool = True
    sched: tuple = None              # optional (a0, b0, a1, b1): drift weight a and gradient weight b go
                                     # linearly from (a0, b0) to (a1, b1) over the run; gate = a(psi - psibar) - b psibar
    drift_center: bool = False        # drift and gradient parts in separate traces; the drift part is multiplied by
                                      # delta minus its running mean (critic bias removed), the gradient part by delta
    local_sep: float = 0.0            # > 0: local-reward updates get their own AdaBelief state, with this
                                      # fraction of the learning rate (otherwise they share the task's)
    local_baseline: bool = False      # subtract each neuron's running mean of its local reward
    homeo_c: float = 0.0              # plasticity homeostat through the drift: modulator c * (psibar - homeo_target)
    homeo_target: float = 0.1
    homeo_sep: float = 0.0            # > 0: homeostat updates get their own AdaBelief state with this share of the lr
    homeo_dscale: bool = False
    eprop: float = 0.0                # weight of a random-feedback e-prop term (actor-critic, Bellec et al. 2020):
                                      # learning signal L = Bp (pi - onehot) - c_V Bv, update delta x lambda-trace(ebar L)
    intrinsic: bool = False           # intrinsic plasticity control: per-neuron bias keeps psibar near homeo_target        # scale the homeostat by the running mean |delta| of the network
    dv_mode: str = "spike"           # |dV| term: "spike" (as implemented: the reset jump, v_th z) or "correct"
                                     # (change from the previous post-reset voltage, excluding the reset)
    local_terms: tuple = (1.0, 1.0, 1.0)   # scale of the local rewards: (spike/|dV| term, VReg, firing-rate reward)
    local_rewards: str = "trace"     # neuron-local rewards x: "p_trace" (lambda-trace of the drift part, gate
                                     # psi - psibar), "trace" (whole trace, as implemented),
                                     # "gradient" (instantaneous gradient part, gate -psibar),
                                     # "q_trace" (lambda-trace of the gradient part, gate -psibar), "off"

    @property
    def uses_twins(self): return self.twin_P != 0 or self.twin_Q != 0


VARIANTS = {
    "original": Variant("original", 1.0, 0.0),                     # gate psi (empirically found rule)
    "proposed": Variant("proposed", 1.0, -2.0),                    # gate psi - 2 psibar
    "drift_only": Variant("drift_only", 1.0, -1.0),                # gate psi - psibar
    "gradient_only": Variant("gradient_only", 0.0, -1.0),          # gate -psibar
    "reservoir": Variant("reservoir", 1.0, 0.0, hidden_lr=False),
    "frozen_readout": Variant("frozen_readout", 1.0, 0.0, readout_lr=False),
    "proposed_local_grad": Variant("proposed_local_grad", 1.0, -2.0, local_rewards="gradient"),
    "proposed_local_off": Variant("proposed_local_off", 1.0, -2.0, local_rewards="off"),
    "proposed_local_qtrace": Variant("proposed_local_qtrace", 1.0, -2.0, local_rewards="q_trace"),
    "twin_P": Variant("twin_P", 0, 0, twin_P=1.0),                 # drift only (twin-estimated)
    "twin_Q_code": Variant("twin_Q_code", 0, 0, twin_Q=1.0),       # noise part, original sign
    "twin_Q_reinforce": Variant("twin_Q_reinforce", 0, 0, twin_Q=-1.0),   # noise part, REINFORCE sign
    "twin_P_minus_Q": Variant("twin_P_minus_Q", 0, 0, twin_P=1.0, twin_Q=-1.0),
    "twin_full": Variant("twin_full", 0, 0, twin_P=1.0, twin_Q=1.0),    # = original, via twins
}


class Learner:
    """Online actor-critic training of a batch of independent networks (variants x seeds)."""

    def __init__(self, cfg: Config, variants, seeds, rng_seed=0, twins=None):
        self.cfg = cfg
        self.variants = [VARIANTS[v] if isinstance(v, str) else v for v in variants]
        self.seeds = seeds
        self.B = len(self.variants) * len(seeds)
        self.var_of = np.repeat(np.arange(len(self.variants)), len(seeds))
        self.rng = np.random.default_rng(rng_seed)
        self.env = GridForager(cfg, self.B, self.rng)
        self.lay = Layout(self.env.n_inputs, cfg.hE ** 2, cfg.hI ** 2, cfg.hF ** 2)
        built = [build_weights(cfg, self.lay, s) for s in seeds]
        si = np.arange(self.B) % len(seeds)
        self.M = np.stack([built[k][0] for k in si]); self.W = np.stack([built[k][1] for k in si])
        self.Wp = np.stack([built[k][2] for k in si]); self.Wv = np.stack([built[k][3] for k in si])
        self.W0, self.Wp0, self.Wv0 = self.W.copy(), self.Wp.copy(), self.Wv.copy()
        self.st = NeuronState(self.B, self.lay)
        V = self.variants; vo = self.var_of
        col = lambda f: np.array([f(v) for v in V], f32)[vo]
        self.c_psi, self.c_bar = col(lambda v: v.c_psi)[:, None], col(lambda v: v.c_bar)[:, None]
        sc = lambda k: col(lambda v: (v.sched[k] if v.sched else np.nan))[:, None]
        self.sched = [sc(k) for k in range(4)]
        self.has_sched = np.array([v.sched is not None for v in V])[vo][:, None]
        self.progress = 0.0
        self.c_tP, self.c_tQ = col(lambda v: v.twin_P)[:, None, None], col(lambda v: v.twin_Q)[:, None, None]
        self.lr_h = col(lambda v: cfg.lr * v.hidden_lr)[:, None, None]
        self.lr_o = col(lambda v: cfg.lr * v.readout_lr)[:, None, None]
        self.lr_tr = col(lambda v: 1.0 if v.local_rewards == "trace" else 0.0)[:, None, None]
        self.lr_gr = col(lambda v: 1.0 if v.local_rewards == "gradient" else 0.0)[:, None, None]
        self.lr_qt = col(lambda v: 1.0 if v.local_rewards == "q_trace" else 0.0)[:, None, None]
        self.lr_pt = col(lambda v: 1.0 if v.local_rewards == "p_trace" else 0.0)[:, None, None]
        self.lt = [col(lambda v, k=k: v.local_terms[k])[:, None] for k in range(3)]
        self.dv_correct = col(lambda v: 1.0 if v.dv_mode == "correct" else 0.0)[:, None]
        self.homeo_c, self.homeo_t = col(lambda v: v.homeo_c)[:, None], col(lambda v: v.homeo_target)[:, None]
        self.homeo_sep = col(lambda v: v.homeo_sep)[:, None, None]
        self.homeo_ds = col(lambda v: 1.0 if v.homeo_dscale else 0.0)[:, None]
        self.ep_w = col(lambda v: v.eprop)[:, None, None]
        self.ip = col(lambda v: 1.0 if v.intrinsic else 0.0)[:, None]
        rb = [np.random.default_rng(5000 + s) for s in seeds]
        Bp_s = [r.normal(0, 1 / np.sqrt(4), (self.lay.POST, 4)).astype(f32) for r in rb]
        Bv_s = [r.normal(0, 1.0, (self.lay.POST,)).astype(f32) for r in rb]
        self.Bp = np.stack([Bp_s[k] for k in si]); self.Bv = np.stack([Bv_s[k] for k in si])
        self.c_V = 1.0
        self.sep = col(lambda v: v.local_sep)[:, None, None]
        self.dcen = col(lambda v: 1.0 if v.drift_center else 0.0)[:, None, None]
        self.use_base = col(lambda v: 1.0 if v.local_baseline else 0.0)[:, None, None]
        need_twins = any(v.uses_twins for v in V) or twins is not None
        self.twins = (twins or StaggeredTwins(cfg, self.lay, self.B)) if need_twins else None
        L = self.lay; B = self.B; P, Q = L.PRE, L.POST
        z = lambda *s: np.zeros(s, f32)
        self.rln, self.m_h, self.s_h = z(B, P, Q), z(B, P, Q), z(B, P, Q)
        self.rlq = z(B, P, Q)            # lambda-trace of the gradient part (gate -psibar), for local rewards
        self.rlpd = z(B, P, Q)           # lambda-trace of the drift part (gate psi - psibar), for local rewards
        self.m_l, self.s_l = z(B, P, Q), z(B, P, Q)   # separate optimiser state for local rewards
        self.rP_c, self.rQ_c = z(B, P, Q), z(B, P, Q)  # split traces for drift_center variants
        self.dmean = z(B, 1)
        self.absd = z(B, 1)
        self.rle, self.ebar = z(B, P, Q), z(B, P, Q)          # e-prop trace and filtered eligibility
        self.bias, self.vA_mean = z(B, Q), z(B, Q)            # intrinsic plasticity
        self.m_hh, self.s_hh = z(B, P, Q), z(B, P, Q)   # homeostat optimiser state (homeo_sep)
        self.rbase = z(B, 1, Q)          # running mean of each neuron's local reward
        self.local_stats = {k: np.zeros(B) for k in ("vreg", "dv", "rate_err")}
        self.efilt, self.fav, self.favtr = z(B, P, Q), z(B, Q), z(B, Q)
        self.vp, self.vv, self.vprev, self.r = z(B, 4), z(B, 1), z(B, 1), z(B, 1)
        self.zb_pf, self.zb_vf = z(B, L.NF), z(B, L.NF)
        self.rlp, self.rlv = z(B, L.NF, 4), z(B, L.NF, 1)
        self.m_p, self.s_p, self.m_v, self.s_v = z(B, L.NF, 4), z(B, L.NF, 4), z(B, L.NF, 1), z(B, L.NF, 1)
        self.PG = z(B, 4)
        self.t = 0
        self.zin = z(B, L.NIN)
        self.rates = self.env.rates()
        self.hooks = []                                   # callables(learner, out, extra) per step

    def step(self):
        cfg, L, st = self.cfg, self.lay, self.st
        nz = self.rng.standard_normal((self.B, L.POST)).astype(f32)
        twin = self.twins.step(self.t, st, self.W, self.M, self.zin, nz) if self.twins else None
        v_post_reset_prev = st.v - cfg.v_th * st.z          # previous step's voltage after its reset
        out = advance(cfg, L, st, self.W, self.M, self.zin, nz, dI=self.bias if self.ip.any() else None)
        dvc = np.abs(st.v - v_post_reset_prev)               # integration change this step (pre-reset voltage)
        self.t += 1
        # readouts and TD error
        zpf, zvf = st.z[:, L.pf], st.z[:, L.vf]
        a_o = cfg.alpha
        self.vp = a_o * self.vp + np.einsum("bi,bik->bk", zpf, self.Wp)
        self.vv = a_o * self.vv + np.einsum("bi,bik->bk", zvf, self.Wv)
        self.zb_pf = a_o * self.zb_pf + zpf; self.zb_vf = a_o * self.zb_vf + zvf
        delta = cfg.gamma * self.vv + self.r - self.vprev; self.vprev = self.vv.copy(); self.r *= cfg.rdecay
        # hidden-layer input to the trace
        gate = self.c_psi * out["psi"] + self.c_bar * st.psibar
        if self.has_sched.any():
            a0, b0, a1, b1 = self.sched; f = self.progress
            a_t, b_t = a0 + f * (a1 - a0), b0 + f * (b1 - b0)
            gate = np.where(self.has_sched, a_t * (out["psi"] - st.psibar) - b_t * st.psibar, gate)
        inp = score(cfg, gate, out["pi"], out["xi"])
        extra = {}
        if twin is not None:
            o2, st2 = twin
            s_main = score(cfg, out["psi"], out["pi"], out["xi"])
            s_twin = score(cfg, o2["psi"], o2["pi"], o2["xi"])
            Pt, Qt = 0.5 * (s_main + s_twin), 0.5 * (s_main - s_twin)
            inp = inp + self.c_tP * Pt + self.c_tQ * Qt
            extra = dict(P=Pt, Q=Qt, s_twin=s_twin, twin_out=o2, twin_state=st2)
        g = delta[:, :, None] * self.rln                              # DeltaG uses the trace before update
        if self.dcen.any():
            self.dmean = 0.999 * self.dmean + 0.001 * delta
            gate_P = self.c_psi * (out["psi"] - st.psibar)                # drift part of this variant's gate
            gate_Q = (self.c_bar + self.c_psi) * st.psibar                # remainder (gradient part), e.g. -psibar
            g_c = (delta - self.dmean)[:, :, None] * self.rP_c + delta[:, :, None] * self.rQ_c
            g = np.where(self.dcen > 0, g_c, g)
            self.rP_c = cfg.lam * self.rP_c + score(cfg, gate_P, out["pi"], out["xi"])
            self.rQ_c = cfg.lam * self.rQ_c + score(cfg, gate_Q, out["pi"], out["xi"])
        if cfg.regs:
            afav = np.exp(-1 / cfg.tau_reg)
            self.fav = afav * self.fav + (1 - afav) * st.z
            fire_reg = self.fav - cfg.f_target_hz / 1000
            self.favtr = cfg.lam * self.favtr + (1 - cfg.lam) * np.abs(fire_reg)
            vreg = np.maximum(0, st.v - out["A"]) + np.maximum(0, -st.v - out["A"])
            self.efilt = cfg.alpha * self.efilt + out["psi"][:, None, :] * out["pi"]
            dv_term = self.dv_correct * dvc + (1 - self.dv_correct) * cfg.v_th * st.z
            r_loc = (self.lt[0] * (-0.01 * dv_term) + self.lt[1] * (-cfg.c_reg * vreg)
                     + self.lt[2] * cfg.c_reg * (self.favtr - np.abs(fire_reg)))[:, None, :]
            g_score = score(cfg, -st.psibar, out["pi"], out["xi"])     # instantaneous gradient part
            self.rbase = 0.999 * self.rbase + 0.001 * r_loc
            r_loc = r_loc - self.use_base * self.rbase
            g_local = r_loc * (self.lr_tr * self.rln + self.lr_gr * g_score + self.lr_qt * self.rlq + self.lr_pt * self.rlpd)
            sep = self.sep > 0
            g = g + np.where(sep, 0.0, g_local) + cfg.c_reg * fire_reg[:, None, :] * self.efilt
            if sep.any():
                adabelief(self.W, self.m_l, self.s_l, np.where(sep, g_local, 0.0) * self.M,
                          self.lr_h * self.sep, self.W0, self.t, cfg, l2=0.0)
            ls = self.local_stats
            ls["vreg"] += vreg[:, :L.N].mean(1); ls["dv"] += dvc[:, :L.N].mean(1)
            ls["rate_err"] += np.abs(fire_reg[:, :L.N]).mean(1)
            self.rlq = cfg.lam * self.rlq + g_score
        self.absd = 0.999 * self.absd + 0.001 * np.abs(delta)
        if self.ep_w.any():                                       # random-feedback e-prop (actor-critic)
            g = g + self.ep_w * delta[:, :, None] * self.rle
            Lsig = np.einsum("bqk,bk->bq", self.Bp, self.PG) - self.c_V * self.Bv
            self.ebar = cfg.alpha * self.ebar + out["psi"][:, None, :] * out["pi"]
            self.rle = cfg.lam * self.rle + self.ebar * Lsig[:, None, :]
        if self.ip.any():                                         # intrinsic plasticity: bias toward the gate range
            self.vA_mean = 0.999 * self.vA_mean + 0.001 * (st.v - out["A"])
            push = (self.homeo_t - st.psibar) * -np.sign(self.vA_mean)   # silent: raise bias, saturated: lower it
            self.bias = self.bias + self.ip * 1e-3 * push
        if self.homeo_c.any():                                   # homeostat: modulates the drift trace
            scale = np.where(self.homeo_ds > 0, self.absd, 1.0)
            g_h = (self.homeo_c * scale * (st.psibar - self.homeo_t))[:, None, :] * self.rlpd
            hs = self.homeo_sep > 0
            g = g + np.where(hs, 0.0, g_h)
            if hs.any():
                adabelief(self.W, self.m_hh, self.s_hh, np.where(hs, g_h, 0.0) * self.M,
                          self.lr_h * self.homeo_sep, self.W0, self.t, cfg, l2=0.0)
        if (self.lr_pt.any() if cfg.regs else False) or self.homeo_c.any():
            self.rlpd = cfg.lam * self.rlpd + score(cfg, out["psi"] - st.psibar, out["pi"], out["xi"])
        self.rln = cfg.lam * self.rln + inp
        adabelief(self.W, self.m_h, self.s_h, g * self.M, self.lr_h, self.W0, self.t, cfg)
        gp = delta[:, :, None] * self.rlp
        self.rlp = cfg.lam * self.rlp + self.zb_pf[:, :, None] * self.PG[:, None, :]
        gv = self.rlv * delta[:, :, None]
        self.rlv = cfg.gamma * cfg.lam * self.rlv - self.zb_vf[:, :, None]
        adabelief(self.Wp, self.m_p, self.s_p, gp, self.lr_o, self.Wp0, self.t, cfg)
        adabelief(self.Wv, self.m_v, self.s_v, gv, self.lr_o, self.Wv0, self.t, cfg)
        self.PG[:] = 0
        for h in self.hooks:
            h(self, out, dict(extra, delta=delta, inp=inp))
        return out, delta

    def act(self):
        """Decision at the end of an action window: sample, move, reward."""
        lg = self.vp - self.vp.max(1, keepdims=True); pr = np.exp(lg); pr /= pr.sum(1, keepdims=True)
        act = (self.rng.random(self.B)[:, None] > np.cumsum(pr, 1)).sum(1).clip(0, 3)
        self.PG = pr.copy(); self.PG[np.arange(self.B), act] -= 1
        rew, ate, dead = self.env.step(act)
        self.r = np.where(rew[:, None] != 0, rew[:, None], self.r)    # set_var("reward") replaces
        self.rates = self.env.rates()
        return rew, ate, dead

    def run(self, env_steps, block=2500, log=print, switch_at=None):
        """Train; returns reward per environment step, per block and batch element (blocks x B)."""
        hist, rew_sum, deaths, n = [], np.zeros(self.B), np.zeros(self.B), 0
        total = env_steps
        hz = np.zeros(self.B)
        for es in range(env_steps):
            self.progress = es / max(total - 1, 1)
            if switch_at is not None and es == switch_at:
                self.env.swap_channels = True; self.rates = self.env.rates()
            for k in range(self.cfg.K):
                self.step()
                hz += self.st.z[:, :self.lay.N].mean(1)
                if k == self.cfg.K - 1:
                    rew, ate, dead = self.act(); rew_sum += rew; deaths += dead; n += 1
                self.zin = (self.rng.random((self.B, self.lay.NIN)) < self.rates).astype(f32)
            if (es + 1) % block == 0:
                rate = rew_sum / n; hist.append(rate)
                if log:
                    parts = [f"{v.name} {rate[self.var_of == i].mean():+.3f}±{rate[self.var_of == i].std():.3f}"
                             f" ({1000 * hz[self.var_of == i].mean() / (n * self.cfg.K):.0f}Hz)"
                             for i, v in enumerate(self.variants)]
                    log(f"[{es + 1:6d}] " + "  ".join(parts))
                rew_sum[:] = 0; deaths[:] = 0; hz[:] = 0; n = 0
        return np.array(hist)

    def summary(self, hist, last=1):
        """mean and sd over seeds of the reward rate in the last `last` blocks, per variant."""
        if len(hist) == 0:
            return {v.name: (np.nan, np.nan) for v in self.variants}
        r = hist[-last:].mean(0)
        return {v.name: (r[self.var_of == i].mean(), r[self.var_of == i].std()) for i, v in enumerate(self.variants)}
