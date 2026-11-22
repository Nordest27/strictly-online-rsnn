"""Shared machinery for the experiments: trained snapshots and controlled noise rollouts from a fixed
state, with per-synapse traces tracked only for a set of target synapses (so that hundreds of
thousands of rollouts fit in memory)."""
from dataclasses import dataclass
import numpy as np
from core import Config, Learner, Layout, gate_psi, f32


@dataclass
class Snapshot:
    lay: Layout
    W: np.ndarray        # (PRE, POST)
    M: np.ndarray
    Wv: np.ndarray       # (NF, 1)
    rates: np.ndarray    # (NIN,) input rates of the current observation
    v: np.ndarray; a: np.ndarray; z: np.ndarray; ref: np.ndarray; zbar: np.ndarray; epsA: np.ndarray
    vv: float


def trained_snapshot(cfg: Config, train_steps=10000, seed=0, variant="proposed", warm=400, log=None):
    """Train one network with `variant` for train_steps environment steps, then freeze the weights and
    run `warm` further steps with a fixed observation to obtain a state."""
    L = Learner(cfg, [variant], [seed], rng_seed=seed)
    if train_steps:
        L.run(train_steps, block=max(train_steps // 4, 1), log=log)
    lay = L.lay
    W, M, Wv = L.W[0].copy(), L.M[0].copy(), L.Wv[0].copy()
    rates = L.rates[0].copy()
    rng = np.random.default_rng(100 + seed)
    st = _Light(1, lay); epsA = np.zeros((lay.PRE, lay.POST), f32); vv = np.zeros(1, f32)
    zin = np.zeros((1, lay.NIN), f32)
    for _ in range(warm):
        o = advance_light(cfg, lay, st, W, M, zin, rng.standard_normal((1, lay.POST)).astype(f32))
        epsA = o["psi"][0][None, :] * st.zbar[0][:, None] + (cfg.rho - o["psi"][0][None, :] * cfg.beta) * epsA
        vv = cfg.alpha * vv + st.z[0, lay.vf] @ Wv
        zin = (rng.random((1, lay.NIN)) < rates).astype(f32)
    return Snapshot(lay, W, M, Wv, rates, st.v[0], st.a[0], st.z[0], st.ref[0], st.zbar[0], epsA, float(vv[0]))


class _Light:
    def __init__(self, B, lay):
        self.v, self.a, self.z = (np.zeros((B, lay.POST), f32) for _ in range(3))
        self.ref = np.zeros((B, lay.POST), np.int32); self.zbar = np.zeros((B, lay.PRE), f32)


def advance_light(cfg, lay, st, W, M, zin, nz, dI=None):
    """Neuron dynamics only (same order as core.advance), returns zpre, psi, A, refr."""
    zpre = np.concatenate([zin, st.z[:, :lay.N]], 1)
    I = zpre @ W + cfg.sigma * nz * (zpre @ M)
    if dI is not None:
        I = I + dI
    st.a = cfg.rho * st.a + st.z
    A = cfg.v_th + cfg.beta * st.a
    st.v = cfg.alpha * st.v + I - cfg.v_th * st.z
    st.ref = np.maximum(st.ref - 1, 0); refr = st.ref > 0
    z_new = ((st.v >= A) & ~refr).astype(f32)
    st.ref = np.where(z_new > 0, cfg.n_ref, st.ref)
    psi = gate_psi(cfg, st.v, A, refr)
    st.zbar = cfg.alpha * st.zbar + zpre
    st.z = z_new
    return dict(zpre=zpre, psi=psi, A=A, refr=refr)


def pick_targets(snap: Snapshot, S, rng, min_pre=0.3, post=None):
    ii, jj = np.nonzero(snap.M)
    ok = snap.zbar[ii] > min_pre
    if post is not None:
        ok &= np.isin(jj, post)
    ii, jj = ii[ok], jj[ok]
    pick = rng.choice(len(ii), min(S, len(ii)), replace=False)
    return ii[pick], jj[pick]


def snapshot_rollouts(cfg: Config, snap: Snapshot, R=3000, S=40, T=30, noise_fd=True, weight_fd=True,
                      h_w=0.01, h_n=0.3, seed=1, noise_mask=None, targets=None, record=False):
    """Noise rollouts from the snapshot with a fixed input spike train. Returns, for S target synapses:
      A[t, s]   = E[score] (kappa omitted: a constant), A_se its standard error
      B[s]      = (1/sigma) d/dh E[e_T(n + h v)], v_{j,T-k} = alpha^k z_{i,T-k-1}  (Stein: B = A[T-1])
      C[t, s]   = dE[e_t]/dW_s (weight finite difference)
      Ebar[t,s] = E[e_t]
    Common random numbers across all groups. noise_mask: (POST,) bool, which neurons receive noise."""
    lay = snap.lay
    rng = np.random.default_rng(seed)
    ti, tj = targets if targets is not None else pick_targets(snap, S, rng)
    S = len(ti)
    ZIN = (rng.random((T, lay.NIN)) < snap.rates).astype(f32)
    groups = [("base", None, 0.0)]
    if weight_fd:
        groups += [("w", s, +h_w) for s in range(S)] + [("w", s, -h_w) for s in range(S)]
    if noise_fd:
        groups += [("n", s, +h_n) for s in range(S)] + [("n", s, -h_n) for s in range(S)]
    G = len(groups); B = G * R
    gk = np.array([0 if g[0] == "base" else (1 if g[0] == "w" else 2) for g in groups])
    gs = np.array([-1 if g[1] is None else g[1] for g in groups]); gh = np.array([g[2] for g in groups], f32)
    grp = np.repeat(np.arange(G), R); rid = np.tile(np.arange(R), G)
    kind, tgt, hval = gk[grp], gs[grp], gh[grp]
    NOISE = np.random.default_rng(seed + 6).standard_normal((T, R, lay.POST)).astype(f32)
    if noise_mask is not None:
        NOISE *= noise_mask[None, None, :]
    st = _Light(B, lay)
    st.v[:] = snap.v; st.a[:] = snap.a; st.z[:] = snap.z; st.ref[:] = snap.ref; st.zbar[:] = snap.zbar
    epsS = np.repeat(snap.epsA[ti, tj][None], B, 0); xiS = np.zeros((B, S), f32)
    base = grp == 0
    SC = np.zeros((T, R, S), f32); Ebar = np.zeros((T, S)); Ep = np.zeros((T, S)); Em = np.zeros((T, S))
    Np = np.zeros(S); Nm = np.zeros(S)
    rec = {k: np.zeros((T, R, S), f32) for k in ("e", "psi", "pi", "z", "refr", "vA")} if record else None
    gidx = [np.nonzero(grp == g)[0] for g in range(G)]
    bw = np.nonzero(kind == 1)[0]; bn = np.nonzero(kind == 2)[0]
    for t in range(T):
        zin = np.broadcast_to(ZIN[t], (B, lay.NIN))
        zpre_now = np.concatenate([zin, st.z[:, :lay.N]], 1)
        nz = NOISE[t][rid].copy()
        if len(bn):
            s_ = tgt[bn]
            nz[bn, tj[s_]] += hval[bn] * cfg.alpha ** (T - 1 - t) * zpre_now[bn, ti[s_]]
        dI = None
        if len(bw):
            dI = np.zeros((B, lay.POST), f32)
            s_ = tgt[bw]
            dI[bw, tj[s_]] = hval[bw] * zpre_now[bw, ti[s_]]
        o = advance_light(cfg, lay, st, snap.W, snap.M, zin, nz, dI)
        ps = o["psi"][:, tj]; zb = st.zbar[:, ti]
        pi = zb - cfg.beta * epsS
        e = ps * pi
        epsS = ps * zb + (cfg.rho - ps * cfg.beta) * epsS
        xiS = xiS + cfg.sigma * nz[:, tj] * o["zpre"][:, ti]
        SC[t] = (e * xiS / cfg.sigma ** 2)[base]
        Ebar[t] = e[base].mean(0)
        if record:
            for k, val in (("e", e), ("psi", ps), ("pi", pi), ("z", st.z[:, tj]), ("refr", o["refr"][:, tj]),
                           ("vA", (st.v - o["A"])[:, tj])):
                rec[k][t] = val[base]
        for s in range(S):
            if weight_fd:
                Ep[t, s] = e[gidx[1 + s], s].mean(); Em[t, s] = e[gidx[1 + S + s], s].mean()
            if noise_fd and t == T - 1:
                off = 1 + (2 * S if weight_fd else 0)
                Np[s] = e[gidx[off + s], s].mean(); Nm[s] = e[gidx[off + S + s], s].mean()
        xiS *= cfg.alpha
    out = dict(A=SC.mean(1), A_se=SC.std(1) / np.sqrt(R), Ebar=Ebar, ti=ti, tj=tj, scores=SC)
    if weight_fd:
        out["C"] = (Ep - Em) / (2 * h_w)
    if noise_fd:
        out["B"] = (Np - Nm) / (2 * h_n) / cfg.sigma
    if record:
        out["rec"] = rec
    return out


def corr_slope(x, y):
    x, y = np.ravel(x), np.ravel(y)
    return np.corrcoef(x, y)[0, 1], (x * y).sum() / (x * x).sum()
