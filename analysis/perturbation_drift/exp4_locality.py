"""Experiment 4 (Appendix D): estimating the drift needs only a neuron's own noise.

From a trained network, three target neurons always receive noise, and a growing fraction of the
other neurons receives noise too. For each fraction we measure, over noise rollouts from a fixed state:
  drift estimate      signal-to-noise ratio of the window-summed score of the targets' afferent
                      synapses (its mean is the drift)
  perturbation grad.  signal-to-noise ratio of the REINFORCE estimate V_T * eta_j for the same neurons
                      (eta_j: the neuron's accumulated noise, V_T: value readout at the end of the window)
Targets are chosen by a pilot run as the three neurons whose noise most detectably moves V_T and whose
gate is open. Runtime ~10 min.  Usage: python3 exp4_locality.py
"""
import numpy as np
from core import Config, f32
from experiments_lib import trained_snapshot, _Light, advance_light, pick_targets

T = 30


def rollout(cfg, snap, ZIN, mask, R, seed, ti=None, tj=None):
    lay = snap.lay
    NO = np.random.default_rng(seed).standard_normal((T, R, lay.POST)).astype(f32) * mask[None, None, :]
    st = _Light(R, lay)
    st.v[:] = snap.v; st.a[:] = snap.a; st.z[:] = snap.z; st.ref[:] = snap.ref; st.zbar[:] = snap.zbar
    vv = np.full(R, snap.vv, f32); eta = np.zeros((R, lay.POST), f32); psum = np.zeros((R, lay.POST), f32)
    if ti is not None:
        epsS = np.repeat(snap.epsA[ti, tj][None], R, 0); xiS = np.zeros((R, len(ti)), f32); ssum = np.zeros_like(xiS)
    for t in range(T):
        zin = np.broadcast_to(ZIN[t], (R, lay.NIN))
        o = advance_light(cfg, lay, st, snap.W, snap.M, zin, NO[t])
        c = o["zpre"] @ snap.M
        eta = cfg.alpha * eta + NO[t] * c
        psum += o["psi"]
        vv = cfg.alpha * vv + st.z[:, lay.vf] @ snap.Wv[:, 0]
        if ti is not None:
            ps = o["psi"][:, tj]; zb = st.zbar[:, ti]
            e = ps * (zb - cfg.beta * epsS)
            epsS = ps * zb + (cfg.rho - ps * cfg.beta) * epsS
            xiS = xiS + cfg.sigma * NO[t][:, tj] * o["zpre"][:, ti]
            if t >= 5:
                ssum += e * xiS / cfg.sigma ** 2
            xiS *= cfg.alpha
    return vv, eta, psum / T, (ssum if ti is not None else None)


def snr(x):
    return np.median(np.abs(x.mean(0)) / (x.std(0) + 1e-30))


def one_network(cfg, seed):
    snap = trained_snapshot(cfg, train_steps=10000, seed=seed, variant="proposed")
    lay = snap.lay
    rng = np.random.default_rng(1 + seed)
    ZIN = (rng.random((T, lay.NIN)) < snap.rates).astype(f32)
    VT, eta, pm, _ = rollout(cfg, snap, ZIN, np.ones(lay.POST, bool), 3000, 11 + seed)
    g = VT[:, None] * eta
    pilot = np.abs(g.mean(0)) / (g.std(0) + 1e-30) * (pm.mean(0) > 0.02)
    top = np.argsort(-pilot)[:3]
    ti, tj = pick_targets(snap, 60, rng, min_pre=0.05, post=top)
    others = np.setdiff1d(np.arange(lay.POST), top)
    order = np.random.default_rng(3 + seed).permutation(others)
    rows = []
    for frac in (0.0, 0.25, 0.5, 1.0):
        mask = np.zeros(lay.POST, bool); mask[top] = True; mask[order[:int(round(frac * len(order)))]] = True
        _, _, _, ssum = rollout(cfg, snap, ZIN, mask, 8000, 7 + seed, ti, tj)
        VT, eta, _, _ = rollout(cfg, snap, ZIN, mask, 20000, 21 + seed)
        rows.append((mask.sum(), snr(ssum), snr(VT[:, None] * eta[:, top])))
    return top, len(ti), rows


def main():
    cfg = Config()
    ratios = []
    for seed in (0, 1, 2):
        top, n_syn, rows = one_network(cfg, seed)
        print(f"network {seed}: target neurons {top}, {n_syn} afferent synapses")
        print(f"   {'neurons perturbed':>18s} | {'SNR drift estimate':>18s} | {'SNR perturbation gradient':>26s}")
        for n, sd, sg in rows:
            print(f"   {n:18d} | {sd:18.3f} | {sg:26.4f}")
        ratios.append((rows[-1][1] / rows[0][1], rows[-1][2] / rows[0][2]))
    r = np.array(ratios)
    print(f"\nSNR with all neurons perturbed / SNR with only the targets (mean over 3 networks):")
    print(f"   drift estimate        {r[:, 0].mean():.2f}  (per network {np.round(r[:, 0], 2)})")
    print(f"   perturbation gradient {r[:, 1].mean():.2f}  (per network {np.round(r[:, 1], 2)})")
    print(f"(resolution: drift {1 / np.sqrt(8000):.3f}, gradient {1 / np.sqrt(20000):.4f})")


if __name__ == "__main__":
    main()
