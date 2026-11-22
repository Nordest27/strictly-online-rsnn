"""ALIF bias toy with sparse, decaying, activity-dependent rewards and causal credit.

Every K=30 steps is an action window; at a sparse subset of windows a reward event
fires with value R = tanh((y - target)/0.5), where y is a leaky readout of the
neuron's spikes (decay kappa_y) and target varies per event (+/- signs). Reward is
delivered as a decaying trace r^t = kappa_r r^{t-1} + R [event at t].
Estimators use causal credit: sum_t (r_t - b_t) * sum_{s<=t} c_s  (b_t = batch mean).
Truth: finite differences of E[sum_t r_t] with common random numbers.
Usage: python3 alif_bias_events.py N_SAMPLES [BATCH]"""
import sys, time, numpy as np
import alif_bias_realistic as ar

alpha = np.exp(-1 / 10); rho = np.exp(-1 / 300)
v_th, beta, gamma_pd, n_ref = 0.61, 0.0174, 0.3, 3
EST = ["dist_eprop", "trace_nopsi", "step_pretr", "step_psi", "exact"]
D, T, K, SIGMA, H = 10, 300, 30, np.exp(-5), 5e-4
LAM = 0.8 ** (1 / 30)
import os
PSI_W = float(os.environ.get('PSI_W', 0.61))
kappa_y, kappa_r = np.exp(-1 / 10), np.exp(-1 / 10)

rng0 = np.random.default_rng(5)
X = ar.make_inputs(rng0, D, T)
w = rng0.normal(0.04, 0.03, D)
events = {t: None for t in range(K - 1, T, K) if rng0.random() < 0.4}          # sparse windows
targets = {t: (rng0.uniform(0.3, 1.2), rng0.choice([-1.0, 1.0])) for t in events}  # (threshold, sign)

def batch(seed, B):
    rng = np.random.default_rng(seed); s2 = SIGMA**2
    Wset = np.vstack([w] + [w + s * H * np.eye(D)[d] for s in (1, -1) for d in range(D)])
    M = Wset.shape[0]; drive = Wset @ X
    eps = rng.normal(0, SIGMA, (B, T))
    v = np.zeros((M, B)); a = np.zeros((M, B)); z = np.zeros((M, B)); ref = np.zeros((M, B))
    y = np.zeros((M, B)); r = np.zeros((M, B)); Fsum = np.zeros((M, B))
    xi = np.zeros(B); ev = np.zeros(D); ea = np.zeros((B, D))
    C = {k: np.zeros((B, D)) for k in EST}; E = {k: np.zeros((B, D)) for k in EST}; open_ = 0.0
    for t in range(T):
        xin = X[:, t - 1] if t > 0 else np.zeros(D)
        a = rho * a + z; A = v_th + beta * a
        v = alpha * v + (drive[:, t - 1][:, None] if t > 0 else 0.0) - v_th * z + eps[None, :, t]
        ref = np.maximum(ref - 1, 0); refractory = ref > 0
        z = ((v >= A) & ~refractory).astype(float); ref = np.where(z > 0, n_ref, ref)
        y = kappa_y * y + z
        R = 0.0
        if t in targets:
            thr, sgn = targets[t]; R = sgn * np.tanh((y - thr) / 0.5)
        r = kappa_r * r + R; Fsum += r
        psi = np.where(refractory[0], 0.0, (gamma_pd / PSI_W) * np.maximum(0, 1 - np.abs(v[0] - A[0]) / PSI_W))
        ev = alpha * ev + xin
        open_ += (psi > 0).mean()
        e = psi[:, None] * (ev[None, :] - beta * ea)
        xi = alpha * xi + eps[:, t]
        sc_tr = (1 - alpha**2) * xi / s2; sc_st = eps[:, t] / s2
        for k in EST: C[k] *= LAM
        C["dist_eprop"] += sc_tr[:, None] * e
        C["trace_nopsi"] += sc_tr[:, None] * ev[None, :]
        C["step_pretr"] += sc_st[:, None] * ev[None, :]
        C["step_psi"] += sc_st[:, None] * e
        C["exact"] += sc_st[:, None] * xin[None, :]
        ea = psi[:, None] * ev[None, :] + (rho - psi[:, None] * beta) * ea
        rc = r[0] - r[0].mean()                                        # causal pairing with baseline
        for k in EST: E[k] += rc[:, None] * C[k]
    fd = ((Fsum[1:1 + D] - Fsum[1 + D:]) / (2 * H)).sum(1)
    return fd, {k: E[k].sum(0) for k in EST}, {k: (E[k]**2).sum(0) for k in EST}, open_ / T

if __name__ == "__main__":
    N = int(float(sys.argv[1])); B = int(sys.argv[2]) if len(sys.argv) > 2 else 25000
    nb = max(N // B, 2); cos = lambda p, q: p @ q / np.linalg.norm(p) / np.linalg.norm(q)
    acc = [dict(fd=np.zeros(D), m={k: np.zeros(D) for k in EST}, sq={k: np.zeros(D) for k in EST}, n=0) for _ in range(2)]
    t0 = time.time()
    for i in range(nb):
        fd, m, sq, op = batch(3000 + i, B); h = acc[i % 2]; OPEN = op
        h["fd"] += fd; h["n"] += B
        for k in EST: h["m"][k] += m[k]; h["sq"][k] += sq[k]
        if i == 0: print(f"{len(events)} reward events in {T} steps; first batch {time.time()-t0:.1f}s -> est. {nb*(time.time()-t0)/60:.1f} min", flush=True)
    n = acc[0]["n"] + acc[1]["n"]; truth = (acc[0]["fd"] + acc[1]["fd"]) / n
    fA, fB = acc[0]["fd"] / acc[0]["n"], acc[1]["fd"] / acc[1]["n"]
    print(f"\n== psi width={PSI_W}: sparse decaying reward, lambda-trace: sigma={SIGMA:.4f}, N={n:,}, {time.time()-t0:.0f}s")
    print(f"   truth halves agree: cos={cos(fA, fB):.3f}")
    for k in EST:
        mk = (acc[0]["m"][k] + acc[1]["m"][k]) / n; var = (acc[0]["sq"][k] + acc[1]["sq"][k]) / n - mk**2
        cA = cos(acc[0]["m"][k] / acc[0]["n"], fA); cB = cos(acc[1]["m"][k] / acc[1]["n"], fB)
        sd = np.sqrt(var.sum()); sig = mk @ truth / np.linalg.norm(truth)
        print(f"   {k:12s} cos={cos(mk, truth):6.3f} (halves {cA:.3f}/{cB:.3f})  abs.spread={sd:10.1f}  |mean|={np.linalg.norm(mk):9.1f}  signal along truth={sig:9.1f}"
              f"  rel.noise={sd/np.linalg.norm(mk):5.2f}")
    print(f"   psi open on {100*OPEN:.1f}% of steps")
