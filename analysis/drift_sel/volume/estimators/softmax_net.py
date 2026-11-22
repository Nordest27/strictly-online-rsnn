"""A spiking network that computes a 4-action softmax in its activity (teacher), and the feedback it should relay.

Network (discrete-time LIF, dt = 1 ms, relative reset, refractory 2 steps): K action pools of N_E excitatory neurons
and one shared pool of N_I inhibitory neurons. Input logit x_k drives pool k (current b + g x_k); excitation within a
pool, all E -> I, all I -> E (shared inhibition: a soft winner-take-all). Every neuron gets Gaussian current noise.
Action = the pool with the most spikes in the decision window (ties broken at random).

Phase 1a (calibrate): P(action | x) against softmax(beta x) for random logits; beta fitted by maximum likelihood.
Phase 1b (feedback): for a few logit vectors, the exact effect of a small constant extra current on every neuron on
the action probabilities, dP/dI_i (finite differences with common random numbers), against the softmax Jacobian
prediction for its pool, pi (e_k - pi) * c.

Usage: python softmax_net.py calibrate | feedback
"""
import sys
import numpy as np

K, N_E, N_I = 4, 20, 20
N = K * N_E + N_I
ALPHA, THETA, REFRAC = float(np.exp(-1.0 / 10.0)), 1.0, 2
T, T0 = 60, 20                         # trial length, decision window start


def weights(w_ee=0.06, w_ei=0.04, w_ie=-0.12, w_ii=0.0):
    """W[post, pre]."""
    W = np.zeros((N, N))
    pool = np.repeat(np.arange(K), N_E)
    for k in range(K):
        idx = np.where(pool == k)[0]
        W[np.ix_(idx, idx)] = w_ee
    np.fill_diagonal(W, 0.0)
    E, I = slice(0, K * N_E), slice(K * N_E, N)
    W[I, E] = w_ei
    W[E, I] = w_ie
    W[I, I] = w_ii
    return W


def simulate(x, W, rng, b=0.085, g=0.01, sigma=0.12, extra=None, noise=None, drive_override=None):
    """x: (M, K) logits. extra: (M, N) constant extra current. noise: (T, M, N) or None. Returns action (M,),
    counts (M, K)."""
    M = x.shape[0]
    drive = np.zeros((M, N))
    drive[:, :K * N_E] = b + g * np.repeat(x, N_E, axis=1)
    drive[:, K * N_E:] = 0.0
    if extra is not None:
        drive = drive + extra
    if drive_override is not None:
        drive = drive_override
    V = np.zeros((M, N)); ref = np.zeros((M, N)); z = np.zeros((M, N))
    counts = np.zeros((M, K)); xisum = np.zeros((M, N))
    for t in range(T):
        xi = noise[t] if noise is not None else rng.standard_normal((M, N))
        xisum += xi
        V = ALPHA * V + drive + z @ W.T + sigma * xi
        V = np.where(ref > 0, ALPHA * V, V)          # refractory: no integration of new input... (leak only)
        z = ((V >= THETA) & (ref <= 0)).astype(float)
        V = V - THETA * z
        ref = np.where(z > 0, REFRAC, ref - 1)
        if t >= T0:
            counts += z[:, :K * N_E].reshape(M, K, N_E).sum(2)
    tie = rng.random((M, K)) * 1e-3
    simulate.last_xisum = xisum                     # per-neuron noise summed over the trial (Stein score / sigma)
    return np.argmax(counts + tie, 1), counts


def softmax(v):
    v = v - v.max(-1, keepdims=True); e = np.exp(v); return e / e.sum(-1, keepdims=True)


def calibrate(seed=0, n_x=60, M=400, **kw):
    rng = np.random.default_rng(seed); W = weights(**{k: v for k, v in kw.items() if k.startswith("w_")})
    sim_kw = {k: v for k, v in kw.items() if not k.startswith("w_")}
    xs = rng.normal(0, 1.0, (n_x, K))
    P = np.zeros((n_x, K)); rates = []
    for i, x in enumerate(xs):
        a, c = simulate(np.repeat(x[None], M, 0), W, rng, **sim_kw)
        P[i] = np.bincount(a, minlength=K) / M
        rates.append(c.sum(1).mean() / (N_E * K * (T - T0)) * 1000)
    betas = np.linspace(0.1, 20, 200)
    ll = [np.sum(P * np.log(softmax(b * xs) + 1e-12)) for b in betas]
    beta = betas[int(np.argmax(ll))]
    Q = softmax(beta * xs)
    kl = np.mean(np.sum(P * (np.log(P + 1e-9) - np.log(Q + 1e-9)), 1))
    # how much of the deviation is sampling noise: KL of a multinomial sample of Q itself
    Ps = np.array([np.bincount(rng.choice(K, M, p=q), minlength=K) / M for q in Q])
    kl_null = np.mean(np.sum(Ps * (np.log(Ps + 1e-9) - np.log(Q + 1e-9)), 1))
    return beta, kl, kl_null, float(np.mean(rates)), P, Q, xs


def feedback(seed=1, M=3000, h=0.004, n_states=3, **kw):
    rng = np.random.default_rng(seed); W = weights()
    xs = rng.normal(0, 1.0, (n_states, K))
    out = []
    for x in xs:
        noise = rng.standard_normal((T, M, N))
        tie_seed = int(rng.integers(1 << 30))
        base_rng = lambda: np.random.default_rng(tie_seed)
        a0, _ = simulate(np.repeat(x[None], M, 0), W, base_rng(), noise=noise)
        P0 = np.bincount(a0, minlength=K) / M
        J = np.zeros((N, K))
        for i in range(N):
            ex = np.zeros((M, N)); ex[:, i] = h
            ap, _ = simulate(np.repeat(x[None], M, 0), W, base_rng(), extra=ex, noise=noise)
            am, _ = simulate(np.repeat(x[None], M, 0), W, base_rng(), extra=-ex, noise=noise)
            J[i] = (np.bincount(ap, minlength=K) - np.bincount(am, minlength=K)) / (M * 2 * h)
        out.append((x, P0, J))
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "calibrate"
    if mode == "calibrate":
        for kw in (dict(), dict(sigma=0.08), dict(sigma=0.16), dict(g=0.02), dict(g=0.045), dict(w_ie=-0.2),
                   dict(w_ee=0.03), dict(w_ee=0.09)):
            beta, kl, kl0, hz, *_ = calibrate(**kw)
            print(f"{str(kw):22s} beta {beta:.2f}  KL(P || softmax) {kl:.4f}  (sampling floor {kl0:.4f})  "
                  f"E rate {hz:.1f} Hz", flush=True)
    elif mode == "feedback":
        for x, P0, J in feedback():
            print(f"logits {np.round(x, 2)}  P {np.round(P0, 3)}")
            pool = np.repeat(np.arange(K), N_E)
            for k in range(K):
                Jk = J[:K * N_E][pool == k]
                pred = P0 * ((np.arange(K) == k) - P0[k])          # softmax Jacobian column for logit k
                mean = Jk.mean(0)
                cos = mean @ pred / (np.linalg.norm(mean) * np.linalg.norm(pred) + 1e-12)
                per = [Jn @ pred / (np.linalg.norm(Jn) * np.linalg.norm(pred) + 1e-12) for Jn in Jk]
                print(f"  pool {k}: mean dP/dI {np.round(mean, 2)}  cos with pi(e_k - pi) {cos:+.2f}  "
                      f"per-neuron cos median {np.median(per):+.2f}  |J| sd/mean {np.linalg.norm(Jk, axis=1).std() / np.linalg.norm(Jk, axis=1).mean():.2f}")
            Ji = J[K * N_E:]
            print(f"  inhibitory: mean dP/dI {np.round(Ji.mean(0), 2)}  (sum {Ji.mean(0).sum():+.3f})")


if __name__ == "__main__":
    main()
