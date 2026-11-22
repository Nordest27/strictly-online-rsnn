"""Which local signals estimate the teacher's true feedback, and how many trials do they need?
True credit of an excitatory neuron in pool k for reward vector r (from the measured Jacobians: the softmax column of
its pool): c_k = pi_k (r_k - pi . r). Per trial, delta = r(a) - pi . r (oracle baseline). Estimators, averaged over n
trials at a fixed state, compared by cosine with c across the 80 excitatory neurons:
  own       : delta * sum_t xi_i(t)                          (node perturbation, unbiased)
  pool      : own averaged over the neuron's pool             (relayed by the pool's own excitation)
  win_true  : delta * (1[a = k] - pi_k)                       (REINFORCE with the true probabilities; oracle)
  win_share : delta * (s_k - mean s_k), s_k = pool k's spike share in the window, mean over the n trials (local)
"""
import numpy as np, softmax_net as S

rng = np.random.default_rng(5); W = S.weights(); K, NE = S.K, S.N_E
pool = np.repeat(np.arange(K), NE)
res = {k: {n: [] for n in (10, 100, 1000)} for k in ("own", "pool", "win_true", "win_share")}
for st in range(12):
    x = rng.normal(0, 1, K); r = np.eye(K)[rng.integers(K)]
    M = 4000
    a, cnt = S.simulate(np.repeat(x[None], M, 0), W, rng); xs = S.simulate.last_xisum[:, :K * NE]
    P = np.bincount(a, minlength=K) / M                       # the state's probabilities (large-sample)
    c_true = (P * (r - P @ r))[pool]
    delta = r[a] - P @ r
    share = cnt / np.maximum(cnt.sum(1, keepdims=True), 1)
    for n in (10, 100, 1000):
        for rep in range(20):
            idx = rng.choice(M, n, replace=False); d = delta[idx][:, None]
            est = {"own": (d * xs[idx]).mean(0),
                   "pool": np.repeat((d * xs[idx]).mean(0).reshape(K, NE).mean(1), NE),
                   "win_true": np.repeat((d * (np.eye(K)[a[idx]] - P)).mean(0), NE),
                   "win_share": np.repeat((d * (share[idx] - share[idx].mean(0))).mean(0), NE)}
            for k, e in est.items():
                res[k][n].append(e @ c_true / (np.linalg.norm(e) * np.linalg.norm(c_true) + 1e-12))
print("cosine with the true feedback (80 excitatory neurons), mean over 12 states x 20 repeats")
print(f"{'signal':10s}" + "".join(f"   n={n:<5d}" for n in (10, 100, 1000)))
for k in res:
    print(f"{k:10s}" + "".join(f"   {np.mean(res[k][n]):+.3f}  " for n in (10, 100, 1000)))
