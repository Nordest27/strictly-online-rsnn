"""Toy: when can learned links (alignment rule) estimate a neuron's credit better than the neuron alone?

Receivers i: true credit s_i(t), an Ornstein-Uhlenbeck process (time constant tau, unit sd), observed only as
C_i = s_i + noise (sd sigma). Senders j: rates x_j = 1 + kappa * sum_i A_ji s_i + xi (xi sd 1); A sparse: each sender
reflects one receiver with probability `info` (else nothing). Links u_ji learn u += eta (x_j C_i - decay u); the link
signal L_i = sum_j u_ji (x_j - 1). Baseline: the receiver's own running average of C_i with the best time constant
(searched). Reported: corr(L_i, s_i) and corr(own average, s_i), median over receivers, after learning (second half).

Usage: python link_toy.py
"""
import numpy as np


def run(tau, sigma, info, kappa=1.0, n_r=20, n_s=200, T=200_000, eta=1e-4, decay=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    a = np.exp(-1.0 / tau)
    A = (rng.random((n_s, n_r)) < info / n_r * n_r * 0 + (rng.random((n_s, 1)) < info)) * 0.0
    owner = rng.integers(0, n_r, n_s); informative = rng.random(n_s) < info
    A = np.zeros((n_s, n_r)); A[np.arange(n_s)[informative], owner[informative]] = 1.0
    s = rng.standard_normal(n_r); u = np.zeros((n_s, n_r))
    S, Cs, Ls = [], [], []
    for t in range(T):
        s = a * s + np.sqrt(1 - a * a) * rng.standard_normal(n_r)
        C = s + sigma * rng.standard_normal(n_r)
        x = 1.0 + kappa * (A @ s) + rng.standard_normal(n_s)
        L = (x - 1.0) @ u
        u += eta * (np.outer(x, C) - decay * u)
        if t >= T // 2:
            S.append(s.copy()); Cs.append(C.copy()); Ls.append(L.copy())
    S, Cs, Ls = np.array(S), np.array(Cs), np.array(Ls)
    corr = lambda p, q: np.median([np.corrcoef(p[:, i], q[:, i])[0, 1] for i in range(p.shape[1])])
    best = -1.0
    for tau_avg in (1, 3, 10, 30, 100, 300, 1000, 3000):   # the receiver's own running average, best time constant
        b = np.exp(-1.0 / tau_avg); m = np.zeros(n_r); M = np.empty_like(Cs)
        for t in range(len(Cs)):
            m = b * m + (1 - b) * Cs[t]; M[t] = m
        best = max(best, corr(M, S))
    return corr(Ls, S), best


def main():
    print(f"{'tau':>6s} {'noise sd':>9s} {'informative':>12s}   links corr   own-average corr (best)")
    for tau in (30, 300, 3000):
        for sigma in (10.0, 30.0):
            for info in (0.1, 0.5):
                lc, oc = run(tau, sigma, info, T=60_000 if tau < 3000 else 120_000)
                print(f"{tau:6d} {sigma:9.0f} {info:12.1f}   {lc:+.3f}       {oc:+.3f}")


if __name__ == "__main__":
    main()
