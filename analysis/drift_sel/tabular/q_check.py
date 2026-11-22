import sys, random
import numpy as np
sys.path.insert(0, ".")
from tabular_snake import make_env, key


def train_eval(gamma, eps_end, moves=1_000_000, seed=0, a=0.1):
    rng = np.random.default_rng(seed); random.seed(seed); np.random.seed(seed)
    env = make_env(); Q = {}
    q = lambda k: Q.setdefault(k, np.zeros(4))
    obs = env.reset(); k = key(obs)
    for m in range(moves):
        eps = max(eps_end, 1.0 - m / (0.5 * moves)) if eps_end < 0.1 else eps_end
        act = int(rng.integers(4)) if rng.random() < eps else int(np.argmax(q(k) + 1e-9 * rng.random(4)))
        obs, r, done = env.step(act); k2 = key(obs)
        q(k)[act] += a * (r + (0.0 if done else gamma * q(k2).max()) - q(k)[act])
        k = k2
        if done:
            obs = env.reset(); k = key(obs)
    res = []                                   # greedy evaluation, no learning
    for _ in range(500):
        obs = env.reset(); done = False; apples = 0
        while not done:
            obs, r, done = env.step(int(np.argmax(q(key(obs)) + 1e-9 * rng.random(4)))); apples += r > 0
        res.append(apples)
    return np.mean(res), np.max(res), len(Q)


for gamma, eps in ((0.5, 0.1), (0.5, 0.01), (0.9, 0.01), (0.97, 0.01)):
    m, mx, n = train_eval(gamma, eps)
    print(f"gamma {gamma} (per move), epsilon {eps}{' (decayed from 1)' if eps < 0.1 else ''}: greedy policy {m:.2f} apples per episode (best {mx}), {n} states", flush=True)
