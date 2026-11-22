"""No-generalization baselines for Snake: tabular learners over the exact observations the network sees.

The environment is snake.py's SnakeEnv with the headless runs' settings (board 5, egocentric view of range 5, scale
4, 20x20x3 frames, unbounded camera), stepped once per move (wait_inc 0). Every distinct observation (the frame,
rounded to 3 decimals and hashed) is its own table entry: no similarity between observations is used ("perfect
hashing"), so whatever a tabular learner achieves needs no generalization. Rewards as the env: +1 per apple, -1 for
dying or starving. gamma 0.5 per move (the runs' gamma_env).

Learners, both strictly online (one update per move, as the network):
  - "actor_critic": a softmax policy table H[s, a] and a value table V[s]; delta = r + gamma V(s') - V(s),
    V(s) += a_v delta, H[s, a] += a_p delta (1[a] - pi(s)): the tabular version of TD-driven selection;
  - "q": tabular Q-learning, epsilon-greedy (epsilon 0.1), Q(s, a) += a (r + gamma max Q(s') - Q(s, a)): the
    strongest standard learner without generalization.
Reported per block of moves: mean apples per episode (the network runs' score + 1), distinct observations seen so far.

Usage: python tabular_snake.py [--moves 1000000] [--learner both] [--seed 0]
"""
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, "/home/user/repos/ml_genn/examples/eprop")


def make_env():
    import reservoir_search as rs
    SnakeEnv = rs.snake_env_class()
    return SnakeEnv(size=5, visible_range=5, wait_inc=0, scale=4, inp_shape=(20, 20, 3), bounded_camera=False)


def key(obs):
    return np.round(np.asarray(obs, dtype=np.float32), 3).tobytes()


def run(learner, moves, seed, a_p=0.1, a_v=0.1, a_q=0.1, eps=0.1, gamma=0.5, blocks=20):
    import random
    rng = np.random.default_rng(seed); random.seed(seed); np.random.seed(seed)
    env = make_env()
    H, V, Q, index = [], [], [], {}

    def sid(obs):
        k = key(obs)
        if k not in index:
            index[k] = len(index); H.append(np.zeros(4)); V.append(0.0); Q.append(np.zeros(4))
        return index[k]
    obs = env.reset(); s = sid(obs)
    apples, ep_apples, block_apples, block_eps = 0, [], [], []
    per_block = moves // blocks
    t0 = time.time()
    for m in range(1, moves + 1):
        if learner == "actor_critic":
            p = np.exp(H[s] - H[s].max()); p /= p.sum()
            a = int(rng.choice(4, p=p))
        else:
            a = int(rng.integers(4)) if rng.random() < eps else int(np.argmax(Q[s] + 1e-9 * rng.random(4)))
        obs, r, done = env.step(a)
        if r > 0:
            apples += 1
        s2 = sid(obs)
        if learner == "actor_critic":
            delta = r + (0.0 if done else gamma * V[s2]) - V[s]
            V[s] += a_v * delta
            g = -p; g[a] += 1.0
            H[s] += a_p * delta * g
        else:
            target = r + (0.0 if done else gamma * Q[s2].max())
            Q[s][a] += a_q * (target - Q[s][a])
        s = s2
        if done:
            ep_apples.append(apples); apples = 0
            obs = env.reset(); s = sid(obs)
        if m % per_block == 0:
            start = sum(block_eps)                       # the episodes that ended within this block
            block = ep_apples[start:]
            block_apples.append(float(np.mean(block)) if block else float("nan"))
            block_eps.append(len(block))
            print(f"  {learner:12s} moves {m:8d}: mean apples per episode in this block {block_apples[-1]:.2f} "
                  f"({block_eps[-1]} episodes), distinct observations {len(index)}, {time.time() - t0:.0f}s", flush=True)
    return block_apples, len(index)


def run_ga(moves, seed, pop=20, elite=5, eval_episodes=10, mut=0.05, blocks=20):
    """Genetic algorithm on per-observation action tables (the hashed states, as above): a genome maps every
    observation to an action (unseen observations get a fixed random default from a per-genome seed). Each generation
    every genome plays eval_episodes episodes; the elite are kept and the rest replaced by mutated copies of elite
    genomes (each entry the parent has used is resampled with probability mut). The budget is counted in moves (all
    evaluation moves count), as for the online learners. Reported per block of moves: mean apples per episode of the
    current generation's best genome and of the whole population."""
    import random
    random.seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    env = make_env()
    genomes = [dict() for _ in range(pop)]
    used = 0; next_report = moves // blocks; gen = 0
    t0 = time.time()

    def act(g, k):
        if k not in g:
            g[k] = int(rng.integers(4))
        return g[k]
    while used < moves:
        fits = []
        for g in genomes:
            tot = 0
            for _ in range(eval_episodes):
                obs = env.reset(); done = False; apples = 0
                while not done:
                    obs, r, done = env.step(act(g, key(obs))); used += 1
                    apples += r > 0
                tot += apples
            fits.append(tot / eval_episodes)
        order = np.argsort(fits)[::-1]
        elite_g = [genomes[i] for i in order[:elite]]
        new = [dict(g) for g in elite_g]
        while len(new) < pop:
            child = dict(elite_g[int(rng.integers(elite))])
            for k in child:
                if rng.random() < mut:
                    child[k] = int(rng.integers(4))
            new.append(child)
        genomes = new; gen += 1
        if used >= next_report:
            print(f"  ga           moves {used:8d} (generation {gen}): best genome {fits[order[0]]:.2f}, population mean "
                  f"{np.mean(fits):.2f} apples per episode, states in the best genome {len(elite_g[0])}, "
                  f"{time.time() - t0:.0f}s", flush=True)
            next_report += moves // blocks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--moves", type=int, default=1_000_000)
    ap.add_argument("--learner", default="both", choices=["both", "actor_critic", "q", "ga"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.learner == "ga":
        run_ga(a.moves, a.seed)
    for L in (["actor_critic", "q"] if a.learner == "both" else [a.learner] if a.learner != "ga" else []):
        run(L, a.moves, a.seed)
