"""Backprop baseline for the two-link arm task (snake_switch.ArmEnv): PPO with a Gaussian policy over the two joint
velocities, read from the same 20x20x3 frames the spiking networks see (one frame per move, no memory).

    python arm_baseline.py OUT_DIR [--hidden 256] [--steps 500000] [--seed 0] [--sigma0 0.3]

Policy: mean = tanh(W h + b) (2 joint velocities), learned log s.d. (one per joint, clamped to [-3, 0.5]); value
v = w.h + c; h = the input (--hidden 0: LINEAR) or a shared ReLU trunk (--hidden 256 or 256,128). The action is
drawn from the Gaussian; the environment clips it to [-1, 1]. PPO: GAE (gamma, lambda), clipped ratio, value loss,
a small entropy bonus, Adam (the trunk and heads reuse rate_baseline.Model: exact gradients). Episodes end on a reach
or after max_moves (as the spiking runs). Writes OUT_DIR/episodes.csv (episode, moves, score, reached).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

WT = Path("/tmp/claude-1000/-home-user-repos-personal/443153b0-266e-4142-add4-6a0e6900d11b/scratchpad/mlg_wt")
sys.path.insert(0, str(WT / "examples" / "eprop"))
from rate_baseline import Model  # noqa: E402
import snake_switch  # noqa: E402

SHAPE = (20, 20, 3)
LOGSTD_MIN, LOGSTD_MAX = -3.0, 0.5


def logp_gauss(a, mu, logstd, nprev=None, rho=0.0):
    """log density of a (up to a constant): white Gaussian (rho 0) or, with Ornstein-Uhlenbeck exploration noise,
    the Gaussian given the previous noise nprev: mean mu + sigma rho nprev, variance sigma^2 (1 - rho^2)."""
    s = np.exp(logstd)
    c = 0.0 if not rho else s * rho * nprev
    v = s * s * (1.0 - rho * rho)
    return (-(a - mu - c) ** 2 / (2 * v) - logstd).sum(1)


def ppo_grads(model, logstd, x, a, logp_old, adv, ret, clip=0.2, c_v=0.5, c_e=1e-3, nprev=None, rho=0.0):
    """Loss and gradients of the PPO objective for one minibatch (mean over samples)."""
    z, v, cache = model.forward(x)
    mu = np.tanh(z)
    sd = np.exp(logstd)
    cshift = 0.0 if not rho else sd * rho * nprev
    s2 = sd * sd * (1.0 - rho * rho)
    logp = logp_gauss(a, mu, logstd, nprev, rho)
    r = np.exp(logp - logp_old)
    unclipped = r * adv
    clipped = np.clip(r, 1 - clip, 1 + clip) * adv
    loss = -np.minimum(unclipped, clipped).mean() + c_v * 0.5 * ((v - ret) ** 2).mean() - c_e * logstd.sum()
    active = ~(((adv >= 0) & (r > 1 + clip)) | ((adv < 0) & (r < 1 - clip)))
    dlogp = np.where(active, -adv * r, 0.0) / len(x)                    # dL / dlogp per sample
    u = a - mu - cshift
    dmu = dlogp[:, None] * u / s2
    dz = dmu * (1 - mu ** 2)
    dlogstd = (dlogp[:, None] * ((u * cshift + u * u) / s2 - 1)).sum(0) - c_e
    dv = c_v * (v - ret) / len(x)
    return loss, model.grads(cache, dz, dv), dlogstd


def gradient_check(seed=0, rho=0.0):
    """Finite-difference check of ppo_grads on a tiny model (relative error of the trunk, head and log s.d.)."""
    rng = np.random.default_rng(seed)
    npv = rng.normal(0, 1, (7, 2))
    m = Model(6, 2, [5], rng); m.p["Wv"] = rng.normal(0, 0.3, m.p["Wv"].shape)
    ls = np.array([-0.5, -1.0])
    x = rng.normal(0, 1, (7, 6)); a = rng.normal(0, 0.5, (7, 2)); adv = rng.normal(0, 1, 7); ret = rng.normal(0, 1, 7)
    z, _, _ = m.forward(x); lpo = logp_gauss(a, np.tanh(z), ls, npv, rho) + rng.normal(0, 0.05, 7)   # ratios near 1, unclipped
    _, g, gls = ppo_grads(m, ls, x, a, lpo, adv, ret, clip=10.0, nprev=npv, rho=rho)
    errs = []
    for k in ("W1", "Wp", "Wv", "bp"):
        num = np.zeros_like(m.p[k]); it = np.nditer(m.p[k], flags=["multi_index"])
        for _ in it:
            i = it.multi_index; old = m.p[k][i]
            m.p[k][i] = old + 1e-6; lp = ppo_grads(m, ls, x, a, lpo, adv, ret, clip=10.0, nprev=npv, rho=rho)[0]
            m.p[k][i] = old - 1e-6; lm = ppo_grads(m, ls, x, a, lpo, adv, ret, clip=10.0, nprev=npv, rho=rho)[0]
            m.p[k][i] = old; num[i] = (lp - lm) / 2e-6
        errs.append(np.abs(num - g[k]).max() / (np.abs(num).max() + 1e-12))
    num = np.zeros(2)
    for j in range(2):
        d = np.zeros(2); d[j] = 1e-6
        num[j] = (ppo_grads(m, ls + d, x, a, lpo, adv, ret, clip=10.0, nprev=npv, rho=rho)[0]
                  - ppo_grads(m, ls - d, x, a, lpo, adv, ret, clip=10.0, nprev=npv, rho=rho)[0]) / 2e-6
    errs.append(np.abs(num - gls).max() / (np.abs(num).max() + 1e-12))
    return max(errs)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out")
    ap.add_argument("--hidden", default="256"); ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--sigma0", type=float, default=0.3)
    ap.add_argument("--envs", type=int, default=8); ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--joint_speed", type=float, default=0.02); ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--rho", type=float, default=0.0, help="per-step correlation of the exploration noise (OU)")
    ap.add_argument("--curriculum", default=None, help='JSON ArmEnv curriculum, e.g. {} for the defaults')
    ap.add_argument("--continuous", action="store_true", help="never reset the arm (segments of --segment steps)")
    ap.add_argument("--segment", type=int, default=3000); ap.add_argument("--target_steps", type=int, default=300)
    ap.add_argument("--reward_mode", default="progress"); ap.add_argument("--sign_reward", type=float, default=0.05)
    ap.add_argument("--episode_per_target", action="store_true", help="continuous: each target is a terminal episode")
    ap.add_argument("--progress_fade", type=float, default=None)
    ap.add_argument("--kinematics", default="arm"); ap.add_argument("--view", default="allo")
    ap.add_argument("--blob", type=int, default=2)
    ap.add_argument("--coins", default=None, help='JSON CoinGazeEnv options: the coin task (4 action axes) instead')
    ap.add_argument("--shooter", default=None, help='JSON ShooterEnv options: the shooter (4 action axes) instead')
    ap.add_argument("--arena", default=None, help='JSON ArenaEnv options: coins + enemies in the dark (4 axes)')
    ap.add_argument("--save_every", type=int, default=0, help="save the policy to OUT/model.npz every N updates")
    ap.add_argument("--target_time", default=None, help='JSON ArmEnv target_time, e.g. {} for the defaults')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    hidden = [int(h) for h in a.hidden.split(",") if int(h) > 0]
    four = a.coins is not None or a.shooter is not None or a.arena is not None
    na = 4 if four else 2
    if four:
        a.continuous, a.episode_per_target = True, False    # segments are truncations; reached = coins per segment
    model = Model(int(np.prod(SHAPE)), na, hidden, rng)
    logstd = np.full(na, np.log(a.sigma0)); ms, vs, ts = np.zeros(na), np.zeros(na), 0
    import json as _json
    cur = None if a.curriculum is None else _json.loads(a.curriculum)
    envs = [snake_switch.ArenaEnv(0, SHAPE, seed=a.seed * 1000 + i, **_json.loads(a.arena))
            for i in range(a.envs)] if a.arena is not None else [snake_switch.ShooterEnv(0, SHAPE, seed=a.seed * 1000 + i, **_json.loads(a.shooter))
            for i in range(a.envs)] if a.shooter is not None else [snake_switch.CoinGazeEnv(0, SHAPE, seed=a.seed * 1000 + i, **_json.loads(a.coins))
            for i in range(a.envs)] if a.coins is not None else [snake_switch.ArmEnv(0, SHAPE, seed=a.seed * 1000 + i, joint_speed=a.joint_speed, max_steps=a.max_steps,
                                curriculum=cur, continuous=a.continuous, segment_steps=a.segment,
                                target_steps=a.target_steps, reward_mode=a.reward_mode, sign_reward=a.sign_reward,
                                episode_per_target=a.episode_per_target, progress_fade=a.progress_fade,
                                kinematics=a.kinematics, view=a.view, blob=a.blob,
                                target_time=None if a.target_time is None else _json.loads(a.target_time))
            for i in range(a.envs)]
    obs = np.stack([e.reset().ravel() for e in envs])
    ep_ret, ep_len = np.zeros(a.envs), np.zeros(a.envs, int)
    noise = np.zeros((a.envs, na))                     # OU state per arm (reset at episode ends)
    rows, steps, ep = [], 0, 0
    with open(out / "episodes.csv", "w", newline="") as f:
        csv.writer(f).writerow(["episode", "steps", "moves", "score", "reached", "level"])
    T = a.batch // a.envs
    while steps < a.steps:
        X, A, LP, R, D, V, NP = [], [], [], [], [], [], []
        for _ in range(T):
            z, v, _ = model.forward(obs)
            mu = np.tanh(z)
            nprev = noise.copy()
            noise = a.rho * noise + np.sqrt(1 - a.rho ** 2) * rng.standard_normal(mu.shape)
            act = mu + np.exp(logstd) * noise
            X.append(obs); A.append(act); LP.append(logp_gauss(act, mu, logstd, nprev, a.rho)); V.append(v)
            NP.append(nprev)
            rew, done, nxt = np.zeros(a.envs), np.zeros(a.envs), []
            for i, e in enumerate(envs):
                o, r, d = e.step(act[i])
                rew[i], done[i] = r, (d and (not a.continuous or a.episode_per_target))   # segment ends: truncations
                ep_ret[i] += r; ep_len[i] += 1
                if d and not a.continuous:
                    noise[i] = 0.0
                if d:
                    ep += 1
                    rows.append([ep, steps, int(ep_len[i]), float(ep_ret[i]),
                                 int(len(e.snake) - 1) if (a.continuous and not a.episode_per_target)
                                 else int(len(e.snake) > 1), float(e.level)])
                    ep_ret[i], ep_len[i] = 0.0, 0
                    o = e.reset()
                nxt.append(o.ravel())
            obs = np.stack(nxt); R.append(rew); D.append(done); steps += a.envs
        _, v_last, _ = model.forward(obs)
        V.append(v_last)
        R, D, V = np.array(R), np.array(D), np.array(V)
        adv = np.zeros_like(R); g = np.zeros(a.envs)
        for t in range(T - 1, -1, -1):
            delta = R[t] + a.gamma * V[t + 1] * (1 - D[t]) - V[t]
            g = delta + a.gamma * a.lam * (1 - D[t]) * g
            adv[t] = g
        ret = adv + V[:-1]
        X = np.concatenate(X); A = np.concatenate(A); LP = np.concatenate(LP); NP = np.concatenate(NP)
        adv, ret = adv.ravel(), ret.ravel()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(a.epochs):
            perm = rng.permutation(len(X))
            for s in range(0, len(X), a.minibatch):
                idx = perm[s:s + a.minibatch]
                _, gr, gls = ppo_grads(model, logstd, X[idx], A[idx], LP[idx], adv[idx], ret[idx],
                                       nprev=NP[idx], rho=a.rho)
                model.adam(gr, a.lr)
                ts += 1; ms = 0.9 * ms + 0.1 * gls; vs = 0.999 * vs + 0.001 * gls * gls
                logstd = np.clip(logstd - a.lr * (ms / (1 - 0.9 ** ts)) / (np.sqrt(vs / (1 - 0.999 ** ts)) + 1e-8),
                                 LOGSTD_MIN, LOGSTD_MAX)
        n_upd = getattr(model, "_updates", 0) + 1; model._updates = n_upd
        if a.save_every and (n_upd % a.save_every == 0 or steps >= a.steps):
            np.savez(out / "model.npz", hidden=np.array(model.hidden), logstd=logstd, steps=steps,
                     level=np.mean([e.level for e in envs]), **model.p)
        if rows:
            with open(out / "episodes.csv", "a", newline="") as f:
                csv.writer(f).writerows(rows)
            recent = rows[-200:] if len(rows) >= 200 else rows
            last = np.array([[r[3], r[4], r[2]] for r in recent])
            print(f"steps {steps:7d}  episodes {ep:5d}  score {last[:, 0].mean():+.3f}  reached {last[:, 1].mean():.2f}"
                  f"  moves {last[:, 2].mean():.1f}  sd {np.exp(logstd).round(3)}  level {np.mean([e.level for e in envs]):.2f}", flush=True)
            rows = []


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        print("max relative gradient error: white", gradient_check(), " OU rho 0.9", gradient_check(rho=0.9))
    else:
        main()
