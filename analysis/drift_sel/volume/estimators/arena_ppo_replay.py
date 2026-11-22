"""Replay a saved arena PPO policy (arm_baseline.py --arena ... --save_every N writes OUT/model.npz) as a GIF.

    python arena_ppo_replay.py OUT/model.npz -o replay.gif [--level 1.0] [--steps 900] [--stride 3] [--sample]

The policy plays snake_switch.ArenaEnv at a fixed curriculum level (default: the hardest, 1.0); every frame shows
the whole world from above (left: body blue, gaze point white, lit discs grey, coins yellow, enemies red, shots
light blue; dim = in the dark, unseen by the agent) and the agent's own 20 x 20 input (right), with the running
coins, kills and hits. --sample draws actions from the policy (with its exploration noise); default: the mean action.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

WT = Path("/tmp/claude-1000/-home-user-repos-personal/443153b0-266e-4142-add4-6a0e6900d11b/scratchpad/mlg_wt")
sys.path.insert(0, str(WT / "examples" / "eprop"))
import snake_switch  # noqa: E402

SHAPE = (20, 20, 3)


def load(path):
    z = np.load(path)
    p = {k: z[k] for k in z.files if k not in ("hidden", "logstd", "steps", "level")}
    return p, [int(h) for h in np.atleast_1d(z["hidden"])], z["logstd"], int(z["steps"]), float(z["level"])


def policy_mean(p, hidden, x):
    h = x
    for i in range(1, len(hidden) + 1):
        h = np.maximum(h @ p[f"W{i}"] + p[f"b{i}"], 0.0)
    return np.tanh(h @ p["Wp"] + p["bp"])


def frame(env, scale=12):
    view = np.kron((np.clip(env.get_local_img_observation(), 0, 1) * 255).astype(np.uint8),
                   np.ones((scale, scale, 1), np.uint8))
    h = view.shape[0]
    k = max(1, h // 80)
    world = np.kron(env.world_img(h // k), np.ones((k, k, 1), np.uint8))
    world = np.pad(world, ((0, h - world.shape[0]), (0, h - world.shape[1]), (0, 0)))
    return np.concatenate([world, np.full((h, 6, 3), 255, np.uint8), view], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model"); ap.add_argument("-o", "--out", default="arena_replay.gif")
    ap.add_argument("--level", type=float, default=1.0); ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--stride", type=int, default=3); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sample", action="store_true"); ap.add_argument("--fps", type=int, default=20)
    a = ap.parse_args()
    from PIL import Image, ImageDraw
    p, hidden, logstd, steps, trained_level = load(a.model)
    env = snake_switch.ArenaEnv(0, SHAPE, seed=a.seed, curriculum={"step": 0.0, "up": 10 ** 9, "down": -10 ** 9},
                                segment_steps=10 ** 9)
    env.level = a.level
    env._apply_level()
    obs = env.reset()
    rng = np.random.default_rng(a.seed)
    noise = np.zeros(4)
    frames = []
    for t in range(a.steps):
        act = policy_mean(p, hidden, obs.ravel()[None])[0]
        if a.sample:
            noise = 0.95 * noise + np.sqrt(1 - 0.95 ** 2) * rng.standard_normal(4)
            act = act + np.exp(logstd) * noise
        obs, _, _ = env.step(act)
        if t % a.stride == 0:
            im = Image.fromarray(frame(env))
            d = ImageDraw.Draw(im)
            d.text((4, 2), f"step {t}  coins {env.n_got}  kills {env.kills}  hits {env.hits}  level {env.level:.1f}",
                   fill=(255, 255, 255))
            frames.append(im)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=int(1000 / a.fps), loop=0)
    print(f"wrote {a.out}: {len(frames)} frames; policy trained {steps} steps (training level {trained_level:.2f});"
          f" played at level {a.level}: coins {env.n_got}, kills {env.kills}, hits {env.hits}")


if __name__ == "__main__":
    main()
