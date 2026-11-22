"""Experiment 16: plasticity after a task switch. Forager for 30k environment steps; at 15k the food and wall
input channels are swapped, so the hidden features must change. One run, five seeds:
  original, proposed, proposed + homeostat (|delta|-scaled, c = 5), gradient_only + homeostat (same)
Reports the reward rate per 2500-step block; the switch is after block 6.
Runtime ~60 min.  Usage: python3 exp16_task_switch.py [env_steps]
"""
import sys, time
import numpy as np
from core import Config, Learner, Variant

H = dict(homeo_c=5.0, homeo_dscale=True, local_rewards="off")
VARIANTS = [Variant("original", 1.0, 0.0, local_rewards="off"), Variant("proposed", 1.0, -2.0, local_rewards="off"),
            Variant("proposed+homeo", 1.0, -2.0, **H), Variant("gradient+homeo", 0.0, -1.0, **H)]


def main(env_steps=30000):
    L = Learner(Config(), VARIANTS, seeds=[0, 1, 2, 3, 4], rng_seed=0)
    t0 = time.time()
    hist = L.run(env_steps, block=2500, switch_at=env_steps // 2)
    print(f"\nreward rate per 2500-step block, mean over 5 seeds (switch after block {env_steps // 2 // 2500})"
          f"  [{time.time() - t0:.0f}s]")
    for i, v in enumerate(VARIANTS):
        sel = L.var_of == i
        print(f"   {v.name:16s} " + " ".join(f"{h[sel].mean():+.3f}" for h in hist))
    sw = env_steps // 2 // 2500
    print("\nbefore switch (last block) / right after (first block) / end:")
    for i, v in enumerate(VARIANTS):
        sel = L.var_of == i
        print(f"   {v.name:16s} {hist[sw - 1][sel].mean():+.3f} / {hist[sw][sel].mean():+.3f} / "
              f"{hist[-1][sel].mean():+.3f} ± {hist[-1][sel].std():.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30000)
