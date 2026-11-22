"""Follow-up of the volume search (13 Nov 2026).

1. Selection, 2M steps, paired seeds 0-2: v031 vs the seed genome vs the seed genome with half input weights
   (drive control: is v031's gain the lower input drive or input confined to the front slab?).
2. Snake on CPU with the v031 volume (same rule/recipe as the selection runs, Snake env, lr 1e-5).
At most 4 simulations at once (das.wait_slot), Snake started first.

Usage: python v031_followup.py OUT_DIR [--snake-steps 10000000]
"""
import json, sys, copy, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "search")); sys.path.insert(0, str(HERE.parent))
import volume_search as vs  # noqa: E402
import drift_arch_search as das  # noqa: E402
from slab_compare import genome  # noqa: E402

SLOTS = 5   # wait_slot also counts the thermal guard (its --match argument names run_headless.py): 4 sims


def selection_cfg(variant, seed, steps):
    g = genome("v031" if variant == "v031" else "seed")
    name = f"{variant}_s{seed}"
    c = vs.config(g, name, seed, steps)
    if variant == "halfin":
        c["network"]["input"] = {k: v * 0.5 for k, v in c["network"]["input"].items()}
    return name, c


def snake_cfg(steps):
    c = vs.config(genome("v031"), "snake_v031_s0", 0, steps)
    c.update(selection=None, move_log=None, headless={"seed": 0}, td_lambda_env=0.8, gamma_env=0.5, monitor_every=50000)
    return "snake_v031_s0", c


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out")
    ap.add_argument("--snake-steps", type=int, default=10_000_000)
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--skip", nargs="*", default=[]); a = ap.parse_args()
    out = Path(a.out).resolve()
    jobs = [snake_cfg(a.snake_steps)]
    for seed in range(3):
        for variant in ("v031", "seed", "halfin"):
            jobs.append(selection_cfg(variant, seed, a.steps))

    def run(job):
        name, c = job
        ok = das.run_probe(c, out / name, SLOTS, 24 * 3600)
        if ok and c.get("selection"):
            m = vs.train_metrics(out / name, name)
            print(name, json.dumps(m), flush=True)
        else:
            print(name, "done" if ok else "failed/timeout", flush=True)
        vs.clean(out / name)

    jobs = [j for j in jobs if j[0] not in a.skip]
    with ThreadPoolExecutor(SLOTS) as ex:
        list(ex.map(run, jobs))


if __name__ == "__main__":
    main()
