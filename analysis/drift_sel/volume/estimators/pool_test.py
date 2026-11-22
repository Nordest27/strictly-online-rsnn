"""Closed-loop check of the headless pool features and locally coherent noise on the v031 volume, headless selection
task (600k steps; baseline = route_cmp/base_s*, same config and seeds without these features).
Variants: pool_rand (pool_credit + pool_homeo, random pools), pool_cont (same, contiguous pools), homeo (pool_homeo
only, contiguous), local_credit / local_inject (network.local_noise sigma 1.5, the rule's pool_noise [+ inject]).
Usage: python pool_test.py OUT_DIR [--seeds 0 1] [--variants ...]
"""
import sys, json, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent / "search")); sys.path.insert(0, str(HERE.parent.parent))
import volume_search as vs, drift_arch_search as das
from slab_compare import genome

HOMEO = 1e-4
VARIANTS = {
    "pool_rand": {"headless": {"pools": "random", "pool_credit": True, "pool_homeo": HOMEO}},
    "pool_cont": {"headless": {"pools": "contiguous", "pool_credit": True, "pool_homeo": HOMEO}},
    "homeo": {"headless": {"pools": "contiguous", "pool_homeo": HOMEO}},
    "local_credit": {"network": {"local_noise": {"sigma": 1.5, "mode": "credit"}}, "hidden_rule": {"pool_noise": True}},
    "local_inject": {"network": {"local_noise": {"sigma": 1.5, "mode": "inject"}},
                     "hidden_rule": {"pool_noise": True, "pool_noise_inject": True}},
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS)); ap.add_argument("--slots", type=int, default=6)
    ap.add_argument("--steps", type=int, default=600_000); ap.add_argument("--keep", action="store_true")
    a = ap.parse_args(); out = Path(a.out).resolve()
    for seed in a.seeds:
        for vname in a.variants:
            name = f"{vname}_s{seed}"
            if (out / name / "metrics.json").exists():
                continue
            c = vs.config(genome("v031"), name, seed, a.steps)
            for key, upd in VARIANTS[vname].items():
                c[key] = {**(c.get(key) or {}), **upd}
            ok = das.run_probe(c, out / name, a.slots, 8 * 3600)
            m = vs.train_metrics(out / name, name) if ok else None
            json.dump(m, open(out / name / "metrics.json", "w"))
            print(name, json.dumps(m), flush=True)
            if not a.keep:
                vs.clean(out / name)


if __name__ == "__main__":
    main()
