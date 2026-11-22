"""Closed-loop test of the delta-gated contrast baselines (local.contrast_baseline) on the v031 volume, selection task,
600k steps (the search's setting; the "mean" baseline = the search's own v031 runs, same config and seeds).

  B gated_mean      : delta * (ContrD - running mean of ContrD)
  C gated_synapse   : delta * (ContrD - per-synapse critic of ContrD), critic rate 1e-3
  D gated_synapse+G : C with the gradient term G = 1 (the direction left to G)
One simulation at a time beside the search (slot limit 6 = 5 simulations + the thermal guard).
Usage: python gated_contrast_test.py OUT_DIR [--seeds 0 1]
"""
import sys, json, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent / "search")); sys.path.insert(0, str(HERE.parent.parent))
import volume_search as vs, drift_arch_search as das
from slab_compare import genome

VARIANTS = {"gated_mean": ({"contrast_baseline": "gated_mean"}, None),
            "gated_synapse": ({"contrast_baseline": "gated_synapse", "contrast_critic_eta": 1e-3}, None),
            "gated_synapse_G": ({"contrast_baseline": "gated_synapse", "contrast_critic_eta": 1e-3}, 1.0)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    a = ap.parse_args(); out = Path(a.out).resolve()
    for seed in a.seeds:
        for vname, (loc, grad) in VARIANTS.items():
            name = f"{vname}_s{seed}"
            if (out / name / "metrics.json").exists():
                continue
            c = vs.config(genome("v031"), name, seed, 600_000)
            c["hidden_rule"]["local"].update(loc)
            if grad is not None:
                c["hidden_rule"]["gradient"] = grad
            ok = das.run_probe(c, out / name, 6, 6 * 3600)
            m = vs.train_metrics(out / name, name) if ok else None
            json.dump(m, open(out / name / "metrics.json", "w"))
            print(name, json.dumps(m), flush=True)
            vs.clean(out / name)


if __name__ == "__main__":
    main()
