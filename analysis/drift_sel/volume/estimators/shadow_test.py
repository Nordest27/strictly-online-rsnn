"""Closed-loop test of the shadow-centred drift (hidden_rule.drift_centre "shadow", synapse noise + relays) on the v031
volume, selection, 600k steps; the current rule = the search's own v031 runs (same config and seeds).
Usage: python shadow_test.py OUT_DIR [--seeds 0 1] [--slots 6]"""
import sys, json, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent / "search")); sys.path.insert(0, str(HERE.parent.parent))
import volume_search as vs, drift_arch_search as das
from slab_compare import genome

VARIANTS = {"shadow_drift": {"drift_centre": "shadow"},
            "shadow_cv": {"drift_centre": "shadow_cv", "shadow_align_rate": 0.001, "shadow_align_tau": 1000.0},
            "shadow_cv_lr3": {"drift_centre": "shadow_cv", "shadow_align_rate": 0.001, "shadow_align_tau": 1000.0,
                              "_top": {"lr": 1e-5 / 3}},
            "shadow_cv_lr10": {"drift_centre": "shadow_cv", "shadow_align_rate": 0.001, "shadow_align_tau": 1000.0,
                               "_top": {"lr": 1e-6}}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS)); ap.add_argument("--slots", type=int, default=6)
    a = ap.parse_args(); out = Path(a.out).resolve()
    for seed in a.seeds:
        for v in a.variants:
            name = f"{v}_s{seed}"
            if (out / name / "metrics.json").exists():
                continue
            c = vs.config(genome("v031"), name, seed, 600_000)
            upd = dict(VARIANTS[v]); top = upd.pop("_top", {})
            c["hidden_rule"].update(upd); c.update(top)
            ok = das.run_probe(c, out / name, a.slots, 6 * 3600)
            m = vs.train_metrics(out / name, name) if ok else None
            json.dump(m, open(out / name / "metrics.json", "w"))
            print(name, json.dumps(m), flush=True)
            vs.clean(out / name)


if __name__ == "__main__":
    main()
