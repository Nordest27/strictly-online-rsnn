"""Closed-loop test of the learned learning signal (hidden_rule.credit + network.credit) on the v031 volume, selection,
600k steps; the current rule = the search's own v031 runs (same config and seeds). Variants: dict of hidden_rule
updates, "_net" network updates, "_top" top-level config updates.
Usage: python credit_test.py OUT_DIR --variants NAME [...] --seeds 0 1 [--slots 6]"""
import sys, json, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent / "search")); sys.path.insert(0, str(HERE.parent.parent))
import volume_search as vs, drift_arch_search as das
from slab_compare import genome

CREDIT_NET = {"credit": {"sources": ["policy_field", "value_field"], "k": 16, "self_k": 16, "seed": 0}}
VARIANTS = {
    "credit1": {"credit": 1.0, "credit_eta": 1e-3, "_net": CREDIT_NET},
    "g1": {"gradient": 1.0},
    "credit3": {"credit": 3.0, "credit_eta": 1e-3, "_net": CREDIT_NET},
    "credit03": {"credit": 0.3, "credit_eta": 1e-3, "_net": CREDIT_NET},
    "credit1_eta1e4": {"credit": 1.0, "credit_eta": 1e-4, "_net": CREDIT_NET},
    "credit1_eta3e4": {"credit": 1.0, "credit_eta": 3e-4, "_net": CREDIT_NET},
    "credit1_eta1e2": {"credit": 1.0, "credit_eta": 1e-2, "_net": CREDIT_NET},
    "credit1_g1": {"credit": 1.0, "credit_eta": 1e-3, "gradient": 1.0, "_net": CREDIT_NET},
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--variants", nargs="+", required=True); ap.add_argument("--slots", type=int, default=6)
    a = ap.parse_args(); out = Path(a.out).resolve()
    for seed in a.seeds:
        for v in a.variants:
            name = f"{v}_s{seed}"
            if (out / name / "metrics.json").exists():
                continue
            c = vs.config(genome("v031"), name, seed, 600_000)
            upd = dict(VARIANTS[v]); net = upd.pop("_net", {}); top = upd.pop("_top", {})
            c["hidden_rule"].update(upd); c["network"].update(net); c.update(top)
            ok = das.run_probe(c, out / name, a.slots, 6 * 3600)
            m = vs.train_metrics(out / name, name) if ok else None
            json.dump(m, open(out / name / "metrics.json", "w"))
            print(name, json.dumps(m), flush=True)
            vs.clean(out / name)


if __name__ == "__main__":
    main()
