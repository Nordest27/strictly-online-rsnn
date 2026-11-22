"""Closed-loop test of the reward neurons (local.learned_reward + network.reward_pool) on the v031 volume, selection,
600k steps; the current rule = the search's own v031 runs (same config and seeds). Variants: dict of hidden_rule
updates, "_net" network updates, "_top" top-level config updates.
Usage: python credit_test.py OUT_DIR --variants NAME [...] --seeds 0 1 [--slots 6]"""
import sys, json, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent / "search")); sys.path.insert(0, str(HERE.parent.parent))
import volume_search as vs, drift_arch_search as das
from slab_compare import genome

POOL = {"reward_pool": {"n": 64, "in_from": ["L1_E"], "k_in": 16, "in_mean": 0.5, "in_sd": 0.1, "targets": "all", "k_out": 8, "out_scale": 1.0, "seed": 0}}
POOL_COOL = {"reward_pool": {**POOL["reward_pool"], "in_mean": 0.2}}
POOL_SIGNAL = {"reward_pool": {**POOL_COOL["reward_pool"], "channel": "signal"}}
VARIANTS = {
    "smoke": {"local": {"learned_reward": 1.0}, "_net": POOL},
    "smoke_links": {"credit": 10.0, "local": {"learned_reward": 10.0},
                    "_net": {"reward_links": {"channel": "signal", "topology": "mirror", "scale": 1.0, "seed": 0}}},
    "reward1": {"local": {"learned_reward": 1.0}, "_net": POOL},
    "reward10": {"local": {"learned_reward": 10.0}, "_net": POOL},
    "reward10_frozen": {"local": {"learned_reward": 10.0}, "_net": POOL, "_pool_rule": "frozen"},
    "reward10_slow": {"local": {"learned_reward": 10.0}, "_net": POOL, "_pool_rule": {"trace_lambda": 0.9999}},
    "reward10_cool": {"local": {"learned_reward": 10.0}, "_net": POOL_COOL},
    "reward10_cool_frozen": {"local": {"learned_reward": 10.0}, "_net": POOL_COOL, "_pool_rule": "frozen"},
    "signal10_cool": {"credit": 10.0, "_net": POOL_SIGNAL},
    "links_reward10": {"local": {"learned_reward": 10.0},
                       "_net": {"reward_links": {"channel": "reward", "topology": "mirror", "scale": 1.0, "seed": 0}}},
    "tlinks_reward": {"local": {"learned_reward": 1.0}, "link_eta": 1e-5, "link_decay": 1e-4,
                      "_net": {"reward_links": {"channel": "reward", "topology": "mirror", "init": "zero", "seed": 0}}},
    "tlinks_signal": {"credit": 1.0, "link_eta": 1e-5, "link_decay": 1e-4,
                      "_net": {"reward_links": {"channel": "signal", "topology": "mirror", "init": "zero", "seed": 0}}},
    "links_signal10": {"credit": 10.0, "_net": {"reward_links": {"channel": "signal", "topology": "mirror", "scale": 1.0, "seed": 0}}},
    "signal10_cool_frozen": {"credit": 10.0, "_net": POOL_SIGNAL, "_pool_rule": "frozen"},
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
            c = vs.config(genome("v031"), name, seed, 30_000 if v.startswith("smoke") else 600_000)
            upd = dict(VARIANTS[v]); net = upd.pop("_net", {}); top = upd.pop("_top", {})
            loc = upd.pop("local", {}); pool_rule = upd.pop("_pool_rule", None)
            c["hidden_rule"].update(upd); c["hidden_rule"]["local"].update(loc); c["network"].update(json.loads(json.dumps(net))); c.update(top)
            if pool_rule == "frozen":        # the reward neurons do not learn: a fixed random reward generator
                c["network"]["reward_pool"]["rule"] = {"preset": "drift_only", "drift": 0.0, "gradient": 0.0}
            elif isinstance(pool_rule, dict):  # the global rule with overrides (e.g. a long trace horizon)
                c["network"]["reward_pool"]["rule"] = {**json.loads(json.dumps(c["hidden_rule"])), **pool_rule}
            ok = das.run_probe(c, out / name, a.slots, 6 * 3600)
            m = vs.train_metrics(out / name, name) if ok else None
            json.dump(m, open(out / name / "metrics.json", "w"))
            print(name, json.dumps(m), flush=True)
            if v.startswith("tlinks"):        # keep the checkpoints (trained link structure); drop the code
                import shutil
                for f in (out / name).glob("*_CODE"):
                    shutil.rmtree(f, ignore_errors=True)
            else:
                vs.clean(out / name)


if __name__ == "__main__":
    main()
