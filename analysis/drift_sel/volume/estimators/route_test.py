"""Closed-loop test of routed credit (hidden_rule.route): learned computation sensitivities on route links reversed
along the network's own wiring (network.reward_links channel "route", topology "reverse", zero init), per-neuron trust
(prior + own evidence). v031 volume, headless selection task; baseline = the same config without route (the search's
own v031 runs share config and seeds at 600k steps).
One simulation per call of run_probe (slot limit as the other tests: thermal guard counted).
Usage: python route_test.py OUT_DIR [--seeds 0 1] [--variants ...] [--steps 600000]
"""
import sys, json, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent / "search")); sys.path.insert(0, str(HERE.parent.parent))
import volume_search as vs, drift_arch_search as das
from slab_compare import genome

LINKS = {"channel": "route", "topology": "reverse", "init": "zero"}
VARIANTS = {"base": None,
            "route": {"route": 1.0},
            "route_1hop": {"route": 1.0, "route_recursive": False},
            "route_fast": {"route": 1.0, "route_eta": 1e-2, "route_prior": 2e4},
            # links not along the wiring: 16 random senders per neuron, slower score for indirect influences
            "route_random": {"route": 1.0, "route_score_decay": 0.8},
            # links between nearby neurons (Gaussian in space, sigma 1.5 cells), whether wired or not
            "route_local": {"route": 1.0, "route_score_decay": 0.8},
            # links learned from activity (the receiver's spikes): high-signal transport
            "spike": {"route": 1.0, "route_link_signal": "spike", "route_eta": 1e-3},
            "spike_slow": {"route": 1.0, "route_link_signal": "spike", "route_eta": 1e-4},
            "spike_local": {"route": 1.0, "route_link_signal": "spike", "route_eta": 1e-3},
            "noise_slow": {"route": 1.0, "route_eta": 1e-4}}
TOPOLOGY = {"route_random": {"channel": "route", "topology": "random", "k": 16, "init": "zero"},
            "route_local": {"channel": "route", "topology": "local", "k": 16, "sigma": 1.5, "init": "zero"},
            "spike_local": {"channel": "route", "topology": "local", "k": 16, "sigma": 1.5, "init": "zero"}}


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
            if VARIANTS[vname] is not None:
                c["hidden_rule"].update(VARIANTS[vname])
                c["network"]["reward_links"] = dict(TOPOLOGY.get(vname, LINKS))
            ok = das.run_probe(c, out / name, a.slots, 8 * 3600)
            m = vs.train_metrics(out / name, name) if ok else None
            json.dump(m, open(out / name / "metrics.json", "w"))
            print(name, json.dumps(m), flush=True)
            if not a.keep:
                vs.clean(out / name)


if __name__ == "__main__":
    main()
