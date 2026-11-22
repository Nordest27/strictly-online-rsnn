"""Experiment 1: passive (trained but unused) learning-signal links in v031 under the current rule. The links learn by
the alignment rule (link_eta) but credit = 0, so the network is unchanged; recorded at 400k (8 states x 4 windows) with
the link signal CreditL, the receivers' credit CreditY and the link weights.
Usage: python passive_links.py OUT_DIR TOPOLOGY SEED"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent / "search")); sys.path.insert(0, str(HERE.parent.parent))
import volume_search as vs, drift_arch_search as das
from slab_compare import genome
out, topo, seed = Path(sys.argv[1]).resolve(), sys.argv[2], int(sys.argv[3])
name = f"passive_{topo}_s{seed}"
c = vs.config(genome("v031"), name, seed, 2_000_000)
c["hidden_rule"].update(link_eta=1e-5, link_decay=1e-4)
c["network"]["reward_links"] = {"channel": "signal", "topology": topo, "init": "zero", "seed": 0}
c["drift_check"] = {"at_timestep": 400000, "group": "L1_E->L1_E", "mode": "record", "windows": 4, "window": 200,
                    "repeat": 8, "every": 5000, "seed": 0}
print(name, das.run_probe(c, out / name, 6, 4 * 3600), flush=True)
