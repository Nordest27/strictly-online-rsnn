"""v031 with the learned learning signal (credit 1): train to 400k steps, then record 8 states x 4 windows of 200 steps
(incl. CreditH, CreditY, CreditL) to measure how well the credit prediction L tracks its target Y."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent / "search")); sys.path.insert(0, str(HERE.parent.parent))
import volume_search as vs, drift_arch_search as das
from slab_compare import genome
c = vs.config(genome("v031"), "rec_credit", 0, 2_000_000)
c["hidden_rule"].update(credit=1.0, credit_eta=1e-3)
c["network"]["credit"] = {"sources": ["policy_field", "value_field"], "k": 16, "self_k": 16, "seed": 0}
c["drift_check"] = {"at_timestep": 400000, "group": "L1_E->L1_E", "mode": "record", "windows": 4, "window": 200,
                    "repeat": 8, "every": 5000, "seed": 0}
print(das.run_probe(c, HERE / "rec_credit", 6, 4 * 3600))
