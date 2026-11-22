"""Train the v031 volume on selection to 400k steps and record windows for the open-loop toy (drift_check record)."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent / "search")); sys.path.insert(0, str(HERE.parent.parent))
import volume_search as vs, drift_arch_search as das
from slab_compare import genome
c = vs.config(genome("v031"), "recm_v031", 0, 2_000_000)
c["drift_check"] = {"at_timestep": 400000, "group": "L1_E->L1_E", "mode": "record", "windows": 16, "window": 200, "repeat": 16, "every": 5000,
                    "seed": 0}
print(das.run_probe(c, HERE / "recm_v031", 6, 4 * 3600))
