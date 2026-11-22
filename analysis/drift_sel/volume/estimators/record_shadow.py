"""v031 with the shadow (drift_centre shadow, synapse noise): train 20k steps, record 8 windows (incl. ShadowV) for the
end-to-end check of the relay + shadow against the open-loop replay with zero noise."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent / "search")); sys.path.insert(0, str(HERE.parent.parent))
import volume_search as vs, drift_arch_search as das
from slab_compare import genome
c = vs.config(genome("v031"), "rec_shadow", 0, 2_000_000)
c["hidden_rule"]["drift_centre"] = "shadow"
c["drift_check"] = {"at_timestep": 20000, "group": "L1_E->L1_E", "mode": "record", "windows": 8, "window": 200, "seed": 0}
print(das.run_probe(c, HERE / "rec_shadow", 6, 3600))
