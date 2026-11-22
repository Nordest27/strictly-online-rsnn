"""v031 with the shadow-centred drift, alignment rate kappa (argv[1]); train 100k steps, record 8 windows of 200 steps
(V, A, refractoriness, the shadow's state, Psi, PsiBar) to measure how well the live shadow tracks the neuron."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent / "search")); sys.path.insert(0, str(HERE.parent.parent))
import volume_search as vs, drift_arch_search as das
from slab_compare import genome
k = float(sys.argv[1]); name = f"rec_align_{sys.argv[1].replace('.', 'p')}"
c = vs.config(genome("v031"), name, 0, 2_000_000)
c["hidden_rule"].update(drift_centre="shadow", shadow_align_rate=k, shadow_align_tau=1000.0)
c["drift_check"] = {"at_timestep": 100000, "group": "L1_E->L1_E", "mode": "record", "windows": 8, "window": 200, "seed": 0}
print(name, das.run_probe(c, HERE / name, 6, 3 * 3600), flush=True)
