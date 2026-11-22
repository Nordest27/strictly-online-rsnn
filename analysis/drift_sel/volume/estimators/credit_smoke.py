"""Smoke run: v031 selection with the learned learning signal (credit 1, sources: the fields + own population)."""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent / "search")); sys.path.insert(0, str(HERE.parent.parent))
import volume_search as vs, drift_arch_search as das
from slab_compare import genome
c = vs.config(genome("v031"), "credit_smoke", 0, 30000)
c["hidden_rule"].update(credit=1.0, credit_eta=1e-3)
c["network"]["credit"] = {"sources": ["policy_field", "value_field"], "k": 16, "self_k": 16, "seed": 0}
print(das.run_probe(c, HERE / "credit_smoke", 6, 3600))
