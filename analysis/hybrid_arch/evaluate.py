"""Untrained evaluation of the candidates with the architecture search's pipeline (calibrate to 20 Hz, then
per-population decoding of the task variables, input reconstruction, dimensionality, memory, psi, perturbation
reach, search score). Sequential, one network at a time on the GPU."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reservoir_search as rs
import architecture_evolution as ae
from candidates import CANDIDATES

FRAMES = "/home/user/repos/personal/analysis/arch_shift2_tries_full/frames.npz"
names = sys.argv[1:] or list(CANDIDATES)
d = np.load(FRAMES)
rates, ys = d["rates"], d["ys"]
for name in names:
    hp = CANDIDATES[name]
    work = HERE / "work" / name; work.mkdir(parents=True, exist_ok=True)
    built = rs.build(hp, "full", 5)
    wires = ae.wiring(built)
    with rs.Sim(built, name, work, backend="cuda") as sim:
        rs.calibrate(sim, rates[:1500], 20.0, log=lambda s: None)
        per_pop = rs.evaluate(sim, rates[1500:], ys[1500:])
        reach = rs.perturbation_reach(sim, rates[1500:], 0.03)
        factors = dict(sim.factors)
    res = {"name": name, "hp": hp, "per_pop": per_pop, "wiring": wires, "reach": reach, "prior": "relay",
           "calibration_factors": {str(k): v for k, v in factors.items()},
           "snake_config": rs.snake_config(hp, factors, 5, "full")}
    res["score"], res["parts"] = ae.score(res, 5)
    (HERE / f"{name}.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"\n== {name}  score {res['score']:.3f}  parts {json.dumps(res['parts'], default=float)}")
    print(f"   reach {json.dumps({str(k): round(v, 3) for k, v in reach.items()})}")
    print(f"   {'pop':14s} {'n':>5s} {'Hz':>5s} {'dim':>5s} {'decode':>6s} {'in_r2':>6s} {'early':>6s} {'mem1-3':>15s} {'psi':>5s} {'psiD':>5s} {'prev':>5s}")
    for p in per_pop:
        mem = " ".join(f"{m:.2f}" for m in p["memory"]) if p.get("memory") else ""
        print(f"   {p['pop']:14s} {p['n']:5d} {p['hz']:5.1f} {p['dim']:5.1f} {p['decode']:6.3f} {p['input_r2']:6.3f} "
              f"{p['input_r2_early']:6.3f} {mem:>15s} {p['psi']:5.3f} {p['psi_dead']:5.2f} {p['pred_from_prev']:5.2f}", flush=True)
