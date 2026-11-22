"""Big 3D volume for the GPU (13 Nov 2026): config generator + CPU quick tests (frozen, no learning).

The volume recipe of the search (vol8 config: drift + contrast, decorr) scaled up: side x side x depth E sheet, long-
range connections as a count (VolumeSpec.k_global) instead of a probability, input into the front slab, fields read
the back slabs (v031 layout).

  python big_volume.py config OUT.json [--task snake|selection] [--set key=value ...]   write a config
  python big_volume.py slabs OUT_DIR [--set ...]       frozen selection probe: per-slab rate / state decoding
  python big_volume.py snake OUT_DIR [--set ...]       frozen Snake run: per-population rates (neuron monitor)
"""
import json, sys, argparse, copy
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "search")); sys.path.insert(0, str(HERE.parent))
import drift_arch_search as das  # noqa: E402

BASE_CFG = HERE.parent / "volume_train/vol8_s0/config.json"
DEFAULTS = {"side": 16, "side_i": 8, "depth": 16, "z_shift": 1.5, "sigma_cells": 1.0, "p_max": 0.8,
            "k_global": 3.0, "k_global_mode": "in", "ee_gain": 0.6, "ii_gain": 0.1, "inh_sigma_ratio": 1.0,
            "input_depth": 1, "output_depth": 3, "field_sigma_cells": 1.0, "field_gain": 1.0, "input_gain": 1.0,
            "decorr": 0.001, "lr": 1e-6, "seed": 0}


def make_config(p, task="snake", name="bigvol", steps=30_000_000, frozen=False):
    c = json.load(open(BASE_CFG))
    net = c["network"]
    for k in ("conn", "conn_i"):
        net["fields"][k] = {**net["fields"][k], "mean_scale": net["fields"][k]["mean_scale"] * p["field_gain"],
                            "sd_scale": net["fields"][k]["sd_scale"] * p["field_gain"]}
    net["input"] = {k: v * p["input_gain"] for k, v in net["input"].items()}
    vol = {k: p[k] for k in ("depth", "input_depth", "output_depth", "sigma_cells", "p_max", "z_shift",
                             "inh_sigma_ratio", "field_sigma_cells", "k_global", "k_global_mode")}
    net["layers"] = [{"volume": vol, "internal": {"mean_scale": p["ee_gain"]}, "internal_i": {"mean_scale": p["ii_gain"]}}]
    c["hid_e"], c["hid_i"] = p["side"], p["side_i"]
    c["hidden_rule"]["local"]["decorr"] = p["decorr"]
    c.update(seed=p["seed"], csv_prefix=name, lr=p["lr"], max_timesteps=int(steps), monitor=True, monitor_every=2000)
    c.pop("backend", None)                                  # GPU: GeNN's default (CUDA)
    if task == "snake":
        c.update(selection=None, move_log=None, headless={"seed": p["seed"]}, gamma_env=0.5, td_lambda_env=0.8)
    else:
        c.update(selection={**c["selection"], "seed": p["seed"]}, headless={**c["headless"], "seed": p["seed"]},
                 move_log={"probe_every": 10 ** 12, "probe_moves": 16})
    if frozen:
        c.update(lr=0.0, backend="single_threaded_cpu")
    return c


def params(sets):
    p = dict(DEFAULTS)
    for s in sets or []:
        k, v = s.split("=", 1)
        p[k] = type(DEFAULTS[k])(v) if k in DEFAULTS and not isinstance(DEFAULTS[k], str) else v
    return p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("out")
    ap.add_argument("--task", default="snake"); ap.add_argument("--steps", type=float, default=30e6)
    ap.add_argument("--name", default="bigvol"); ap.add_argument("--set", nargs="*"); a = ap.parse_args()
    p = params(a.set)
    if a.cmd == "config":
        json.dump(make_config(p, a.task, a.name, a.steps), open(a.out, "w"), indent=1); print("wrote", a.out); return
    out = Path(a.out).resolve()
    if a.cmd == "slabs":
        import analyze_sep
        moves = 400
        c = make_config(p, "selection", a.name, moves * das.STEPS_PER_MOVE + 2000, frozen=True)
        c["move_log"] = {"probe_every": 10 ** 12, "probe_moves": 16, "activity": moves - 60}
        work = out / a.name
        assert das.run_probe(c, work, 5, 4 * 3600), "probe failed"
        act = np.load(work / "outputs" / f"{a.name}(0)_activity.npz"); ct = act["ctype"]; D = p["depth"]
        print(f"== {a.name} {json.dumps(p)}")
        for z in range(D):
            row = []
            for pop in ("L1_E", "L1_I"):
                Z = act[pop].astype(float).reshape(len(ct), -1, D)[:, :, z]
                _, dec, active, cnt = analyze_sep.separation(Z, ct)
                row.append(f"{pop[-1]} spk {cnt:.2f} act {active:.2f} dec {dec:.2f}")
            tag = "in " if z < p["input_depth"] else ("out" if z >= D - p["output_depth"] else "   ")
            print(f"  slab {z:2d} {tag} " + " | ".join(row))
        _, dec, active, cnt = analyze_sep.separation(act["policy_field"].astype(float), ct)
        print(f"  field   spk {cnt:.2f} act {active:.2f} dec {dec:.2f}")
    elif a.cmd == "snake":
        import csv
        c = make_config(p, "snake", a.name, a.steps, frozen=True)
        work = out / a.name
        assert das.run_probe(c, work, 5, 4 * 3600), "run failed"
        rows = list(csv.DictReader(open(work / "outputs" / f"{a.name}(0)_neurons.csv")))
        keys = [k for k in rows[0] if k.endswith("_hz") or k.endswith("_dead")] if rows else []
        print(f"== {a.name} {json.dumps(p)}")
        for r in rows:
            print("  ", r.get("timestep"), " ".join(f"{k} {float(r[k]):.2f}" for k in keys))


if __name__ == "__main__":
    main()
