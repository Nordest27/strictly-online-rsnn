"""Frozen probe of 3D volume networks on the selection task (no training): per-slab activity and the field's view.

Builds the config from the 1-layer mexhat arch (depth_contrast/arch_1l.json) with network.layers[0].volume set, runs
snake.py frozen (lr 0) for `moves` decisions with per-move activity (drift_arch_search.probe_config), then reports per
slab of the volume: spikes per move, active fraction, state decoding (nearest centroid, analyze_sep); and the policy
field's rate and state decoding.

Usage: python probe_volume.py OUT_DIR NAME '{"depth": 8, ...}' ['{extra network.layers[0] keys}'] [--moves 600]
"""
import json, sys, argparse
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "search")); sys.path.insert(0, str(HERE.parent))
import drift_arch_search as das  # noqa: E402
import analyze_sep  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("name"); ap.add_argument("volume")
    ap.add_argument("layer", nargs="?", default="{}"); ap.add_argument("--moves", type=int, default=600)
    ap.add_argument("--slots", type=int, default=5); a = ap.parse_args()
    arch = json.load(open(HERE.parent / "depth_contrast/arch_1l.json"))
    vol = json.loads(a.volume)
    arch["network"]["layers"] = [{**json.loads(a.layer), "volume": vol}]
    out = Path(a.out).resolve(); work = out / a.name
    cfg = das.probe_config(arch, a.name, a.moves, True)
    cfg["move_log"]["activity"] = a.moves - 60          # the run can end a few moves short of `moves`
    if not das.run_probe(cfg, work, a.slots, 3600):
        print(a.name, "probe failed"); return
    act = np.load(f"{work}/outputs/{a.name}(0)_activity.npz")
    ct = act["ctype"]; depth = vol["depth"]
    res = {"name": a.name, "volume": vol, "layer": json.loads(a.layer), "slabs": {}, "pops": {}}
    for pop in ("L1_E", "L1_I"):
        X = act[pop].astype(float); n = X.shape[1]
        Z = X.reshape(len(X), -1, depth)        # flat C order: depth last
        res["slabs"][pop] = []
        for z in range(depth):
            s, dec, active, cnt = analyze_sep.separation(Z[:, :, z], ct)
            res["slabs"][pop].append({"count": float(cnt), "active": float(active), "decode": float(dec)})
    for pop in ("policy_field", "value_field"):
        s, dec, active, cnt = analyze_sep.separation(act[pop].astype(float), ct)
        res["pops"][pop] = {"count": float(cnt), "active": float(active), "decode": float(dec)}
    json.dump(res, open(out / f"{a.name}.json", "w"), indent=1)
    print(f"== {a.name} {json.dumps(vol)} {a.layer}")
    for z, r in enumerate(res["slabs"]["L1_E"]):
        print(f"  slab {z}: E spk/move {r['count']:.2f} active {r['active']:.2f} state {r['decode']:.2f} | "
              f"I spk/move {res['slabs']['L1_I'][z]['count']:.2f}")
    pf = res["pops"]["policy_field"]
    print(f"  policy field: spk/move {pf['count']:.2f} active {pf['active']:.2f} state {pf['decode']:.2f}")


if __name__ == "__main__":
    main()
