"""Per-slab frozen probe of search genomes with the search's exact config (volume_search.config).

Usage: python slab_compare.py OUT_DIR NAME [NAME ...]   (names from search/volume_v1/results.jsonl)
Per slab of the volume (E and I): spikes per move, active fraction, state decoding; plus the policy field.
"""
import json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "search")); sys.path.insert(0, str(HERE.parent))
import volume_search as vs  # noqa: E402
import drift_arch_search as das  # noqa: E402
import analyze_sep  # noqa: E402


def genome(name):
    for l in open(HERE.parent / "search/volume_v1/results.jsonl"):
        r = json.loads(l)
        if r["name"] == name:
            return r["genome"]
    raise KeyError(name)


def probe(out, name):
    g = genome(name); tag = f"{name}_slab"; work = out / tag
    if not (work / "outputs" / f"{tag}(0)_activity.npz").exists():
        assert das.run_probe(vs.config(g, tag, 0), work, 3, 3600), name
    act = np.load(work / "outputs" / f"{tag}(0)_activity.npz"); ct = act["ctype"]; D = g["depth"]
    res = {"name": name, "genome": g, "slabs": {}}
    for pop in ("L1_E", "L1_I"):
        Z = act[pop].astype(float).reshape(len(ct), -1, D)
        res["slabs"][pop] = [dict(zip(("sep", "decode", "active", "spikes"),
                                      map(float, analyze_sep.separation(Z[:, :, z], ct)))) for z in range(D)]
    s, d, a_, c = analyze_sep.separation(act["policy_field"].astype(float), ct)
    res["field"] = {"decode": float(d), "active": float(a_), "spikes": float(c)}
    return res


def main():
    out = Path(sys.argv[1]).resolve(); out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(len(sys.argv) - 2) as ex:
        rs = list(ex.map(lambda n: probe(out, n), sys.argv[2:]))
    json.dump(rs, open(out / "slabs.json", "w"), indent=1)
    for r in rs:
        g = r["genome"]; D = g["depth"]
        print(f"== {r['name']}: input slabs 0..{g['input_depth'] - 1}, output slabs {D - g['output_depth']}..{D - 1}")
        for z in range(D):
            e, i = r["slabs"]["L1_E"][z], r["slabs"]["L1_I"][z]
            tag = "in " if z < g["input_depth"] else ("out" if z >= D - g["output_depth"] else "   ")
            print(f"  slab {z} {tag} E spk {e['spikes']:.2f} act {e['active']:.2f} dec {e['decode']:.2f} | "
                  f"I spk {i['spikes']:.2f} dec {i['decode']:.2f}")
        f = r["field"]; print(f"  field  spk {f['spikes']:.2f} act {f['active']:.2f} dec {f['decode']:.2f}")


if __name__ == "__main__":
    main()
