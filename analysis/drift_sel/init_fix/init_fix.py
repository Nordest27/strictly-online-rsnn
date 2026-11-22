"""Deep-layer init fix for the 3-layer mexhat stack (no training).

The 3-layer runs had 21-33% of L2/L3 E neurons silent from the first measurement. Variants of the species-search base
genome (3 layers + fields, rates calibrated layer by layer to target_hz on selection frames, then a frozen probe on
the selection task; species_search.calibrated / metrics): the feedforward spread ff_sigma (fan-in per neuron) and
random global inputs. Reported per E population: active fraction, state decoding, spikes per move; per stage the
drift's per-state snr; the policy field's state decoding; behavioural variability.

Usage: python init_fix.py OUT_DIR [--moves 800] [--slots 4]
"""
import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "search")); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "action"))
import species_search as ss  # noqa: E402
import drift_arch_search as das  # noqa: E402

VARIANTS = {
    "ff0.6": {},
    "ff1.0": {"ff_sigma": 1.0},
    "ff1.6": {"ff_sigma": 1.6},
    "ff1.0_glob8": {"ff_sigma": 1.0, "global_fan_in": 8},
    "ff0.6_glob8": {"global_fan_in": 8},
}


def one(out, name, extra, a):
    work = out / "work" / name
    work.mkdir(parents=True, exist_ok=True)
    genome = {**ss.BASE, **extra}
    try:
        arch = ss.calibrated(genome, 3, work)
        cfg = das.probe_config(arch, name, a.moves, True)
        ok = das.run_probe(cfg, work, a.slots, 3600)
        m = ss.metrics(work, name, 3) if ok else None
    except Exception as e:
        print(name, "failed:", repr(e), flush=True); return
    json.dump({"name": name, "genome": genome, "metrics": m}, open(out / f"{name}.json", "w"))
    report(name, m)


def report(name, m):
    if m is None:
        print(name, "no metrics", flush=True); return
    pops = m["pops"]
    line = f"{name:13s} var {m['variability']:.2f} |"
    for p in ("L1_E", "L2_E", "L3_E", "policy_field"):
        v = pops.get(p, {})
        line += f" {p} act {v.get('active', float('nan')):.2f} dec {v.get('decode', float('nan')):.2f} spk {v.get('count', float('nan')):.2f} |"
    line += " snr " + " ".join(f"{k}:{(s or {}).get('snr', float('nan')):.1e}" for k, s in m["stages"].items())
    print(line, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--moves", type=int, default=800)
    ap.add_argument("--slots", type=int, default=4); a = ap.parse_args()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(3) as ex:
        for n, e in VARIANTS.items():
            ex.submit(one, out, n, e, a); time.sleep(5)
