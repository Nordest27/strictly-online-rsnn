"""Architecture search over 3D volume networks (snake_network VolumeSpec) for the headless drift + contrast recipe.

    python volume_search.py OUT_DIR [--hours 12] [--parallel 4] [--train-steps 600000]
    python volume_search.py OUT_DIR --report

Each candidate is a volume genome (GENES below; 6x6 E / 4x4 I sheets, `depth` slabs, at least MIN_GAP slabs between
the input slabs and the output slabs, so information must travel). Evaluation:
1. frozen probe (lr 0, PROBE_MOVES decisions on the selection task, per-move activity): rejected if the back (output)
   slabs or the policy field are nearly silent, or the field carries no state information (nearest-centroid
   decoding of the 16 candidate types below MIN_FIELD_STATE; chance 1/16);
2. training: the recipe of depth_contrast/L1 and volume_train (drift + contrast reward on the lambda-trace,
   uncentred, decorr = the genome's brake) on the selection task for --train-steps. Fitness
       late gap (commit rate on good minus on bad candidates over the last LATE decisions)
       + 0.5 x trend (late gap minus the gap over the first LATE decisions, clipped at 0)
       - 0.5 x the fraction of dead policy-field neurons at the end.
   Rejected candidates get -1 + 0.1 x their field state decoding.
Seeds vary a lot (the first volume learned in one seed of two), so whenever a candidate enters the top TOP_RESEED
with one seed it is re-trained with the next seed; its fitness is the mean over its seeds.
New candidates: the seed genome (the volume that learned in volume_train) first, then mutations (1 gene changed with
probability 1, each other with probability 0.25) of a genome that passed the frozen filter, chosen among the top 8
passing ones (the seed genome while none has passed yet). Fully random genomes are not used: almost all of them are
silent beyond 4 slabs (the first 13 of the v1 run), so the search stays in the region that relays activity.
Results in OUT_DIR/results.jsonl (resumable). At most --parallel simulations at once (the thermal guard is extra).
"""
import argparse
import copy
import csv
import json
import os
import random
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import drift_arch_search as das  # noqa: E402
import analyze_sep  # noqa: E402

BASE_CFG = HERE.parent / "volume_train/vol8_s0/config.json"
GENES = {   # name: (kind, lo, hi)
    "depth": ("int", 6, 12),
    "z_shift": ("lin", 0.0, 2.0),
    "sigma_cells": ("log", 0.7, 1.6),
    "p_max": ("lin", 0.4, 1.0),
    "p_global": ("log0", 0.001, 0.03),        # with probability 0.2: 0 (no shortcuts)
    "ee_gain": ("log", 0.3, 1.0),             # internal (E sources) mean_scale
    "ii_gain": ("log", 0.05, 0.4),            # internal_i (I sources) mean_scale
    "inh_sigma_ratio": ("log", 0.7, 2.5),
    "input_depth": ("int", 1, 3),
    "output_depth": ("int", 1, 4),
    "field_sigma_cells": ("log", 0.7, 2.5),
    "field_gain": ("log", 0.5, 2.0),          # x the field connections' weights (E and I sources)
    "decorr": ("log", 0.001, 0.1),            # the brake
}
SEED_GENOME = {"depth": 8, "z_shift": 1.5, "sigma_cells": 1.0, "p_max": 0.8, "p_global": 0.01, "ee_gain": 0.6,
               "ii_gain": 0.1, "inh_sigma_ratio": 1.0, "input_depth": 2, "output_depth": 2, "field_sigma_cells": 1.0,
               "field_gain": 1.0, "decorr": 0.001}
MIN_GAP = 4                 # slabs between the input slabs and the output slabs
PROBE_MOVES = 600
MIN_FIELD_STATE = 0.09
LATE = 4000
TOP_RESEED = 5
N_RANDOM = 12
LOCK = threading.Lock()


def sample_gene(rng, name):
    kind, lo, hi = GENES[name]
    if kind == "int":
        return int(rng.integers(lo, hi + 1))
    if kind == "lin":
        return float(rng.uniform(lo, hi))
    if kind == "log0" and rng.random() < 0.2:
        return 0.0
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def valid(g):
    return g["depth"] - g["input_depth"] - g["output_depth"] >= MIN_GAP


def random_genome(rng):
    while True:
        g = {k: sample_gene(rng, k) for k in GENES}
        if valid(g):
            return g


def mutate(rng, g):
    while True:
        out = dict(g)
        keys = list(GENES)
        first = keys[int(rng.integers(len(keys)))]
        for k in keys:
            if k == first or rng.random() < 0.25:
                kind, lo, hi = GENES[k]
                if kind == "int":
                    out[k] = int(np.clip(out[k] + rng.choice([-1, 1]), lo, hi))
                elif kind == "lin":
                    out[k] = float(np.clip(out[k] + rng.normal(0, 0.2 * (hi - lo)), lo, hi))
                elif kind == "log0" and (out[k] == 0.0 or rng.random() < 0.15):
                    out[k] = sample_gene(rng, k)
                else:
                    out[k] = float(np.clip(out[k] * np.exp(rng.normal(0, 0.3)), lo, hi))
        if valid(out) and out != g:
            return out


def config(g, name, seed, train_steps=None):
    """The snake config of genome g: the volume_train recipe with the genome's volume, gains and brake."""
    c = json.load(open(BASE_CFG))
    net = c["network"]
    for k in ("conn", "conn_i"):
        net["fields"][k] = {**net["fields"][k], "mean_scale": net["fields"][k]["mean_scale"] * g["field_gain"],
                            "sd_scale": net["fields"][k]["sd_scale"] * g["field_gain"]}
    net["layers"] = [{"volume": {"depth": g["depth"], "input_depth": g["input_depth"],
                                 "output_depth": g["output_depth"], "sigma_cells": g["sigma_cells"],
                                 "p_max": g["p_max"], "p_global": g["p_global"], "z_shift": g["z_shift"],
                                 "inh_sigma_ratio": g["inh_sigma_ratio"],
                                 "field_sigma_cells": g["field_sigma_cells"]},
                      "internal": {"mean_scale": g["ee_gain"]}, "internal_i": {"mean_scale": g["ii_gain"]}}]
    c["hidden_rule"]["local"]["decorr"] = g["decorr"]
    c.update(seed=seed, csv_prefix=name, selection={**c["selection"], "seed": seed},
             headless={**c["headless"], "seed": seed})
    if train_steps is None:                 # frozen probe
        c.update(lr=0.0, monitor=False, max_timesteps=PROBE_MOVES * das.STEPS_PER_MOVE + 2000,
                 move_log={"probe_every": 10 ** 12, "probe_moves": 16, "activity": PROBE_MOVES - 60})
    else:
        c.update(max_timesteps=int(train_steps), move_log={"probe_every": 10 ** 12, "probe_moves": 16})
    return c


def frozen_metrics(work, name, g):
    act = np.load(f"{work}/outputs/{name}(0)_activity.npz")
    ct = act["ctype"]
    X = act["L1_E"].astype(float).reshape(len(ct), -1, g["depth"])
    back = X[:, :, g["depth"] - g["output_depth"]:].reshape(len(ct), -1)
    _, back_dec, _, back_cnt = analyze_sep.separation(back, ct)
    _, f_dec, f_act, f_cnt = analyze_sep.separation(act["policy_field"].astype(float), ct)
    return {"back_state": float(back_dec), "back_spikes": float(back_cnt), "field_state": float(f_dec),
            "field_spikes": float(f_cnt), "field_active": float(f_act)}


def train_metrics(work, name):
    rows = list(csv.DictReader(open(f"{work}/outputs/{name}(0)_moves.csv")))
    good = np.array([float(r["good"]) > 0.5 for r in rows]); com = np.array([int(float(r["action"])) == 0 for r in rows])

    def gap(sl):
        g_, c_ = good[sl], com[sl]
        return float(c_[g_].mean() - c_[~g_].mean()) if g_.any() and (~g_).any() else 0.0
    late, early = gap(slice(-LATE, None)), gap(slice(0, LATE))
    n = list(csv.DictReader(open(f"{work}/outputs/{name}(0)_neurons.csv")))
    dead = float(n[-1]["policy_field_dead"]) if n else 1.0
    hz = float(n[-1]["L1_E_hz"]) if n else float("nan")
    fit = late + 0.5 * max(late - early, 0.0) - 0.5 * dead
    return {"late_gap": late, "early_gap": early, "field_dead": dead, "e_hz": hz, "decisions": len(rows),
            "fitness": fit}


def clean(work):
    for f in work.glob("*_CODE"):
        shutil.rmtree(f, ignore_errors=True)
    shutil.rmtree(work / "snake_checkpoints_unified", ignore_errors=True)
    for f in (work / "outputs").glob("*_activity*.npz"):
        f.unlink(missing_ok=True)


def evaluate(out, name, g, seed, a, frozen=None):
    rec = {"name": name, "genome": g, "seed": seed, "t0": time.time()}
    if frozen is None:
        work = out / "work" / f"{name}_probe"
        ok = das.run_probe(config(g, f"{name}_probe", seed), work, a.parallel + 1, 3600)
        try:
            frozen = frozen_metrics(work, f"{name}_probe", g) if ok else None
        except Exception as e:                       # e.g. a run that ended before the activity dump
            print(name, "probe metrics failed:", repr(e), flush=True)
            frozen = None
        clean(work)
    rec["frozen"] = frozen
    if (frozen is None or frozen["field_state"] < MIN_FIELD_STATE or frozen["back_spikes"] < 0.05
            or frozen["field_spikes"] < 0.05):
        rec["fitness"] = -1.0 + 0.1 * (frozen or {}).get("field_state", 0.0)
        rec["rejected"] = True
    else:
        work = out / "work" / f"{name}_s{seed}"
        ok = das.run_probe(config(g, f"{name}_s{seed}", seed, a.train_steps), work, a.parallel + 1, 6 * 3600)
        try:
            rec["train"] = train_metrics(work, f"{name}_s{seed}") if ok else None
        except Exception as e:
            print(name, "train metrics failed:", repr(e), flush=True)
            rec["train"] = None
        rec["fitness"] = rec["train"]["fitness"] if rec.get("train") else -1.0
        rec["rejected"] = False
        clean(work)
    rec["seconds"] = round(time.time() - rec.pop("t0"), 1)
    with LOCK:
        with open(out / "results.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    tr = rec.get("train") or {}
    print(f"{name} s{seed} fitness {rec['fitness']:+.3f} | frozen {json.dumps(frozen)} | train "
          f"{json.dumps({k: round(v, 3) for k, v in tr.items()})} ({rec['seconds']:.0f}s)", flush=True)
    return rec


def load(out):
    p = out / "results.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def table(recs):
    """Per genome name: mean fitness over its seeds, the seeds, the genome and its last records."""
    by, genomes = {}, {}
    for r in recs:                  # a name reused for another genome (old resume bug): keep them apart as name+"b"
        g = json.dumps(r["genome"], sort_keys=True)
        n = r["name"]
        while genomes.setdefault(n, g) != g:
            n += "b"
        by.setdefault(n, []).append(r)
    rows = []
    for n, rs in by.items():
        rows.append({"name": n, "fitness": float(np.mean([r["fitness"] for r in rs])), "seeds": [r["seed"] for r in rs],
                     "genome": rs[0]["genome"], "records": rs})
    return sorted(rows, key=lambda r: -r["fitness"])


def report(out, top=12):
    rows = table(load(out))
    print(f"{len(rows)} genomes, {sum(len(r['records']) for r in rows)} evaluations")
    for r in rows[:top]:
        g = r["genome"]; tr = [x.get("train") or {} for x in r["records"]]
        print(f"{r['name']:8s} fit {r['fitness']:+.3f} seeds {r['seeds']} late gaps "
              f"{[round(t.get('late_gap', float('nan')), 2) for t in tr]} dead "
              f"{[round(t.get('field_dead', float('nan')), 2) for t in tr]} | "
              + " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in g.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--parallel", type=int, default=4); ap.add_argument("--train-steps", type=int, default=600000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    if a.report:
        report(out); return
    rng = np.random.default_rng(a.seed + len(load(out)))
    deadline = time.time() + a.hours * 3600
    pending, started = set(), {r["name"] for r in table(load(out))}
    used = [int(r["name"][1:4]) for r in load(out) if r["name"][:1] == "v" and r["name"][1:4].isdigit()]
    counter = [max(used + [0]) + 10]   # past every recorded name (+10: names of killed, unrecorded jobs)

    def next_job():
        """(name, genome, seed): a re-seed of a top genome with one seed, else a new candidate."""
        with LOCK:
            rows = table(load(out))
            for r in rows[:TOP_RESEED]:
                if (len(r["seeds"]) < 2 and r["fitness"] > -0.5 and not r["records"][0].get("rejected")
                        and (r["name"], 1) not in pending):
                    pending.add((r["name"], 1))
                    return r["name"], r["genome"], 1, r["records"][0]["frozen"]
            counter[0] += 1
            name = f"v{counter[0]:03d}"
            seeded = any(r["genome"] == SEED_GENOME for r in rows) or any(n == "seed" for n, _ in pending)
            if not seeded:
                name, g = "seed", dict(SEED_GENOME)
            else:
                parents = [r for r in rows if not r["records"][0].get("rejected")][:8]
                parent = parents[int(rng.integers(len(parents)))]["genome"] if parents else SEED_GENOME
                g = mutate(rng, parent)
            pending.add((name, 0))
            return name, g, 0, None

    def worker(k):
        time.sleep(40 * k)                          # stagger the compiles
        while time.time() < deadline:
            name, g, seed, frozen = next_job()
            try:
                evaluate(out, name, g, seed, a, frozen)
            except Exception as e:
                print(name, "failed:", repr(e), flush=True)
            with LOCK:
                pending.discard((name, seed))
    with ThreadPoolExecutor(a.parallel) as ex:
        for k in range(a.parallel):
            ex.submit(worker, k)
    report(out)


if __name__ == "__main__":
    main()
