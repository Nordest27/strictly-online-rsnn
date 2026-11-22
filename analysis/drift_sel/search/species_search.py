"""Speciated architecture search for the drift + contrast rules, starting from the mexhat at 2 layers.

    python species_search.py OUT_DIR [--hours 6] [--max 80] [--parallel 2] [--moves 1200]
    python species_search.py OUT_DIR --report

Species = depth (number of hidden layers). The search starts with the 2-layer mexhat and mutations of it. Each new
candidate is either a mutation of a good genome of an alive species (same depth), or, with probability
--depth-prob, a copy of a good genome with one more layer, which founds (or joins) the species one deeper. A species
dies once it has --min-evals evaluations and its best score is below --survival times the best score of the species
one shallower (the 2-layer species never dies). The best genome of every species is kept and exported.

Each candidate (small size) is calibrated layer by layer to its target rate on selection-task frames, then probed
frozen on the selection task (drift_arch_search: aggregate drift probe + per-move activity + the decisions), with the
policy field as the motor population (log-sum-exp decoder). Per stage (every hidden layer's E population and the
policy field) three terms, each in [0, 1]:
  - drift: the per-state drift eligibility's signal-to-noise, (log10 snr + 4) / 3 clipped (1e-4 -> 0, 1e-1 -> 1)
    (into the layer: its input and recurrent E -> E synapses; into the field: last layer E -> policy field);
  - state: state decoding from one move (what the contrast's between-state part and the drift need);
  - reach: action decoding within a state, (acc - 0.5) / 0.25 clipped: how much the stage's trial-to-trial
    fluctuations reach the decision (what the contrast's selection part needs).
Stage score = mean of the three; candidate score = the weakest stage's score (depth must carry everything to the
motor field) - (1 - variability) / 2, variability = mean over candidate types of 4 p (1 - p), p the commit rate (an
undecided network at the start). Rates are set by the calibration.

Learning stage (unless --no-train): a candidate that passes the frozen filter (policy field state decoding >=
--min-field-state and variability >= --min-variability) is then trained for --train-steps on the contingency task
(commit rewarded on candidate types 0-7, skip on 8-15; no early end) with drift + contrast reward on the lambda-trace,
centred TD error, homeostat, eligibility centring and decorr 1 (RULE below; no c_reg). Its fitness is
    selectivity over the last 2000 decisions (commit rate on 0-7 minus on 8-15) - mean dead fraction of the hidden E
    populations and the policy field at the end + 0.25 * the frozen score;
candidates that fail the filter keep the frozen score - 1.

Genes: architecture_evolution.SPACE (local connectivity, inhibition, feed-forward and field spread, field size,
target rate, feed-forward inhibition and targets) plus field_ei (fields as E/I layers). All layers share the genes.
"""
import argparse
import copy
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "action"))
import drift_arch_search as das  # noqa: E402

BASE = {"sigma_cells": 0.75, "p_max": 0.791, "input_sigma_cells": 1.857, "inhibition_scale": 7.963, "ff_sigma": 0.6,
        "field_sigma": 1.124, "field_shape": "inhibitory", "target_hz": 20.0, "ff_inhibition": True,
        "ff_targets": "both", "inh_sigma_ratio": 1.891, "field_ei": False}
LOCK = threading.Lock()


def mutate(rng, g):
    import architecture_evolution as ae
    out = ae.mutate(rng, {k: v for k, v in g.items() if k != "field_ei"}, 2)
    out["field_ei"] = (not g.get("field_ei", False)) if rng.random() < 0.15 else g.get("field_ei", False)
    return out


def calibrated(genome, layers, work):
    import multiprocessing as mp
    g = {k: v for k, v in genome.items() if k != "field_ei"}
    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        arch = pool.apply_async(_calib_child, (g, layers, str(work), genome.get("field_ei", False))).get(timeout=1800)
    return arch


def _calib_child(g, layers, work, field_ei):
    import architecture_evolution as ae
    import reservoir_search as rs
    hp = ae.genome_to_hp(g, layers, "small")
    if field_ei:
        hp["field_ei"] = {}
    built = rs.build(hp, "small", layers)
    rates = das.selection_frames(1500)
    cwd = os.getcwd()
    with rs.Sim(built, "calib", work) as sim:
        rs.calibrate(sim, rates, g["target_hz"], log=lambda s: None)
        factors = dict(sim.factors)
    os.chdir(cwd)
    cfg = rs.snake_config(hp, factors, layers, "small")
    if field_ei:
        cfg["field_ei"] = {}
    return cfg


def metrics(work, name, layers):
    import csv
    import analyze_action
    import analyze_sep
    m = das.metrics(work, name, layers, True)
    if m is None:
        return None
    moves = {int(r["timestep"]): int(r["action"]) for r in csv.DictReader(open(f"{work}/outputs/{name}(0)_moves.csv"))}
    a = np.load(f"{work}/outputs/{name}(0)_activity.npz")
    keep = np.array([int(t) in moves for t in a["timestep"]])
    ct, act = a["ctype"][keep], np.array([moves[int(t)] for t in a["timestep"][keep]])
    reach = {}
    for k in a.files:
        if k in ("ctype", "timestep"):
            continue
        acc, nt = analyze_action.action_decoding(a[k][keep].astype(np.float64), ct, act)
        reach[k] = {"action_decode": acc, "types": nt}
    m["reach"] = reach
    return m


def score(m, layers):
    if m is None:
        return -9.0, {}
    stages = {}
    for k in range(1, layers + 1):
        stages[f"L{k}"] = (m["stages"].get(f"L{k}"), f"L{k}_E")
    stages["fields"] = (m["stages"].get("fields"), "policy_field")
    per = {}
    for label, (st, pop) in stages.items():
        snr = st["snr"] if st is not None and np.isfinite(st["snr"]) else 0.0
        drift = float(np.clip((np.log10(max(snr, 1e-6)) + 4.0) / 3.0, 0.0, 1.0))
        state = float(m["pops"].get(pop, {}).get("decode", 0.0))
        acc = m["reach"].get(pop, {}).get("action_decode", float("nan"))
        reach = float(np.clip((acc - 0.5) / 0.25, 0.0, 1.0)) if np.isfinite(acc) else 0.0
        per[label] = {"drift": drift, "state": state, "reach": reach, "stage": (drift + state + reach) / 3.0}
    weakest = min(per, key=lambda k: per[k]["stage"])
    s = per[weakest]["stage"] - 0.5 * (1.0 - m["variability"])
    return float(s), {"stages": per, "weakest": weakest}


# the CPU setup that learns the contingency task within ~100k steps (centred TD error, homeostat, eligibility
# centring, decorr 1, contrast reward on the lambda-trace); the GPU recipe (uncentred, decorr 0.001) did not learn
# the task within 200k steps, so it could not separate architectures
RULE = {"preset": "proposed", "gradient": 0.0, "diagnose_parts": False, "center_drift": True, "homeostat": 1.0,
        "center_eligibility": 0.9999,
        "local": {"route": "GRADIENT_INSTANT", "dv": 0.0, "vreg": 0.0, "rate": 0.0, "decorr": 1.0,
                  "contrast": 1.0, "contrast_route": "GRADIENT_TRACE"}}


def train_eval(work, name, arch, layers, a):
    """Short training on the contingency task; returns (selectivity, dead fraction, rates) or None."""
    import csv
    cfg = das.probe_config(arch, name, 16, True)
    cfg.update(lr=None, c_reg=0.0, hidden_rule=RULE, max_timesteps=int(a.train_steps), monitor=True,
               monitor_every=1500)
    cfg.pop("lr"); cfg.pop("readout_optimiser", None)
    cfg["selection"] = {"n_candidates": 8, "seed": 0, "commit_types": list(range(8)),
                        "skip_types": list(range(8, 16)), "table_reward": 1.0}
    cfg["move_log"] = {"probe_every": 10 ** 12, "probe_moves": 16}
    cfg["hidden_rule"] = copy.deepcopy(RULE)
    tw = work / "train"
    if not das.run_probe(cfg, tw, a.slots, a.timeout):
        return None
    rows = list(csv.DictReader(open(f"{tw}/outputs/{name}(0)_moves.csv")))
    if len(rows) < 500:
        return None
    last = rows[-min(2000, len(rows) // 2):]          # the last 2000 decisions (half the run if it is shorter)
    ct = np.array([int(r["ctype"]) for r in last]); act = np.array([int(r["action"]) for r in last]) == 0
    sel = float(act[ct < 8].mean() - act[ct >= 8].mean())
    nr = list(csv.DictReader(open(f"{tw}/outputs/{name}(0)_neurons.csv")))
    pops = [f"L{k}_E" for k in range(1, layers + 1)] + ["policy_field"]
    dead = float(np.mean([float(nr[-1][f"{p}_dead"]) for p in pops])) if nr else float("nan")
    rates = {p: float(nr[-1][f"{p}_hz"]) for p in pops} if nr else {}
    return {"selectivity": sel, "dead": dead, "rates": rates, "decisions": len(rows)}


def evaluate(out, name, genome, layers, a):
    work = out / "work" / name
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        arch = calibrated(genome, layers, work)
        cfg = das.probe_config(arch, name, a.moves, True)
        ok = das.run_probe(cfg, work, a.slots, a.timeout)
        m = metrics(work, name, layers) if ok else None
    except Exception as e:
        m, arch = None, None
        print(name, "failed:", repr(e), flush=True)
    frozen, parts = score(m, layers)
    s, train = frozen, None
    if not a.no_train and m is not None:
        fs = parts["stages"]["fields"]["state"]
        if fs >= a.min_field_state and m["variability"] >= a.min_variability:
            try:
                train = train_eval(work, name, arch, layers, a)
            except Exception as e:
                print(name, "training failed:", repr(e), flush=True)
            s = (train["selectivity"] - (train["dead"] if np.isfinite(train["dead"]) else 0.0) + 0.25 * frozen
                 if train else frozen - 1.0)
        else:
            s = frozen - 1.0
    res = {"name": name, "layers": layers, "genome": genome, "arch": arch, "metrics": m, "score": s, "frozen": frozen,
           "train": train, "parts": parts, "seconds": round(time.time() - t0, 1)}
    with LOCK:
        with open(out / "results.jsonl", "a") as f:
            f.write(json.dumps(res) + "\n")
    import shutil
    import glob
    for d in glob.glob(f"{work}/*_CODE") + [str(work / "snake_checkpoints_unified")]:
        shutil.rmtree(d, ignore_errors=True)
    w = parts.get("weakest")
    tr = f" | trained: sel {train['selectivity']:+.2f} dead {train['dead']:.2f}" if train else " | not trained"
    print(f"{name} L={layers} score {s:+.3f} (frozen {frozen:+.3f}{tr}) weakest {w} "
          + " ".join(f"{k}:{v['stage']:.2f}(d{v['drift']:.2f}/s{v['state']:.2f}/r{v['reach']:.2f})"
                     for k, v in parts.get("stages", {}).items()) + f" ({res['seconds']:.0f}s)", flush=True)
    return res


def load(out):
    p = Path(out) / "results.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def species_table(results, min_evals, survival):
    by = {}
    for r in results:
        by.setdefault(r["layers"], []).append(r)
    alive, best = {}, {}
    for L in sorted(by):
        rs_ = sorted(by[L], key=lambda r: -r["score"])
        best[L] = rs_[0]
        alive[L] = True
        if L > 2 and len(rs_) >= min_evals and (L - 1) in best:   # too far behind the species one shallower: dies
            ref = best[L - 1]["score"]
            alive[L] = rs_[0]["score"] >= (survival * ref if ref > 0 else ref - 0.1)
    return by, alive, best


def report(out):
    res = load(out)
    by, alive, best = species_table(res, 6, 0.7)
    print(f"{len(res)} candidates")
    for L in sorted(by):
        b = best[L]
        tr = b.get("train") or {}
        print(f"== species {L} layers: {len(by[L])} evaluated, {'alive' if alive[L] else 'extinct'}, best {b['name']} "
              f"score {b['score']:+.3f} (frozen {b.get('frozen', b['score']):+.3f}, trained selectivity "
              f"{tr.get('selectivity', float('nan')):+.2f}, dead {tr.get('dead', float('nan')):.2f}) "
              f"weakest {b['parts'].get('weakest')}")
        for k, v in b["parts"].get("stages", {}).items():
            print(f"     {k:7s} stage {v['stage']:.2f}  drift {v['drift']:.2f}  state {v['state']:.2f}  reach {v['reach']:.2f}")
        print("     genome " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in b["genome"].items()}))
        if b.get("arch"):
            json.dump(b["arch"], open(Path(out) / f"best_L{L}.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--max", type=int, default=80)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--moves", type=int, default=1200)
    ap.add_argument("--depth-prob", type=float, default=0.25)
    ap.add_argument("--min-evals", type=int, default=6, help="evaluations before a species can die")
    ap.add_argument("--survival", type=float, default=0.7)
    ap.add_argument("--max-layers", type=int, default=6)
    ap.add_argument("--slots", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-train", action="store_true", help="frozen metrics only")
    ap.add_argument("--train-steps", type=float, default=200000)
    ap.add_argument("--min-field-state", type=float, default=0.2)
    ap.add_argument("--min-variability", type=float, default=0.3)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    if a.report:
        report(out); return
    rng = random.Random(a.seed)
    t_end = time.time() + 3600 * a.hours
    pending = []
    n_started = len(load(out))
    with ThreadPoolExecutor(a.parallel) as ex:
        while time.time() < t_end and n_started < a.max:
            pending = [f for f in pending if not f.done()]
            if len(pending) >= a.parallel:
                time.sleep(10); continue
            res = load(out)
            by, alive, best = species_table(res, a.min_evals, a.survival)
            if not res and not pending:
                genome, layers = dict(BASE), 2                  # the 2-layer mexhat itself first
            else:
                live = [L for L in (by or {2: []}) if alive.get(L, True)] or [2]
                weights = np.array([max(best[L]["score"], 0.01) if L in best else 0.01 for L in live])
                L = live[int(rng.choices(range(len(live)), weights=weights)[0])] if res else 2
                pool = sorted(by.get(L, []), key=lambda r: -r["score"])[:5] or [{"genome": dict(BASE)}]
                parent = pool[rng.randrange(len(pool))]["genome"]
                if L < a.max_layers and len(by.get(L, [])) >= 3 and rng.random() < a.depth_prob:
                    genome, layers = dict(parent), L + 1           # one more layer: a new species (or joins it)
                else:
                    genome, layers = mutate(rng, parent), L
            name = f"c{n_started:03d}_L{layers}"
            n_started += 1
            pending.append(ex.submit(evaluate, out, name, genome, layers, a))
            time.sleep(30)
    report(out)


if __name__ == "__main__":
    main()
