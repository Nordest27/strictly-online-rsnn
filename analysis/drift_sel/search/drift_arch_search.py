"""Architecture search scored by what activity selection needs (frozen networks, the real rule's drift eligibility).

    python drift_arch_search.py OUT_DIR --layers 5 --no-fields [--hours 12] [--max 60] [--moves 800]
    python drift_arch_search.py OUT_DIR --layers 1                       # with the policy / value fields
    python drift_arch_search.py OUT_DIR --report

Per candidate (a genome of architecture_evolution.py, small size):
1. build and calibrate the rates layer by layer on selection-task frames (reservoir_search, inference compiler);
2. run snake.py frozen (lr 0) on the selection task with the proposed rule's diagnostics: `moves` decisions with the
   aggregate drift-eligibility probe and the per-move activity;
3. per stage (a layer: its input and recurrent E -> E synapses; with fields: also last layer E -> policy field):
   same / diff (split-half cosine of the per-state drift eligibility), snr (per-move signal-to-noise of the
   state-specific part), overlap by shared features (raw cosine for 0..3 shared features; slope = c3 - c0);
   per population: state decoding from one move, separation, active fraction, spikes per move;
   behaviour: variability = mean over states of 4 p (1 - p), p = the commit rate (1: undecided, 0: fixed).
Score (higher is better): the weakest stage's log10(snr, floored at 1e-4) + its generalisation slope (0 .. 0.5)
    + 3 x the state decoding of the stage's population (0 .. 1; a stage that no longer carries the state has nothing
    to select, whatever its drift)
    - (1 - variability) - 1 per E population outside 0.1 .. 4 spikes per move.
Every metric is stored in OUT_DIR/results.jsonl (resumable; --report re-ranks).
At most `--slots` run_headless.py processes run on the machine at once (the thermal guard counts as one).
"""
import argparse
import copy
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

EPROP = Path("/home/user/repos/ml_genn/examples/eprop")
HERE = Path(__file__).resolve().parent
PY = "/home/user/repos/personal/derivations/network_exp/.venv/bin/python"
sys.path.insert(0, str(EPROP)); sys.path.insert(0, str(HERE.parent))
STEPS_PER_MOVE = 56


def selection_frames(n, seed=1):
    import snake_switch
    env = snake_switch.SelectionEnv(0, (20, 20, 3), seed=seed)
    out = []
    for _ in range(n):
        env.reset()
        out.append((np.clip(env.obs, 0.0, 1.0) * 0.7).astype(np.float32).ravel())
    return np.array(out)


def calibrated_config(genome, layers, work):
    """Runs in a child process (GeNN model released afterwards): the snake config with calibrated gains."""
    import architecture_evolution as ae
    import reservoir_search as rs
    hp = ae.genome_to_hp(genome, layers, "small")
    built = rs.build(hp, "small", layers)
    rates = selection_frames(1500)
    cwd = os.getcwd()
    with rs.Sim(built, "calib", work) as sim:
        rs.calibrate(sim, rates, genome["target_hz"], log=lambda s: None)
        factors = dict(sim.factors)
    os.chdir(cwd)
    return rs.snake_config(hp, factors, layers, "small")


def probe_config(arch, name, moves, fields):
    base = json.load(open(HERE.parent / "sep/base/config.json"))
    for k in ("toroidal", "local", "network", "ei_layers", "hid_e", "hid_i", "fan_in", "feedforward_targets"):
        base.pop(k, None)
    cfg = {**base, **copy.deepcopy(arch)}
    if not fields:
        cfg["network"]["fields"] = None
    cfg.update(csv_prefix=name, max_timesteps=moves * STEPS_PER_MOVE + 2000, monitor=False, lr=0.0,
               headless={"seed": 0, "decoder": "lse", "beta": 1.0},
               move_log={"probe_every": 10 ** 12, "probe_moves": moves, "aggregate": True, "activity": moves})
    return cfg


def wait_slot(slots):
    def running():
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
        return sum(1 for l in out if l.split()[:1] and "python" in l.split()[0] and "run_headless.py" in l)
    while running() >= slots:
        time.sleep(20)


def run_probe(cfg, work, slots, timeout):
    work.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(work / "config.json", "w"), indent=1)
    env = {**os.environ, "PYTHONPATH": "/home/user/repos/ml_genn/ml_genn", "SNAKE_CONFIG": str(work / "config.json"),
           "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "BUILD_JOBS": "2"}
    import fcntl
    with open(work / "log.txt", "w") as log:
        with open(HERE / "launch.lock", "w") as lock:      # one launcher at a time, so two searches never take one slot
            fcntl.flock(lock, fcntl.LOCK_EX)
            wait_slot(slots)
            proc = subprocess.Popen(["nice", "-n", "5", PY, "-u", str(EPROP / "run_headless.py")], cwd=work, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
            time.sleep(15)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            return False
    return True


def metrics(work, name, layers, fields):
    import csv
    import analyze_agg
    import analyze_overlap
    import analyze_sep
    files = glob.glob(f"{work}/outputs/{name}(0)_moves/agg_*.npz")
    if not files:
        return None
    d = np.load(files[0])
    stages = {}
    names = [(f"L{k}", ["Pop0->L1_E" if k == 1 else f"L{k - 1}_E->L{k}_E", f"L{k}_E->L{k}_E"])
             for k in range(1, layers + 1)]
    if fields:
        names.append(("fields", [f"L{layers}_E->policy_field"]))
    for label, keys in names:
        keys = [k for k in keys if f"{k}|S" in d.files]
        if not keys:
            stages[label] = None
            continue
        S = np.concatenate([d[f"{k}|S"] for k in keys], axis=2).astype(np.float64)
        C = d[f"{keys[0]}|C"]
        SQ = sum(d[f"{k}|SQ"].astype(np.float64) for k in keys)
        _, n, same, diff, _, common, snr = analyze_agg.stats(S, C, SQ)
        raw = analyze_overlap.table(S, C, False)
        stages[label] = {"same": float(same), "diff": float(diff), "snr": float(snr), "common": float(common),
                         "overlap": [float(x) for x in raw], "n_per_type": int(n)}
    act = np.load(f"{work}/outputs/{name}(0)_activity.npz")
    pops = {}
    for k in act.files:
        if k not in ("ctype", "timestep"):
            s, dec, active, cnt = analyze_sep.separation(act[k].astype(np.float64), act["ctype"])
            pops[k] = {"sep": float(s), "decode": float(dec), "active": float(active), "count": float(cnt)}
    rows = list(csv.DictReader(open(f"{work}/outputs/{name}(0)_moves.csv")))
    ct = np.array([int(r["ctype"]) for r in rows]); a = np.array([int(r["action"]) for r in rows])
    p = np.array([(a[ct == t] == 0).mean() for t in range(16) if (ct == t).sum() >= 5])
    return {"stages": stages, "pops": pops, "variability": float(np.mean(4 * p * (1 - p))),
            "commit_rate": float((a == 0).mean())}


def score(m):
    if m is None or any(s is None for s in m["stages"].values()):
        return -99.0, {}
    per = {}
    for label, s in m["stages"].items():
        snr = s["snr"] if np.isfinite(s["snr"]) else 0.0
        slope = s["overlap"][3] - s["overlap"][0] if s["same"] > 0.2 and np.isfinite(s["overlap"][3] - s["overlap"][0]) else 0.0
        pop = m["pops"].get("policy_field" if label == "fields" else f"{label}_E", {})
        # below ~1e-4 the probe cannot resolve the snr: the state decoding of the stage's population keeps a gradient
        per[label] = float(np.log10(max(snr, 1e-4)) + min(max(slope, 0.0), 0.5) + 3.0 * pop.get("decode", 0.0))
    rate_pen = sum(1.0 for k, v in m["pops"].items() if k.endswith("_E") and not 0.1 <= v["count"] <= 4.0)
    parts = {"stages": per, "weakest": min(per, key=per.get), "variability_penalty": 1.0 - m["variability"],
             "rate_penalty": rate_pen}
    return float(min(per.values()) - parts["variability_penalty"] - rate_pen), parts


def load(out):
    p = Path(out) / "results.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def report(out, top=10):
    res = load(out)
    for r in res:
        r["score"], r["parts"] = score(r.get("metrics"))
    res.sort(key=lambda r: -r["score"])
    print(f"{len(res)} candidates ({sum(r['score'] > -90 for r in res)} evaluated)")
    for r in res[:top]:
        m = r["metrics"]
        st = "  ".join(f"{k} snr {v['snr']:.4f} same {v['same']:+.2f} ov {v['overlap'][0]:+.2f}>{v['overlap'][3]:+.2f}"
                       for k, v in m["stages"].items())
        dec = " ".join(f"{k}:{v['decode']:.2f}/{v['count']:.1f}" for k, v in m["pops"].items() if not k.endswith("_I"))
        print(f"{r['name']:8s} score {r['score']:+.2f}  var {m['variability']:.2f}  | {st}\n{'':8s} decode/spikes {dec}")
        print(f"{'':8s} genome " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in (r.get('genome') or {}).items()}))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--no-fields", action="store_true")
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--max", type=int, default=60)
    ap.add_argument("--moves", type=int, default=800)
    ap.add_argument("--random", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slots", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--refs", nargs="*", default=[], help="existing config.json files to evaluate first (references)")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report(a.out); return
    import multiprocessing as mp
    import architecture_evolution as ae
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    fields = not a.no_fields
    rng = random.Random(a.seed)
    done = load(out)
    for r in done:                               # ranked with the current score
        r["score"], r["parts"] = score(r.get("metrics"))
    t_end = time.time() + a.hours * 3600
    ctx = mp.get_context("spawn")
    for i, ref in enumerate(a.refs):
        name = f"ref{i}"
        if any(r["name"] == name for r in done):
            continue
        arch = {k: v for k, v in json.load(open(ref)).items()
                if k in ("toroidal", "local", "network", "ei_layers", "hid_e", "hid_i", "fan_in", "feedforward_targets")}
        done.append(evaluate(out, name, None, arch, a, fields, ref=ref))
    while time.time() < t_end and len(done) < a.max:
        idx = sum(1 for r in done if r["name"].startswith("c"))
        ranked = sorted((r for r in done if r.get("genome") and r["score"] > -90), key=lambda r: -r["score"])
        for _ in range(idx + 1):                 # the same random stream on resume
            pick = rng.random()
            fresh = ae.random_genome(rng, a.layers)
            parent = rng.randrange(5)
            mseed = rng.random()
        if idx < a.random or pick > 0.7 or not ranked:
            genome = fresh
        else:
            genome = ae.mutate(random.Random(mseed), ranked[min(parent, len(ranked) - 1)]["genome"], a.layers)
        name = f"c{idx:03d}"
        work = out / "work" / name; work.mkdir(parents=True, exist_ok=True)
        with ctx.Pool(1) as pool:
            try:
                arch = pool.apply_async(calibrated_config, (genome, a.layers, str(work))).get(timeout=1200)
            except Exception as e:
                arch = None
                err = repr(e)
        if arch is None:
            res = {"name": name, "genome": genome, "metrics": None, "score": -99.0, "error": err}
            with open(out / "results.jsonl", "a") as f:
                f.write(json.dumps(res) + "\n")
            done.append(res); continue
        done.append(evaluate(out, name, genome, arch, a, fields))


def evaluate(out, name, genome, arch, a, fields, ref=None):
    work = out / "work" / name
    t0 = time.time()
    cfg = probe_config(arch, name, a.moves, fields)
    ok = run_probe(cfg, work, a.slots, a.timeout)
    try:
        m = metrics(work, name, cfg["ei_layers"], fields) if ok else None
    except Exception as e:
        m = None
        print(name, "metrics failed:", repr(e), flush=True)
    s, parts = score(m)
    res = {"name": name, "genome": genome, "arch": arch, "metrics": m, "score": s, "parts": parts, "ref": ref,
           "seconds": round(time.time() - t0, 1)}
    with open(out / "results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    for junk in ("snakeEPropCompiler_CODE", "snake_checkpoints_unified", "calib_CODE"):
        shutil.rmtree(work / junk, ignore_errors=True)
    for f in glob.glob(f"{work}/*_CODE"):
        shutil.rmtree(f, ignore_errors=True)
    print(f"{name} score {s:+.2f} {json.dumps(parts)} ({res['seconds']:.0f}s)", flush=True)
    return res


if __name__ == "__main__":
    main()
