"""Coherent-noise lab on a small real network: the v031 volume (6x6 sheet, depth 8, ~290 E neurons, 4x4 fields) on the
headless selection task (8 candidates, commit / skip), 600k steps (~13 min single-threaded), with the real rule.
Base rule: drift + G (gradient 1) + contrast ("dgc"); "dc" = drift + contrast (gradient 0, the earlier base).
Variants add network.local_noise (fixed Gaussian coherence on the sheet; "inject": the injected noise is the
neighbourhood mixture, unit variance; "credit": own noise injected, the neighbourhood mean credited).
Usage: python coh_lab.py OUT_DIR --variants dgc inj05 ... --seeds 0 1 2 3 [--ml_genn PATH] [--jobs 4]
Metrics per run (volume_search.train_metrics): late_gap (good - bad commit rate, late), early_gap, field_dead, e_hz.
"""
import sys, json, os, argparse, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent / "search")); sys.path.insert(0, str(HERE.parent.parent))
import volume_search as vs
from slab_compare import genome

PY = "/home/user/repos/personal/derivations/network_exp/.venv/bin/python"
DGC = {"hidden_rule": {"gradient": 1.0}}


def local(sigma, mode):
    rule = {"gradient": 1.0, "pool_noise": True}
    if mode == "inject":
        rule["pool_noise_inject"] = True
    return {"hidden_rule": rule, "network": {"local_noise": {"sigma": sigma, "mode": mode}}}


VARIANTS = {
    "dc": {},
    "dgc": DGC,
    "inj05": local(0.5, "inject"), "inj08": local(0.8, "inject"), "inj12": local(1.2, "inject"),
    "cred08": local(0.8, "credit"),
}


TASKS = {"default": {}, "easy": {"selection": {"required": [0]}}}   # easy: good iff feature 0 is present


def run_one(out, ml_genn, vname, seed, steps, task="default"):
    name = f"{vname}_s{seed}" if task == "default" else f"{task}_{vname}_s{seed}"
    work = out / name
    if (work / "metrics.json").exists():
        return name, json.load(open(work / "metrics.json"))
    c = vs.config(genome("v031"), name, seed, steps)
    for key, upd in list(VARIANTS[vname].items()) + list(TASKS[task].items()):
        c[key] = {**(c.get(key) or {}), **upd}
    work.mkdir(parents=True, exist_ok=True)
    json.dump(c, open(work / "config.json", "w"), indent=1)
    env = {**os.environ, "PYTHONPATH": str(ml_genn / "ml_genn"), "SNAKE_CONFIG": str(work / "config.json"),
           "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "BUILD_JOBS": "2"}
    with open(work / "log.txt", "w") as log:
        rc = subprocess.call(["nice", "-n", "5", PY, "-u", str(ml_genn / "examples/eprop/run_headless.py")], cwd=work,
                             env=env, stdout=log, stderr=subprocess.STDOUT, timeout=4 * 3600)
    m = vs.train_metrics(work, name) if rc == 0 else None
    json.dump(m, open(work / "metrics.json", "w"))
    vs.clean(work)
    return name, m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out")
    ap.add_argument("--variants", nargs="+", default=["dgc"]); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--steps", type=int, default=600_000); ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--task", default="default", choices=sorted(TASKS))
    ap.add_argument("--ml_genn", default="/tmp/claude-1000/-home-user-repos-personal/443153b0-266e-4142-add4-6a0e6900d11b/scratchpad/mlg_wt")
    a = ap.parse_args(); out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    jobs = [(v, s) for v in a.variants for s in a.seeds]
    with ThreadPoolExecutor(a.jobs) as ex:
        for name, m in ex.map(lambda j: run_one(out, Path(a.ml_genn), j[0], j[1], a.steps, a.task), jobs):
            print(name, json.dumps(m), flush=True)


if __name__ == "__main__":
    main()
