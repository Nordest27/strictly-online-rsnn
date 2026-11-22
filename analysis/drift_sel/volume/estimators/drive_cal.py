"""Motor-drive calibration of the arm task on the v057 volume: input scale x field gain, short runs of the rule with
surrogate readouts (rule_sur_motor), activity recorded over the last 40 windows (31 steps each).
Reported per setting: fraction of windows with motor-field spikes, motor-field and deep-sheet (E z4-z7) spikes per
neuron per window, input-sheet (E z0) rate, and the episodes' reach rate (no learning expected this short).
Usage: python drive_cal.py OUT [--steps 30000] [--jobs 4]
"""
import sys, argparse, glob, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm_cpu

GRID = [(i, f) for i in (2.0, 3.0, 4.0) for f in (1.0, 1.5, 2.0)]


def one(out, inp, fg, steps):
    arm_cpu.GENOME, arm_cpu.GENOME_SET, arm_cpu.INPUT_SCALE = "v057", {"field_gain": fg}, inp
    arm_cpu.RECORD = {"every": 0, "activity": 40}
    tag = f"in{inp:g}_fg{fg:g}"
    name = arm_cpu.run_one(out / tag, "rule_sur_motor", 0, steps)
    return tag, out / tag / name


def stats(work):
    f = sorted(glob.glob(str(work / "outputs" / "*_activity_*.npz")))
    if not f:
        return None
    z = np.load(f[-1])
    pf = z["policy_field"].astype(float); E = z["L1_E"].astype(float)
    depth = 8; side = int(round(np.sqrt(E.shape[1] / depth)))
    Es = E.reshape(len(E), side, side, depth)
    ep = list(__import__("csv").DictReader(open(glob.glob(str(work / "outputs" / "*(0).csv"))[0])))
    reach = np.mean([int(x["ep_steps"]) < 1200 for x in ep]) if ep else float("nan")
    return {"windows_with_motor_spikes": float((pf.sum(1) > 0).mean()), "motor_per_neuron": float(pf.mean()),
            "deep_E_per_neuron": float(Es[..., 4:].mean()), "input_E_per_neuron": float(Es[..., 0].mean()),
            "reach": float(reach)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=30000); ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(); out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    # run_one reads the module globals at config time; one setting per process slot, so build configs serially
    import threading
    lock = threading.Lock()

    def job(p):
        with lock:
            arm_cpu.GENOME, arm_cpu.GENOME_SET, arm_cpu.INPUT_SCALE = "v057", {"field_gain": p[1]}, p[0]
            arm_cpu.RECORD = {"every": 0, "activity": 40}
            tag = f"in{p[0]:g}_fg{p[1]:g}"
            work = out / tag / "rule_sur_motor_s0"
            work.mkdir(parents=True, exist_ok=True)
            cfg = arm_cpu.config("rule_sur_motor", "rule_sur_motor_s0", 0, a.steps)
            json.dump(cfg, open(work / "config.json", "w"), indent=1)
        return tag, work

    jobs = [job(p) for p in GRID]
    import os, subprocess

    def run(tw):
        tag, work = tw
        if (work / "done").exists():
            return tag, work
        env = {**os.environ, "PYTHONPATH": str(arm_cpu.WT / "ml_genn"), "SNAKE_CONFIG": str(work / "config.json"),
               "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "BUILD_JOBS": "2"}
        with open(work / "log.txt", "w") as log:
            rc = subprocess.call(["nice", "-n", "5", arm_cpu.PY, "-u", str(arm_cpu.WT / "examples/eprop/run_headless.py")],
                                 cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3 * 3600)
        (work / "done").write_text(str(rc))
        for d in ("snakeEPropCompiler_CODE", "snake_checkpoints_unified"):
            subprocess.call(["rm", "-rf", str(work / d)])
        return tag, work

    with ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(run, jobs))
    print(f"{'setting':14s} {'windows w/ motor spikes':>24s} {'motor/neuron':>13s} {'deep E/neuron':>14s} {'input E/neuron':>15s}")
    for tag, work in results:
        s = stats(work)
        if s is None:
            print(f"{tag:14s} no activity recorded"); continue
        print(f"{tag:14s} {s['windows_with_motor_spikes']:24.2f} {s['motor_per_neuron']:13.2f} {s['deep_E_per_neuron']:14.2f}"
              f" {s['input_E_per_neuron']:15.2f}")


if __name__ == "__main__":
    main()
