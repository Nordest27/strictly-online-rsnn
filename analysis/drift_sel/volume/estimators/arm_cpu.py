"""CPU feasibility runs of the two-link arm task on the small v031 network (6x6x8 volume, 4x4x3 fields), from the
worktree ml_genn. Variants: rule (the v031 rule: drift + contrast, no policy), rule_drive (input scale x 2), eprop_sur
(surrogate e-prop, "hidden_rule": "eprop", surrogate_sigma 0.3), rule_sur_motor (rule_drive below, the motor pools
learn only from the surrogate's e-prop term: drift, gradient and local objectives 0 on them).
Usage: python arm_cpu.py OUT --variants rule eprop_sur --seeds 0 1 [--steps 600000] [--jobs 4]
       python arm_cpu.py OUT --summary
"""
import sys, json, os, argparse, subprocess, csv, glob
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent / "search")); sys.path.insert(0, str(HERE.parent.parent))
import volume_search as vs
from slab_compare import genome

PY = "/home/user/repos/personal/derivations/network_exp/.venv/bin/python"
WT = Path("/tmp/claude-1000/-home-user-repos-personal/443153b0-266e-4142-add4-6a0e6900d11b/scratchpad/mlg_wt")


GENOME = "v031"
GENOME_SET = {}    # genome overrides, e.g. {"field_gain": 1.5}
INPUT_SCALE = 2.0  # input drive factor of the *_drive / rule_sur_motor variants
ARM_SET = {}       # arm task overrides, e.g. {"joint_speed": 0.02, "max_steps": 300}
HEAD_SET = {}      # headless overrides applied last, e.g. {"surrogate_rho": 0.95}
TOP_SET = {}       # top-level config overrides applied last, e.g. {"episodic_traces": false}
RECORD = None      # {"every": 50, "activity": 150}: arm trajectories every N episodes + activity windows (start, end)


def config(variant, name, seed, steps):
    c = vs.config({**genome(GENOME), **GENOME_SET}, name, seed, steps)
    c["selection"] = None
    c["arm"] = {"seed": seed, **ARM_SET}
    if RECORD:
        c["arm"]["record_every"] = int(RECORD["every"])
        n = int(RECORD["activity"])
        c["move_log"] = {**(c.get("move_log") or {}), "activity": n,
                         "activity_at": [0, max(0, steps - (n + 40) * 31)]}
    c["headless"] = {"seed": seed, "decoder": "joints", "gain": 3.0}
    if variant == "rule_drive":
        net = c.setdefault("network", {})
        inp = dict(net.get("input") or {})
        inp["mean_scale"] = INPUT_SCALE * float(inp.get("mean_scale", 0.15))
        inp["sd_scale"] = INPUT_SCALE * float(inp.get("sd_scale", 0.075))
        net["input"] = inp
    elif variant in ("rule_sur_motor", "rule_sur_motor_s01"):    # the rule below; surrogate-trained motor readouts
        net = c.setdefault("network", {})
        inp = dict(net.get("input") or {})
        inp["mean_scale"] = INPUT_SCALE * float(inp.get("mean_scale", 0.15))
        inp["sd_scale"] = INPUT_SCALE * float(inp.get("sd_scale", 0.075))
        net["input"] = inp
        c["headless"]["surrogate_sigma"] = 0.1 if variant.endswith("s01") else 0.3
        c["headless"]["motor_rule"] = {"eprop": 1.0, "drift": 0.0, "gradient": 0.0, "local": None,
                                       **({"eprop_center_td": True} if variant.endswith("s01") else {})}
    elif variant in ("eprop_sur_ctd", "eprop_sur_ctd_s01", "eprop_ctd_s01_drive"):   # centred e-prop (both parts with delta - its mean)
        c["headless"]["surrogate_sigma"] = 0.1 if "s01" in variant else 0.3
        c["hidden_rule"] = {"preset": "eprop", "eprop_center_td": True}
        c["c_reg"] = 1e-4                      # e-prop's usual firing-rate regulariser (the v031 configs set 0)
        if variant.endswith("drive"):
            net = c.setdefault("network", {}); inp = dict(net.get("input") or {})
            inp["mean_scale"] = INPUT_SCALE * float(inp.get("mean_scale", 0.15)); inp["sd_scale"] = INPUT_SCALE * float(inp.get("sd_scale", 0.075))
            net["input"] = inp
    elif variant == "eprop_readout":            # e-prop with its own learned readout: a Gaussian policy (no headless)
        c["hidden_rule"] = {"preset": "eprop", "eprop_center_td": True}
        c["c_reg"] = 1e-4
        net = c.setdefault("network", {}); inp = dict(net.get("input") or {})
        inp["mean_scale"] = INPUT_SCALE * float(inp.get("mean_scale", 0.15)); inp["sd_scale"] = INPUT_SCALE * float(inp.get("sd_scale", 0.075))
        net["input"] = inp
        c["arm_readout"] = {"sigma": 0.1, "gain": 1.0}
    elif variant == "rule_readout":             # the rule (genome's hidden rule) with the same learned Gaussian readout
        net = c.setdefault("network", {}); inp = dict(net.get("input") or {})
        inp["mean_scale"] = INPUT_SCALE * float(inp.get("mean_scale", 0.15)); inp["sd_scale"] = INPUT_SCALE * float(inp.get("sd_scale", 0.075))
        net["input"] = inp
        c["arm_readout"] = {"sigma": 0.1, "gain": 1.0}
    elif variant == "eprop_sur":
        c["headless"]["surrogate_sigma"] = 0.3
        c["hidden_rule"] = "eprop"
    elif variant != "rule":
        raise KeyError(variant)
    c["headless"] = None if variant in ("eprop_readout", "rule_readout") else {**c["headless"], **HEAD_SET}
    c.update(TOP_SET)
    return c


def run_one(out, variant, seed, steps):
    name = f"{variant}_s{seed}"
    work = out / name
    if (work / "done").exists():
        return name
    work.mkdir(parents=True, exist_ok=True)
    json.dump(config(variant, name, seed, steps), open(work / "config.json", "w"), indent=1)
    env = {**os.environ, "PYTHONPATH": str(WT / "ml_genn"), "SNAKE_CONFIG": str(work / "config.json"),
           "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "BUILD_JOBS": "2"}
    with open(work / "log.txt", "w") as log:
        rc = subprocess.call(["nice", "-n", "5", PY, "-u", str(WT / "examples/eprop/run_headless.py")], cwd=work,
                             env=env, stdout=log, stderr=subprocess.STDOUT, timeout=4 * 3600)
    (work / "done").write_text(str(rc))
    for d in ("snakeEPropCompiler_CODE", "snake_checkpoints_unified"):
        subprocess.call(["rm", "-rf", str(work / d)])
    return name


def summary(out):
    for work in sorted(p for p in out.iterdir() if p.is_dir()):
        mv = glob.glob(str(work / "outputs" / "*_moves.csv"))
        ep = [f for f in glob.glob(str(work / "outputs" / "*(0).csv"))]
        if not mv or not ep:
            print(f"{work.name}: no output yet"); continue
        r = list(csv.DictReader(open(mv[0])))
        e = list(csv.DictReader(open(ep[0])))
        sc = np.array([float(x["score"]) for x in e])
        q = max(1, len(sc) // 4)
        cs = np.array([float(x["cos"]) for x in r]); pools = np.array([[float(x[f"pool{i}"]) for i in range(4)] for x in r])
        n = max(1, len(r) // 4)
        good = lambda a: float(np.mean(a > np.cos(np.pi / 4)))
        print(f"{work.name}: episodes {len(sc)}  score first/last quarter {sc[:q].mean():+.3f} / {sc[-q:].mean():+.3f}"
              f"  toward target first/last {good(cs[:n]):.2f} / {good(cs[-n:]):.2f}"
              f"  moves with motor spikes first/last {np.mean(pools[:n].sum(1) > 0):.2f} / {np.mean(pools[-n:].sum(1) > 0):.2f}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out")
    ap.add_argument("--variants", nargs="+", default=["rule", "rule_drive", "eprop_sur"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1]); ap.add_argument("--steps", type=int, default=600_000)
    ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--summary", action="store_true")
    ap.add_argument("--arm", default="{}", help='JSON arm overrides, e.g. {"joint_speed": 0.02, "max_steps": 300}')
    ap.add_argument("--input_scale", type=float, default=2.0)
    ap.add_argument("--headless", default="{}", help='JSON headless overrides, e.g. {"surrogate_rho": 0.95}')
    ap.add_argument("--set", dest="top", default="{}", help='JSON top-level overrides, e.g. {"episodic_traces": false}')
    ap.add_argument("--genome_set", default="{}", help='JSON genome overrides, e.g. {"field_gain": 2.0}')
    ap.add_argument("--genome", default="v031"); ap.add_argument("--record", type=int, nargs=2, default=None,
                                                                 metavar=("EVERY", "ACTIVITY"))
    a = ap.parse_args(); out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    global GENOME, RECORD
    GENOME = a.genome
    global ARM_SET, INPUT_SCALE, GENOME_SET
    ARM_SET, INPUT_SCALE, GENOME_SET = json.loads(a.arm), a.input_scale, json.loads(a.genome_set)
    global HEAD_SET
    HEAD_SET = json.loads(a.headless)
    global TOP_SET
    TOP_SET = json.loads(a.top)
    RECORD = {"every": a.record[0], "activity": a.record[1]} if a.record else None
    if a.summary:
        summary(out); return
    jobs = [(v, s) for s in a.seeds for v in a.variants]
    with ThreadPoolExecutor(a.jobs) as ex:
        for name in ex.map(lambda j: run_one(out, j[0], j[1], a.steps), jobs):
            print(name, "done", flush=True)
    summary(out)


if __name__ == "__main__":
    main()
