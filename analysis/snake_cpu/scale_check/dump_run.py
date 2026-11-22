"""Run Snake (SNAKE_CONFIG) and, at timestep DUMP_AT, save for every synapse group with the diagnostic sums:
AccE, AccG, AccG1, pre/post indices, the postsynaptic PsiBar and the presynaptic neurons' rates (FAvg)."""
import os, sys
sys.path.insert(0, "/home/user/repos/ml_genn/examples/eprop")
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import ml_genn.compilers.compiled_network as cn
DUMP_AT = [int(float(x)) for x in os.environ.get("DUMP_AT", "350000,700000,1050000,1400000").split(",")]
_orig = cn.CompiledNetwork.step_time


def post_var_of(net, pop, name):
    """A weight-update post var of `pop`. GeNN fuses identical post vars of the synapse groups targeting a
    population, so only one of those groups holds the array: try them all."""
    for c, sg in net.connection_populations.items():
        if c.target() is pop and not c.is_feedback and name in sg.post_vars:
            try:
                sg.post_vars[name].pull_from_device()
                return np.asarray(sg.post_vars[name].view).ravel().copy()
            except AttributeError:
                continue
    return None


def rates_of(net, pop):
    return post_var_of(net, pop, "FAvg")


def step_time(self, *a, **k):
    r = _orig(self, *a, **k)
    if self.genn_model.timestep in DUMP_AT:
        out = {}
        for c, sg in self.connection_populations.items():
            if c.is_feedback or "AccE" not in sg.vars:
                continue
            key = f"{c.source().name}->{c.target().name}"
            for v in ("AccE", "AccG", "AccG1"):
                sg.vars[v].pull_from_device(); out[f"{key}|{v}"] = np.asarray(sg.vars[v].values).copy()
            sg.pull_connectivity_from_device()
            out[f"{key}|pre"] = np.asarray(sg.get_sparse_pre_inds()).copy()
            out[f"{key}|post"] = np.asarray(sg.get_sparse_post_inds()).copy()
            rate = rates_of(self, c.source())
            if rate is not None:
                out[f"{key}|rate_pre"] = rate
        ts = self.genn_model.timestep
        np.savez(f"diag_dump_{ts}.npz", **out)
        print(f"DUMPED {len(out)} arrays at timestep {ts}", flush=True)
    return r


cn.CompiledNetwork.step_time = step_time
import snake
snake.train_snake_agent_with_ipc(metrics_q=None, best_run_q=None, random_run_q=None, sigma_q=None)
