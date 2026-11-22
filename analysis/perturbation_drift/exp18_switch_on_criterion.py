"""Experiment 18: performance-triggered task switches (the toy version of snake_switch).

Each network gets a new task whenever its reward per environment step over the last WINDOW steps since its
previous switch reaches CRITERION: the food and wall input channels are exchanged and the four actions are
remapped by a new permutation; each mapping (channel order + action permutation) is drawn uniformly from those
the network has not seen, so a speed-up across tasks cannot come from remembering an earlier mapping (the same
seeded sequence for every network). Each task is as learnable as the
first, but the learned mapping is wrong after a switch, so a rule that keeps its plasticity reaches more
switches in the same budget. Per network we report the number of switches, the environment steps each task took
(a growing sequence means plasticity is being lost), and the fraction of plasticity-dead hidden neurons
(mean psi < PSI_EPS over a block) with the fraction of them that recovered by the next block.
Variants: proposed and gradient_only, each with and without the |delta|-scaled homeostat (c = 5), 5 seeds.
Runtime ~80 min for 60k steps.  Usage: python3 exp18_switch_on_criterion.py [env_steps] [criterion]
"""
import sys, time
import numpy as np
from core import Config, Learner, Variant

CRITERION, WINDOW = -0.06, 2000          # reward per env step; the toy starts near -0.12 and learns to ~-0.03
BLOCK, PSI_EPS = 2500, 0.01
H = dict(homeo_c=5.0, homeo_dscale=True, local_rewards="off")
VARIANTS = [Variant("proposed", 1.0, -2.0, local_rewards="off"), Variant("proposed+homeo", 1.0, -2.0, **H),
            Variant("gradient_only", 0.0, -1.0, local_rewards="off"), Variant("gradient+homeo", 0.0, -1.0, **H)]


class SwitchingForager:
    """Per-network channel swap and action permutation over a GridForager batch."""
    def __init__(self, env, seed=0):
        self.env, self.B = env, env.B
        self.swapped = np.zeros(self.B, bool)
        self.perm = np.tile(np.arange(4), (self.B, 1))
        self.seq_rng = np.random.default_rng(seed)
        self.sequence = []                             # shared sequence of action permutations
        self.n = np.zeros(self.B, int)                 # switches so far, per network

    def _mapping(self, k):
        """k-th mapping of the shared sequence: (channels swapped, action permutation), drawn uniformly from
        the 48 mappings the networks have not seen yet (the start is unswapped + identity)."""
        if not self.sequence:
            self.seen = {(False, (0, 1, 2, 3))}
        while len(self.sequence) <= k:
            if len(self.seen) == 48:
                self.seen = {self.sequence[-1][0:1] + (tuple(self.sequence[-1][1]),)}
            while True:
                m = (bool(self.seq_rng.integers(2)), tuple(int(x) for x in self.seq_rng.permutation(4)))
                if m not in self.seen:
                    break
            self.seen.add(m)
            self.sequence.append((m[0], np.array(m[1])))
        return self.sequence[k]

    def switch(self, mask):
        for b in np.nonzero(mask)[0]:
            self.swapped[b], self.perm[b] = self._mapping(self.n[b])
            self.n[b] += 1

    def rates(self):
        r = self.env.rates().reshape(self.B, -1, 2)
        r[self.swapped] = r[self.swapped][:, :, ::-1]
        return r.reshape(self.B, -1)

    def step(self, act):
        return self.env.step(self.perm[np.arange(self.B), act])


def main(env_steps=60000, criterion=CRITERION):
    cfg = Config()
    L = Learner(cfg, VARIANTS, seeds=[0, 1, 2, 3, 4], rng_seed=0)
    env = SwitchingForager(L.env)
    L.env = env; L.rates = env.rates()
    B = L.B
    win = np.zeros((WINDOW, B)); filled = np.zeros(B, int); since = np.zeros(B, int)
    task_lengths = [[] for _ in range(B)]
    psi_sum = None; nsamp = 0
    dead_hist, rec_hist = [], []
    prev_dead = None
    act0 = L.act

    def act():
        rew, ate, dead = act0()
        i = since % WINDOW
        win[i, np.arange(B)] = rew
        since[:] += 1; filled[:] = np.minimum(filled + 1, WINDOW)
        ready = (filled == WINDOW) & (win.mean(0) >= criterion)
        if ready.any():
            for b in np.nonzero(ready)[0]:
                task_lengths[b].append(int(since[b]))
            env.switch(ready)
            since[ready] = 0; filled[ready] = 0; win[:, ready] = 0
            L.rates = env.rates()
        return rew, ate, dead
    L.act = act

    def hook(l, out, ex):
        nonlocal psi_sum, nsamp
        p = out["psi"][:, :l.lay.N]
        psi_sum = p.astype(np.float64) if psi_sum is None else psi_sum + p
        nsamp += 1
    L.hooks.append(hook)

    t0 = time.time()
    for blk in range(env_steps // BLOCK):
        L.run(BLOCK, block=BLOCK, log=None)
        psibar = psi_sum / nsamp; psi_sum = None; nsamp = 0
        dead = psibar < PSI_EPS
        dead_hist.append(dead.mean(1))
        rec = np.full(B, np.nan)
        if prev_dead is not None:
            for b in range(B):
                if prev_dead[b].any():
                    rec[b] = (~dead[b][prev_dead[b]]).mean()
        rec_hist.append(rec); prev_dead = dead
        line = "  ".join(f"{v.name} sw {env.n[L.var_of == i].mean():.1f} dead {dead.mean(1)[L.var_of == i].mean():.2f}"
                         for i, v in enumerate(VARIANTS))
        print(f"[{(blk + 1) * BLOCK:6d}] {line}  [{time.time() - t0:.0f}s]", flush=True)

    print(f"\ncriterion {criterion} over {WINDOW} steps; {env_steps} env steps; 5 seeds")
    print(f"   {'variant':16s} {'switches':>12s} {'steps per task (mean over seeds; task 1, 2, ...)':>50s}")
    for i, v in enumerate(VARIANTS):
        sel = np.nonzero(L.var_of == i)[0]
        n = env.n[sel]
        k = max(len(task_lengths[b]) for b in sel)
        per_task = []
        for t in range(k):
            vals = [task_lengths[b][t] for b in sel if len(task_lengths[b]) > t]
            per_task.append(f"{np.mean(vals):.0f}({len(vals)})")
        print(f"   {v.name:16s} {n.mean():5.1f} ± {n.std():.1f}   " + " ".join(per_task))
    dh, rh = np.array(dead_hist), np.array(rec_hist)
    print("\nplasticity-dead hidden neurons (mean psi < %.2g), first / middle / last block; recovered per block (mean):"
          % PSI_EPS)
    for i, v in enumerate(VARIANTS):
        sel = L.var_of == i
        m = len(dh) // 2
        print(f"   {v.name:16s} {dh[0, sel].mean():.3f} / {dh[m, sel].mean():.3f} / {dh[-1, sel].mean():.3f}   "
              f"recovered {np.nanmean(rh[1:, sel]):.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60000, float(sys.argv[2]) if len(sys.argv) > 2 else CRITERION)
