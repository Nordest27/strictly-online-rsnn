# Analyses of perturbation-based e-prop

Code for the analyses in the appendices of the paper: what the perturbation-induced drift is, how the
update splits into the drift and the perturbation gradient, its Hebbian structure, and why estimating
the drift is local. Plain numpy, single-threaded (set `OMP_NUM_THREADS=1`).

The mechanism was first identified through a series of small exploratory experiments written and run
with an LLM-based coding assistant. The analyses reported in the paper were re-implemented here as
documented scripts with tests and reproduced from them.

## Contents

| file | what it is |
|---|---|
| `core.py` | `Config`; `GridForager` task; ALIF network with e-prop eligibility, shared synaptic noise and per-synapse noise trace (`advance`, same update order as the ml_genn/GeNN implementation); `Learner` (online actor-critic, AdaBelief every step, hidden rule with a configurable gate or twin-estimated parts); `StaggeredTwins` (antithetic decomposition) |
| `experiments_lib.py` | trained snapshots and controlled noise rollouts from a fixed state (traces tracked only for target synapses) |
| `test_core.py` | invariants the analyses rely on (twin exactness, forward/backward trace identity, noise delivery, gate algebra, Stein identity, lightweight vs full dynamics) |
| `exp1_drift_definition.py` | Appendix D.3: the drift is the weight derivative of the eligibility; Stein check; deterministic surrogate (~2 min) |
| `exp2_decomposition.py` | Appendix E: training with each part of the score; sign of the proposed rule's noise part (~45 min) |
| `exp3_hebbian.py` | Appendix F: factorisation and postsynaptic factor of the drift (~5 min) |
| `exp4_locality.py` | Appendix D: signal-to-noise of the drift estimate vs. number of perturbed neurons (~10 min) |

## Run

```
python3 test_core.py            # ~3 s
python3 exp1_drift_definition.py
python3 exp2_decomposition.py   # optional argument: environment steps (default 20000)
python3 exp3_hebbian.py
python3 exp4_locality.py
```

Each experiment prints exactly the numbers reported in the corresponding table.

## The task and network

A Snake-like forager (5x5 board, food +1, leaving the board -1, four actions, egocentric 5x5 view with
food and wall channels) and a small version of the Snake architecture (36 E + 16 I recurrent ALIF
neurons, policy and value fields of 16 neurons, linear readouts; toroidal fan-in 12; initial weights
mean ±0.1/sqrt(12), inhibitory x3, sd 0.05/sqrt(12); Dale sign at initialisation only). Neuron
parameters, noise (sigma = e^-5), score normalisation (1 - alpha^2), regularisers and optimiser follow
the implementation. A random policy scores about -0.13 reward per environment step. Results vary by
about 0.03 between runs that differ only in their random streams, so variants are compared within one
run.

## Conventions

score `s = kappa * g * pi * xi / sigma^2` with `pi = zbar - beta*epsA` (presynaptic factor of e-prop's
ALIF eligibility), gate `g` (`psi`: original rule; `psi - 2*psibar`: proposed rule) and
`kappa = 1 - alpha^2`; trace `p_t = lambda p_{t-1} + s_t`; the optimiser descends on `sum_t delta_t p_t`.
