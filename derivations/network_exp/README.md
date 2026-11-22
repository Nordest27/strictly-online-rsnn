# Rule-variant experiment (Snake 5×5, original architecture)

Question: why does the live weight-dist rule (λ-trace of the *uncentred* noise score) learn much
better than its slowly varying mean (the "drift") alone? Each variant changes only the line that
feeds `RLNoiseTrace` in `eprop_alif_td_model`:

| variant | `RLNoiseTrace` input | meaning |
|---|---|---|
| `full`   | `e · NormNoiseTrace` (score) | live rule, unchanged |
| `drift`  | `ScoreMean` = EMA of the score (β = `SCORE_BETA`, default 0.99) | drift only |
| `fluct`  | score − `ScoreMean` | fluctuation only |
| `frozen` | score, but output weights (policy/value heads) not optimised | does the noise part need a learning readout? |

New-rule ablations (gate multiplying `(ZFilter - Beta*epsA) * NormNoiseTrace`; PsiBar = EMA of Psi,
decay `PSIBAR_DECAY`, default 0.99). The expected update splits into a drift term (delta x d e/dW, Hebbian-like)
and a gradient term (e x d delta/dW, REINFORCE-like):

| variant | gate / setting | tests |
|---|---|---|
| `new` | `Psi - 2*PsiBar` | proposed rule: drift (code sign) + gradient part (REINFORCE sign) |
| `drift_only` | `Psi - PsiBar` | gradient part removed |
| `grad_only` | `-PsiBar` | drift removed, local REINFORCE kept |
| `new_sig03` / `new_sig3` (and `full_*`) | `LOG_SYN_SIG=-1.2 / 1.1` (noise x0.3 / x3) | drift is noise-level independent, gradient-part noise scales 1/sigma |
| `new_rsign` | TD error with random sign per neuron/step (hidden only) | removes reward information from the hidden update |
| `new_const` | running mean of abs(TD error) (hidden only) | drift without advantage weighting |

| `new_lgrad` | `LOCAL_REWARDS=gradient` | neuron-local rewards (abs dV, VReg, firing-rate reward) x the instantaneous gradient part only (gate `-PsiBar`, no lambda-trace) instead of the whole trace |
| `new_lq` | `LOCAL_REWARDS=q_trace` | local rewards x the lambda-trace of the gradient part only (two traces: TD error uses drift and gradient part, local rewards only the gradient part) |
| `new_lq_dv` | `LOCAL_REWARDS=q_trace DV_MODE=correct` | as `new_lq`, with abs(dV) measured from the previous post-reset voltage (the implemented `abs(PrevV - V)` is the reset jump, i.e. a spike penalty, because PrevV is stored before the reset) |
| `new_loff` | `LOCAL_REWARDS=off` | neuron-local rewards off (e-prop-style firing-rate term kept) |

Readouts always learn with the real TD error. Default `VARIANTS="full new drift_only grad_only new_rsign"`.

Everything else (network, rewards, optimiser, regularisers, AdaBelief 1e-5) is the Snake script as it was.

## Setup
Needs a Python with **pygenn built with CUDA**, plus `numpy scipy opencv-python-headless matplotlib line_profiler`.
Your usual GeNN environment should work. The patched ml_genn in `ml_genn_exp/` is put first on
`PYTHONPATH` by the scripts, so **nothing gets installed and your own ml_genn is untouched**.

```
unzip network_exp.zip && cd network_exp
export PYTHON=/path/to/your/env/bin/python    # if `python` isn't that env
export CUDA_PATH=/usr/local/cuda               # if CUDA lives elsewhere
```

Quick check (about 1 min): `./run_variant.sh smoke 20 RULE_VARIANT=drift && tail -3 runs/smoke/log.txt`.
The log should start with `[experiment] RULE_VARIANT=drift ...`.

## Run
```
./batch_variants.sh                               # 4 variants × 3 seeds × 2500 episodes, one at a time
PARALLEL=4 ./batch_variants.sh                    # if the machine has the memory for 4 at once
SEEDS=5 EPS=3000 VARIANTS="full drift" ./batch_variants.sh
```
Each run gets its own folder `runs/<variant>_s<seed>/` (own GeNN build, `log.txt`, CSV).
Rerunning a name overwrites it. `SCORE_BETA=0.999 ./run_variant.sh drift999_s1 2500 RULE_VARIANT=drift`
runs a single custom variant.

## Results
`python summarise.py` prints the reward rate per 250 episodes per run, and the mean ± s.d. per variant.
To send them back: `zip -r results.zip runs/*/outputs runs/*/log.txt` (CSV + logs only; the
GeNN build folders are large and not needed).

## Files
- `run_variant.sh`, `batch_variants.sh`, `summarise.py`: runner and summary
- `ml_genn_exp/`: copy of ml_genn with the switches (`RULE_VARIANT`, `SCORE_BETA`, `FREEZE_READOUT`
  in `eprop_compiler.py`; board/network size, `LR`, `MAX_EPISODES` env vars in `examples/eprop/snake.py`;
  `examples/eprop/run_headless.py` runs without the viewer windows)
- `eprop_compiler_variants.patch`, `snake_env_config.patch`: the same changes as diffs, for
  applying to a newer ml_genn if the copy here is out of date

## First data (laptop, 1 run each, cut short by a crash)
Reward rate per 250 episodes:

| variant | 0–250 | 250–500 | 500–750 | 750–1000 |
|---|---|---|---|---|
| full   | −0.094 | −0.010 | +0.011 (partial) | |
| drift  | −0.125 | −0.098 | −0.062 | −0.009 |
| fluct  | −0.112 | −0.112 | −0.085 | +0.001 |
| frozen | −0.159 | −0.170 | −0.169 | −0.169 (no learning through 2000) |
