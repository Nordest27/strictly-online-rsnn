# Autonomous work log (2026-10-29, while you were away)

This file is the running log of the work done while you were away.

## Goal

Find a neuron model with complex, memory-focused internal dynamics where e-prop's eligibility (ψ × a single decaying input trace) is a poor estimate of the true weight derivative of the neuron's spiking, while our rule (Stein through the real dynamics) is unaffected.

Order, as you asked: oscillation (resonate-and-fire) first, then a bistable plateau (persistent current).

## Method (toy: `ml_genn/examples/eprop/drift_toy.py`)

- Recorded network inputs (`rec_L1`, `rec_L5`) are replayed through one modified neuron. Firing rates are matched to the standard neuron with an input gain.
- e-prop eligibility, summed over 200-step windows, is compared against the true dE[spike]/dW. The true derivative comes from finite differences with common noise.
  - **single trace:** the unknown-model case.
  - **derived trace:** the known-model case, where the eligibility vector is propagated with the neuron's own Jacobian.
- Reference for the standard neuron (from the EIF test): e-prop vs true derivative, window sums, corr 0.84 (L1) and 0.66 (L5).

## Status

- [running] resonator, period 25 ms. Rate-matched gains: L1 2.8, L5 1.7. Script `toy_resonator.py`.
  - Output: background task b3egysz6u.
- [queued after the resonator] plateau neuron (`memory_queue.sh` → `memory_results.txt`; waiter notifies when "PLATEAU DONE").
  - Parameters: gp 0.08, θ_p 0.3, s 0.05, τ 30.
  - Rate-matched gains: L1 0.5, L5 0.55.
  - At these settings the plateau is on 41–67% of the time and switches in 51–75% of windows.
  - Script `toy_memory.py` (shared with the resonator: `toy_memory.py L resonator PERIOD GAIN` or `toy_memory.py L plateau GAIN`).

- [running] model-free check: our Stein estimate of dE[spike]/dW (spike × the synapse's own noise trace / σ²) vs the truth, next to e-prop's single trace.
  - Models: standard, resonator, plateau. Script `toy_memory_stein.py`.
  - Queue `memory_stein_queue.sh` → `memory_stein_results.txt`; waiter notifies on "STEIN DONE".
  - Caveat: the noise trace weights older inputs by αᵏ, while the true weight derivative weights them equally.

## Results

### Resonator (period 25 ms, rate-matched: L1 68 Hz, L5 35 Hz)

e-prop eligibility vs the true dE[spike]/dW, summed over 200-step windows; n ≈ 11–12k synapse-windows.

| | L1 corr / Spearman / sign | L5 corr / Spearman / sign |
|---|---|---|
| standard neuron (reference) | 0.84 / 0.81 / – | 0.66 / 0.66 / – |
| resonator, single trace (unknown model) | 0.66 / 0.64 / 0.75 | 0.47 / 0.42 / 0.48 |
| resonator, derived trace (known model) | 0.69 / 0.67 / 0.76 | 0.51 / 0.48 / 0.49 |

**Reading:**
- The resonator clearly degrades e-prop's eligibility (0.84 → 0.66, 0.66 → 0.47).
- Knowing the model (the derived rotation trace) barely helps (+0.03–0.04). So the loss comes from e-prop's structural approximation, the dropped paths through the neuron's own spikes and reset, and not from missing information.
- In a resonator the reset kicks the oscillation, which changes later spiking; e-prop ignores this even with the full model. This is the "fails because of its assumptions" case, as a toy measure of eligibility accuracy, not a learning result.
- Caveat: the single trace is non-negative by construction, so its sign agreement is just the share of positive true derivatives.

### Plateau neuron (gp 0.08, θ_p 0.3, s 0.05, τ 30; rate-matched: L1 67 Hz, L5 33 Hz)

| | L1 corr / Spearman / sign | L5 corr / Spearman / sign |
|---|---|---|
| standard neuron (reference) | 0.84 / 0.81 / – | 0.66 / 0.66 / – |
| plateau, single trace (unknown model) | 0.72 / 0.70 / 0.63 | 0.59 / 0.66 / 0.53 |
| plateau, derived trace (known model) | 0.64 / 0.66 / 0.63 | 0.60 / 0.65 / 0.53 |

**Reading:**
- The plateau degrades e-prop less than the resonator does (L1 0.84 → 0.72, L5 roughly unchanged by rank).
- Knowing the model again does not help (slightly worse in L1). The derived trace's state-dependent sigmoid slope adds variance without fixing the dropped spike/reset paths.

### Model-free (Stein) comparison: our estimate vs e-prop's eligibility, both against the true derivative

**Setup:**
- Our estimate: the spike × the synapse's own noise trace / σ², summed over the window, 64 rollouts per window.
- Truth: finite differences, 8 pairs per window.
- 30 synapses × 256 windows per case.
- Script `toy_memory_stein.py`; output `memory_stein_results.txt`.

| | ours: noise-corrected corr per synapse-window | ours: Spearman per synapse | e-prop: noise-corrected corr per synapse-window | e-prop: Spearman per synapse |
|---|---|---|---|---|
| L5 standard | **0.85** | **0.92** | 0.40 | 0.83 |
| L5 resonator | **0.79** | **0.92** | 0.44 | 0.77 |
| L5 plateau | 0.33 (our repeatability only 0.28) | **0.91** | 0.55 | 0.85 |
| L1 standard | 0.97 | **0.99** | 0.96 | 0.77 |
| L1 resonator | **0.99** | **0.95** | 0.88 | 0.72 |
| L1 plateau | 0.94 | **0.97** | 0.95 | 0.85 |

**Reading:**
- **Bias:** the noise-based estimate, with no model, tracks the true weight derivative of the neuron's spiking as well as or better than e-prop's eligibility in almost every case.
  - It is clearly better by rank per synapse (0.91–0.99 against 0.72–0.85). e-prop's eligibility is non-negative by construction and misses the negative effects that run through the neuron's own spikes and resets.
  - The gap is largest for the resonator and in L5.
- **Variance:** our per-window estimate is noisy where neurons are quiet (L5 plateau repeatability 0.28). The rule needs time-averaging; e-prop's eligibility is deterministic.
  - So this is a bias-vs-variance comparison: e-prop is low-variance but biased (more so for complex dynamics); ours is unbiased (up to the αᵏ kernel) but noisy.
- **Caveats:**
  - This measures gradient-estimate accuracy in an open-loop toy, not learning.
  - "Ours" here is the first-order derivative of spiking (the gradient-part analogue), not the drift.
  - e-prop's eligibility here omits the −β·ε^a adaptation term (β = 0.0174, small).

## Summary

1. Modifying LIF (shifted thresholds, nonlinear leak, input saturation, non-integrating refractoriness, exponential upswing) does not hurt e-prop when it has correct local information.
2. A resonator degrades e-prop's eligibility clearly, and a derived (known-model) trace does not fix it. The cause is its structural approximation: the dropped paths through the neuron's own spikes and resets. The plateau degrades it a little.
3. The model-free noise estimate is less biased than e-prop's eligibility in all of these neurons, at the price of variance.
4. A possible paper statement: "e-prop's eligibility misses spike- and reset-mediated effects, which grow with the neuron's internal dynamics (resonators); a noise-based estimate captures them without a model, at the cost of variance that the rule averages out over time."

## Resonator learning test in Snake (started)

**Code:**
- New neuron `ml_genn.neurons.ResonatorALIF(period, gain)`: a resonate-and-fire membrane, applied by the e-prop compiler after its other edits (`neuron_logic.apply_resonator`).
- Snake option `"resonator": {"period": 25, "gain": 2.8}`.
- Both rules keep their standard traces: the unknown-model condition.

**Tests:** zero rotation reproduces the standard network bit for bit (voltages and learned weights, both rules); 25 ms oscillates. 79 passed.

**Runs** (`jobs_resonator.txt`; 2-layer network, 2M steps, 4 CPU jobs under the thermal guard):
- `blackbox/res_ours_s0`, `res_ours_s1`: current default rule.
- `blackbox/res_eprop_s0`, `res_eprop_s1`: best e-prop.
- References: `blackbox/bb_ours_h0_s*`, `bb_eprop_h0_s*` (standard neurons, same configs).
- The waiter notifies when the processes exit.

### Result, seeds 0–1 (2M steps, 2-layer network; score per 500k steps)

| | final bin s0 / s1 | E rate L1/L2 |
|---|---|---|
| e-prop, standard neurons | 7.21 / 5.82 | 55–66 Hz |
| **e-prop, resonator** | **0.04 / −0.55** (collapse) | **117–134 Hz** |
| ours, standard neurons | −0.57 (never learned) / 3.34 | 105–130 Hz |
| **ours, resonator** | **0.66 / 0.44** (learns, slowly) | 104–124 Hz |

**Reading:**
- First clear separation: with resonator neurons e-prop collapses and its rates double; our rule still learns.
- Caveats:
  - 2 seeds.
  - e-prop's rate regulariser uses the same eligibility, so its rate control fails too and part of the collapse may be rate-driven.
  - One gain (2.8).
- Running now: seeds 2–3 for both arms (`jobs_resonator2.txt`); the waiter notifies when they exit.
- Not done (needs a decision): a known-model control, e-prop with the derived rotation trace, in GeNN. The toy predicts it would barely help.

### Result, all 4 seeds (gain 2.8)

| | final 500k-step bin, seeds 0–3 | mean |
|---|---|---|
| ours, resonator | 0.66, 0.44, 0.50, 0.58 (rates 104–129 Hz) | **0.55** (every seed learns) |
| e-prop, resonator | 0.04, −0.55, −0.30, −0.36 (rates 117–147 Hz) | **−0.29** (every seed fails) |

The separation is robust across seeds.
- Running now: the rate control, both rules at input gain 1.6 (lower rates), 2 seeds each (`jobs_resonator_g16.txt`).
- If e-prop recovers at the lower gain, its collapse is mainly rate-driven; if not, it is the learning signal itself.

### Rate control (gain 1.6, 2 seeds)

| | final bin s0 / s1 | E rate L1/L2 |
|---|---|---|
| ours | 0.22 / 1.43 | 125/92, 106/77 Hz |
| e-prop | −0.11 / −0.33 | 136/117, 132/124 Hz |

**Reading:**
- e-prop still fails at the lower gain and still reaches ~130 Hz. Its high rates are not caused by the input gain: with resonator neurons it drives itself there, and its rate regulariser (which uses the same eligibility) cannot stop it.
- So the rate problem is part of e-prop's failure mechanism, not an external confound.
- Our rule learns at both gains.
- A cleaner separation (capping e-prop's rates by a mechanism that does not use its eligibility) was not done.

## Overall (resonator neurons, 2-layer Snake)

Our rule learns in all 6 seeds across both gains (final bins 0.22–1.43). e-prop fails in all 6 (−0.55 to 0.04), with rates ~130 Hz. With standard neurons, under the same settings, e-prop scores 6–7 and ours 3.3 in its good seed. This is the first setting where the perturbation rule beats e-prop. It matches the toy finding that resonators break e-prop's eligibility through its dropped spike/reset paths.

## Suggested next steps (not started)

- Known-model control: e-prop with the derived rotation trace in GeNN, to show the failure is structural in learning too (the toy says yes).
- Rate cap for e-prop that does not use its eligibility, to separate learning from rate control.
- A GPU run on your real architecture with resonator neurons.
- Rebuild the zip with today's refractory fixes. Not done: waiting for your OK.

## Plateau (memory) neuron in Snake's memory task (started)

**Why:** the resonator showed e-prop *can* fail, but it is a worse model for both rules. The question now is whether a neuron with useful internal memory helps on a task that needs memory, and which rule can exploit it.

**Code:**
- `ml_genn.neurons.PlateauALIF(g, theta_p, s, tau_p, gain)`: a bistable persistent current, `P += (sigmoid((V-θp)/s) - P)/τp`, `V = αV + g·P_old + gain·I`.
  - Applied by `neuron_logic.apply_plateau` after the compiler's own edits.
  - Snake option `"plateau": {...}`.
- Tests: g=0, gain=1 is bit-exact with the standard neuron (voltages and learned weights, both rules); g>0 changes the dynamics. 82 passed, 2 skipped.
- Task: `"memory": {"visible_moves": 3}`, where the apple disappears 3 moves after spawning.
- Note: the thermal guard was found NOT running and was restarted (log `thermal_guard.log`).

**Calibration smoke:** `memory/smoke_std`, `smoke_pl0.5`, `smoke_pl0.7` (ours, 60k steps), to rate-match the plateau gain.
- Smoke (60k steps, rates still climbing): standard 43 Hz, plateau gain 0.5 → 33 Hz, gain 0.7 → 35 Hz. The monitor CSV does not flush in such short runs.
- **Main runs** use gain 1.0, so the plateau neuron is exactly the standard neuron plus the persistent current, the cleanest comparison of the plateau's effect.
  - `memory/mem_{std,pl}_{ours,eprop}_s{0,1}`, 2M steps, configs as the resonator runs plus `"memory": {"visible_moves": 3}`.
  - 4 parallel lanes (s0 then s1) under the thermal guard. Waiter task bjx7ej9sp prints "MEMORY RUNS DONE".
  - Caution: `grep run_headless` also matches the guard's own command line. Track jobs by process cwd instead.
- **Stopped** at ~550k steps (seed 0 only), after the user said decent resonator performance is enough (better at depth). Score per 250k block:

  | Arm | 0–250k | 250–500k | 500k+ |
  |---|---|---|---|
  | e-prop, plateau | −0.74 | 0.92 | 1.66 |
  | e-prop, standard | −0.51 | 1.06 | 1.06 |
  | ours, plateau | −0.65 | 0.14 | 0.30 |
  | ours, standard | −0.54 | 0.26 | 0.35 |

  The plateau does not break e-prop and does not help ours, so it is no separation.

## Resonator: performance and depth (started)

- `resdepth/`, seed 0, 2M steps, 5 jobs:
  - 2-layer, period 50, gain 1.2, ours and e-prop. A longer period is closer to LIF; the gain follows half the DC-gain ratio, as for period 25.
  - 5-layer, period 25, gain 2.8, ours and e-prop.
  - `std5_eprop`: a 5-layer standard e-prop reference. Ours standard 5-layer CPU: `adam_s0/1`, 2.30 / 1.82 in the last 500k.
- Waiter prints "RESDEPTH DONE".

### Results (seed 0; score per 250k-step block)

| Run | Steps | Rate | Score per block |
|---|---|---|---|
| 2L P50 ours | 1.92M | 120 Hz | −0.72 0.05 1.57 3.44 2.97 3.86 4.57 **4.24** |
| 2L P50 e-prop | 1.95M | 64 Hz | −0.44 3.66 7.14 9.62 8.54 9.87 10.29 **9.98** |
| 5L P25 ours | 1.71M | — | −0.8 flat (no learning) |
| 5L P25 e-prop | 1.70M | 142 Hz | −0.8 flat |
| 5L standard e-prop | 1.89M | 88 Hz | −0.70 −0.25 0.19 0.84 2.03 2.75 2.55 **3.36** |

Reference: ours, standard, 5L: 2.30 / 1.82 in the last 500k.

**Reading:**
- Period 50 is a *better* neuron for both rules: e-prop ≈ 10 (vs 6–7 standard), ours ≈ 4.3 (vs 3.3). E-prop is not broken at this period.
- Period 25 at depth: nothing learns. The signal likely does not propagate through 5 resonator layers, so this is not a rule comparison.

**Next:** 2L sweep at periods 30, 35, 40, both rules. Gain = half the DC-gain ratio: 2.22, 1.83, 1.55. Waiter prints "SWEEP DONE".

### Period sweep (2L, seed 0; last 250k block, with the late trend)

| Period (gain) | ours | e-prop | rates ours / e-prop |
|---|---|---|---|
| 25 (2.8) | 0.55 (4-seed mean) | −0.29 (collapse) | ~115 / ~130 Hz |
| 30 (2.22) | 1.17 | 3.06 (noisy, 1.3–3.1) | 143 / 110 Hz |
| 35 (1.83) | 3.90 | 8.70 | 122 / 82 Hz |
| 40 (1.55) | 4.19 (4.2–4.9) | 10.83 | 122 / 76 Hz |
| 50 (1.2) | 4.24 | 9.98 | 120 / 64 Hz |
| standard neuron | 3.34 / −0.57 | 7.21 / 5.82 | |

**Reading:**
- E-prop wins at every period ≥ 30. The only setting where ours wins is period 25, where both are poor.
- Resonators with period ≥ 35 are a *better* neuron than standard LIF for both rules, and e-prop exploits them more: ≈ 10–11 vs 6–7.
- Below that, both rules degrade together until e-prop collapses at 25. Our rule degrades too; it just never collapses.
- No useful region where the noise-based rule beats e-prop. The honest result: e-prop's structural approximation breaks only where the neuron is too poor to be useful.
