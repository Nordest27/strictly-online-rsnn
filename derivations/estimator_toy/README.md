# Estimator toy: bias and noise of perturbation-based rules

Single-neuron Monte-Carlo checks of the gradient estimators discussed for
distributed e-prop. Plain numpy, single process. Run from this directory
(`python3 <script>.py [N_SAMPLES]`).

## Estimators (for weight w_i, multiplied by the reward minus a baseline)

| name | score profile |
|---|---|
| `dist_eprop` | (1-α²) ξ^t / σ² · e_i^t (perturbation trace × e-prop eligibility, with ψ) |
| `trace_nopsi` | (1-α²) ξ^t / σ² · z̄_i^t (perturbation trace × presynaptic trace) |
| `step_pretr` | ε^t / σ² · z̄_i^t (fresh perturbation × presynaptic trace) |
| `step_psi` | ε^t / σ² · e_i^t (fresh perturbation × e-prop eligibility, with ψ) |
| `exact` | ε^t / σ² · z_i^{t-1} (exact likelihood-ratio estimator, unbiased) |

## Scripts

| script | what it checks |
|---|---|
| `check_bias.py`, `check_bias_multi.py` | linear leaky neuron: exact estimator is unbiased; trace estimator equals the predicted blurred gradient Σ_s g_s z̃^s with z̃^s = Σ_t α^{|t-s|} z̄^t |
| `check_noise_formula.py` | linear neuron: closed-form relative noise² = 1 + \|g_ε\|² Σ\|a_i\|² / Σ(g_ε·a_i)² matches Monte Carlo |
| `alif_bias.py` | ALIF neuron (reset, refractoriness, adaptation, ψ): all five estimators vs finite-difference truth |
| `alif_bias_realistic.py`, `alif_bias_mp.py`, `alif_bias_simple.py` | same at σ = e⁻⁵ with Poisson (p = 0.7) inputs and silent gaps; `alif_bias_mp.py` is the jagged-reward run (multiprocessing scaled poorly on WSL; prefer `alif_bias_simple.py`) |
| `alif_bias_events.py`, `alif_bias_events_lambda.py` | sparse, decaying, activity-dependent rewards with causal credit (cumulative / λ-trace) |
| `alif_bias_events_diag.py` | absolute spread, mean size and ψ-open fraction |
| `alif_bias_psiwidth.py` | ψ width sweep (`PSI_W=0.05 python3 alif_bias_psiwidth.py 100000`) |
| `check_noise_alif.py`, `check_noise_alif_varF.py` | closed-form noise prediction on ALIF (linear part = lower value, total Var(F) = upper value) |
| `psi_regimes.py` | ψ's effect across operating regimes (direct Monte Carlo; exact estimator's mean as truth) |

Lesson learned: finite-difference truth needs a step much smaller than the
noise's effect on the membrane (h = 0.02 at σ = 0.02 was biased; the exact
estimator exposed it).

## Key results (σ = e⁻⁵ unless noted; cells are cosine to truth / relative noise)

**Sparse decaying reward, λ-trace credit** (`alif_bias_events_lambda.py`, 200k):
dist_eprop 0.997/2.6, trace_nopsi 0.992/2.1, step_pretr 0.993/2.4,
step_psi 0.984/3.2, exact 0.999/3.6.

**Reward shape** (dist_eprop cosine): sparse decaying 0.997, jagged 0.993,
smooth 50-step window 0.83 (`alif_bias_simple_results.txt`).

**ψ width** (dist_eprop / step_psi cosine): 0.61 → 0.996 / 0.983;
0.2 → 0.989 / 0.957; 0.05 → 0.958 / 0.844; 0.016 → 0.894 / 0.667;
0.005 → 0.869 / 0.578. e-prop's default width is best; the trace makes the
rule far more robust to a narrow gate.

**Noise bracket, jagged reward** (predicted lower / measured / predicted upper):
dist_eprop 2.20/2.6/2.79, trace_nopsi 2.11/2.5/2.67, step_pretr 2.43/2.9/3.10,
step_psi 2.71/3.4/3.50, exact 3.20/4.0/4.16. Direction predicted to within
0.001 for all five. Only 57% of the reward variance is linear in the noise.

**Operating regimes** (`psi_regimes.py`, 500k, jagged reward):

| regime | fire | ψ open | dist_eprop | trace_nopsi | step_pretr | step_psi | exact |
|---|---|---|---|---|---|---|---|
| mean w 0.01 | 0.4% | 99% | 0.997/3.3 | 0.995/4.1 | 0.982/4.8 | 0.992/3.4 | 1/5.0 |
| mean w 0.02 | 6.7% | 86% | 0.962/7.3 | 0.965/5.1 | 0.936/4.4 | 0.938/5.9 | 1/7.7 |
| mean w 0.04 | 17% | 65% | 0.961/21.6 | 0.960/27.0 | 0.864/16.7 | 0.934/22.8 | 1/14.1 |
| inhibited, 21% far below | 10.6% | 61% | 0.988/5.8 | 0.980/5.7 | 0.959/6.8 | 0.972/8.1 | 1/8.8 |
| inhibited, 47% far below | 3.3% | 49% | 0.991/20.3 | 0.966/23.6 | 0.956/26.3 | 0.973/23.0 | 1/29.4 |
| inhibited, 85% far below | 1.3% | 14% | 0.980/8.0 | 0.909/10.7 | 0.920/12.2 | 0.992/14.5 | 1/14.9 |
| saturated, 35% far above | 30% | 24% | 0.977/7.0 | 0.809/7.3 | 0.835/15.7 | 0.996/10.1 | 1/10.9 |
| saturated, 93% far above | 33% | 2.7% | 0.997/11.0 | 0.727/46.3 | 0.883/431 | 0.997/12.5 | 1/67.8 |

Inhibited regimes: excitatory mean 0.08/0.08/0.12, four inhibitory inputs at
−0.05/−0.10/−0.20. Mean w 0.30 was unreliable (exact relative noise ≈ 900) and
is excluded.

## Control variate and e-prop diagnosis (`control_variate.py`, `eprop_diagnosis.py`)

Node noise, 4 readouts, sparse event reward, whole-trajectory credit (40k samples).
Relative noise, near threshold / inhibited: exact 5.9 / 6.2; control variate with
oracle h 4.7 / 4.2; with e-prop h (symmetric or random feedback) 5.3-5.9 / 5.6-5.7;
distributed e-prop (trace rule) 3.8 / 2.8 at cosine 0.999 / 0.988. All control-variate
estimators stay unbiased (cosine 1.000).

E-prop's own update pointed the wrong way in this toy (symmetric feedback: cosine 0.48
near threshold, -0.93 inhibited). Diagnosis:
- Code verified: single-neuron RTRL = finite differences (4e-7) in a smooth deterministic
  neuron, and = true gradient of E[F] with smooth spikes plus noise (cosine 1.000).
- Main cause (toy-specific): one spike moves a readout by ~1 while the reward's tanh curves on
  a scale of 0.5, so the local derivative mispredicts the effect of adding/removing a spike.
  Scaling each spike's readout effect by 0.1: RTRL (narrow psi) 0.68 -> 0.86 near threshold,
  -0.77 -> 0.95 inhibited; e-prop (narrow psi) -1.00 -> 0.93 inhibited.
- E-prop's approximations also cost accuracy here: dropping the reset derivative (cosine 0.46
  in the smooth deterministic neuron) and the wide default surrogate. With steep spikes, full
  RTRL's gradients explode through the reset; e-prop's truncation keeps them stable.
Conclusion: this single-neuron setting is unfavourable to pathwise methods and is not used to
evaluate e-prop. The same wide psi that is a poor surrogate derivative here is the best gate
for the perturbation rule (see the psi width sweep).
