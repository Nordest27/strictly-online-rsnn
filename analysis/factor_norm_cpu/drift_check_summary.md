# Drift check in the 5-layer Snake network (2026-10-28)

The question is whether the drift (the mean score) equals the weight derivative of the eligibility, P ≈ c̄ ∂E[e]/∂W, as in the paper's toy network (corr 0.91, slope 2.21, R² 0.83).

**Setup**
- Network: `adam_s0` weights at episode 2000 (~1.2M steps), the CPU 5-layer network, then 2000 warm-up steps without learning.
- Score: gate ψ × ZFilter × noise trace. This is the new default; there is no epsA in the score.
- Synapses: 30 recurrent E→E synapses per layer that transmit and are eligible during the 30-step window.
- Stein side: 8192 rollouts.
- Derivative side: 256 rollout pairs with common random numbers, h = 20% of the group's RMS weight.
- Expectations are over the noise and the Poisson input spikes, at fixed input rates (one game frame).
- Code: `ml_genn/examples/eprop/drift_check.py`. Data: `dcw_L1/outputs`, `dcw_L5/outputs` (`*_drift_check.npz`).

| layer | per step: corr / slope / R² | accumulated over the window: corr / slope | well-measured entries: corr / slope | mean score vs E[e]: per step / accumulated | maximum corr allowed by measurement noise |
|---|---|---|---|---|---|
| L1 | **0.79** / 0.40 / 0.63 | **0.95** / 0.30 | 232 of 900: 0.92 / 0.41 | 0.36 / 0.71 | ~0.98 |
| L5 | **0.16** / 0.14 / 0.03 | **0.36** / 0.33 | 129 of 900: 0.32 / 0.25 | 0.43 / 0.51 | ~0.70 |

"Well-measured" means both the score and the derivative are larger than 2 standard errors. The maximum correlation allowed by measurement noise is computed from each side's estimated reliability (1 − s.e.² / variance).

**Correction (same day): the first reading was wrong.** The L1/L5 Pearson difference is a magnitude artefact. A few synapses with large eligibility dominate L1's Pearson correlation, and L5's eligibility is ~6× smaller (median |E[e]| 0.021 vs 0.132). Rank statistics show the two layers are alike:

| | L1 | L5 |
|---|---|---|
| Spearman corr, accumulated over the window | 0.58 | 0.65 |
| Spearman corr, per step | 0.43 | 0.32 |
| Per-synapse correlation over time (median) | 0.17 | 0.11 |

**Reading**
- **Across synapses, accumulated over the window,** the mean score tracks ∂E[e]/∂W moderately in both layers (rank corr ~0.6) and strongly for the most eligible synapses.
- **Step by step within one synapse,** the relation is weak. This matches the paper's factorisation analysis, where single steps are dominated by their own noise.
- **Depth:** nothing here indicates a dependence on depth. The relation is local to the neuron and its own noise; whether a neuron's input comes from Poisson pixels or a learned layer does not enter it.
- **Slope:** 0.3–0.4 here, against 2.2 in the toy. The score's scale differs (the κ factor, the number of afferents sharing each noise sample), so the slope is not compared directly.

**Caveats:** one network, one state, one input frame, 30 synapses per layer, a 30-step window. The per-synapse estimates are the noisiest part; more rollouts would sharpen them.

## Final check matched to the training runs (2026-10-29)

**Setup**
- Gate: the rule's drift gate ψ − ψ̄ on both sides. The earlier checks gated with ψ, which added the gradient part's noise.
- Averaging: both sides go through the rule's own filters, the eligibility trace (λ = 0.9926 per step) and then Adam's momentum (β1 = 0.99), over a 200-step window.
- Samples: 20 synapses per layer, 6144 Stein rollouts, 128 derivative pairs.
- Score: no epsA.
- Data: `dcw200_L1`, `dcw200_L5`. Analysis: `analyse_drift_check.py`.

| | L1 | L5 |
|---|---|---|
| Rule-averaged: score vs ∂E[e]/∂W, corr (Spearman) | **0.80** (0.80) | **0.91** (0.95) |
| Rule-averaged, corrected for measurement noise | 0.81 | 0.95 |
| Rule-averaged: score vs E[e], corr | 0.11 | 0.64 |
| Rule-averaged regression, standardised: β for derivative / eligibility | 0.94 / 0.41 | 0.80 / 0.21 |
| Rule-averaged sign agreement with the score: derivative / eligibility | 0.80 / 0.60 | 0.95 / 0.90 |
| Repeatability of the rule-averaged score / derivative | 1.00 / 0.99 | 0.98 / 0.94 |
| Reliability of one trajectory's 200-step momentum window | 0.19 (noise ≈ 2× signal) | 0.008 (noise ≈ 11× signal) |

**Reading**
- **The drift is the weight derivative of the eligibility, once averaged the way the rule averages it, in both layers.** The derivative explains the score much better than the eligibility itself does.
- **Per-step comparisons are unusable.** The per-step finite differences are not repeatable (L1 −0.33, L5 0.19), probably because a weight nudge occasionally flips a spike, which produces heavy tails. All the earlier per-step "weak correlations" were measurement noise.
- **Signal-to-noise falls sharply with depth.** The relation itself does not depend on depth, but a single training trajectory sees the deep layer's drift with ~11× noise against ~2× in L1. The deep layer's eligibility is ~6× smaller, so its signal is smaller while the noise stays large. This is a candidate explanation for slower, more fragile learning in deep layers.

**Caveats:** one trained network, one state, 20 synapses per layer.

## Toy replay and a correction (2026-10-29)

**Toy:** `drift_toy.py` is an open-loop single-neuron replay of recorded network activity (`rec_L1`, `rec_L5`: 256 windows × 200 steps from the same snapshot).
- With the recorded noise it reproduces GeNN's voltages to ~1e-7.
- Its drift check matches the network's: rule-averaged score toy vs network 1.00 (L1) and 0.97 (L5); score vs derivative 0.81 / 0.96, against 0.80 / 0.91 in the network.
- It runs in 20–110 s instead of ~2 h. The open-loop derivative differs from the network's closed-loop one in L1 (0.64), because recurrent paths add to the network's finite differences but not to the drift.

**Correction:** the "~2× vs ~11× noise per trajectory" figures were wrong. They came from split-half reliability across synapses, which depends on how much the synapses' true drifts differ, and is not a per-synapse signal-to-noise. Measured directly (`toy_snr.py`), one trajectory's rule-averaged drift has |mean|/sd = 0.117 (median) in both L1 and L5, despite 4× fewer afferents, half the presynaptic rate and a third of the ψ in L5.
- There is **no depth effect on signal-to-noise**.
- In both layers one momentum window sees noise ≈ 8.5× signal, so a synapse needs ~10⁴ steps or more to resolve its drift direction.

**Also tested:** broadcast-δ e-prop (δ × eligibility, no noise) does not learn. At ~1.1M steps it is flat at the floor, rates climb with depth and 12–41% of E neurons are silent, the same signature as the noise-proxy runs.

## Signal-to-noise on a broader sample (2026-10-29, revises the previous section)

`toy_snr_dist.py` covers 24 random neurons per layer and every recurrent synapse that transmits during a 200-step window. It does not use the drift check's pilot-selected synapses.

| | L1 | L5 |
|---|---|---|
| Recurrent synapses transmitting in a window | 36% | 71% |
| Per-trajectory SNR of the rule-averaged drift, median (quartiles) | 0.16 (0.05–0.43) | 0.05 (0.02–0.11) |
| Below the search's "unmeasurable" bound (3/√6144 ≈ 0.038) | 23% | 41% |
| Neuron mean ψ (median) | 0.048 | 0.019 |

- **Selection bias:** the "equal SNR across depth" result (0.117 in both layers) came from the 20 pilot-selected synapses, a selection biased towards active, well-conditioned synapses.
- **Broader sample:** L5's SNR is ~3× lower than L1's. The mechanism is not depth-dependent, but this network's deep neurons sit at lower ψ, the regime the synthetic search finds ill-conditioned.
- **Implication:** a large share of the synapses (a quarter in L1, two-fifths in L5) get a drift buried in noise, so their weights effectively random-walk.

## What makes synapses ill-conditioned (2026-10-29, revises the "low ψ" explanation above)

**Tailed gates** (`toy_tails.py`: exponential and fast-sigmoid tails in the score's gate only):
- The drift still tracks the derivative of the tailed eligibility (0.77–0.99).
- The share of ill-conditioned synapses barely changes (L1 ~21%, L5 ~40%). Tails rescue 12–31% of the triangle's bad synapses but worsen about as many.

**Cause: presynaptic sparsity, not the neuron's ψ.** Per synapse, SNR has rank correlation 0.73 (L1) and 0.51 (L5) with presynaptic spikes per window, and −0.2 with neuron ψ.
- Synapses with ≤ 1 presynaptic spike per 200-step window: 49% (L1) and 68% (L5) are below the bound.
- Synapses with more spikes are mostly fine.
- L5 has more such sparse inputs (136 of 309 transmitting synapses, against 47 of 155 in L1). That is the whole depth difference.

**Implications:**
- These synapses get one noise delivery per window. Any rule has little to learn from them per unit time (e-prop's eligibility also scales with the presynaptic trace). This is not a pathology of the drift.
- **Dead zone:** neurons spend ~half their time below the ψ range, but this is not what makes synapses ill-conditioned. The instantaneous revival runs show much of that time is selective neurons being legitimately quiet: they emptied the dead zone, pushed rates to 170–180 Hz and reduced the score. Whether the dead zone hurts learning is an open question.
