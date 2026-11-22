# narrow_5l probe (user's GPU run, 2026-10-30): event-conditioned selectivity

**Run:** all local rewards, no deterministic regularisation, Adam 1e-6 everywhere, Dale pruning on inhibitory synapses.

**Probe:** per move, V and a refractory flag (fired within the last refractory period). First and last 5000 moves. No TD error, ψ or A.

## Scripts and outputs

- `event_selectivity.py` → `event_selectivity.txt`: MI between events and task variables, bias-corrected with circular shifts; bits per event against firing rate.
- `reward_alignment.py` → `reward_alignment.txt`: firing enrichment around apples eaten and deaths, against reward times shifted within episodes.

## Findings

**Selectivity to task variables (late):**
- About 60% of E neurons carry information above the null, about 0.35–0.45 bits per event, and heading dominates.
- Early in the run, sparser neurons carried more bits per event (rank corr −0.2 to −0.5). Late in the run there is no relation (≈0).
- The early-vs-late comparison is confounded: danger is present at 94% of late moves vs 37% early, and late snapshots cover 97 episodes vs 1013.

**Reward moments:**
- Firing drops to about 0.56× baseline at the sample where an apple is eaten.
- Late in the run, firing rises to about 1.45× at the death sample, with 30% of neurons ×2-enriched (null 5%).
- Both effects are the same in every layer, L1 to L5 and the fields, so they are network-wide and most likely input-driven: the frame changes at an apple or a death. This is not neuron-specific selective firing.

**Conclusion:** no evidence in this probe that sparse neurons fire selectively at informative moments, but the probe is not able to show it.
- It samples once per move.
- It has no TD error and no gate.
- The task distribution shifts between snapshots.

**The real test:** record, per step, when each neuron's gate is open (ψ, needs A) together with the TD error, and ask whether gate-open moments coincide with large |TD|, relative to rate-matched shuffles.

## Matched comparison: reservoir replayed on the trained agent's states (2026-10-30)

**Setup:**
- `replay_res_v2/`: the untrained narrow 5-layer reservoir (`d5_narrow_v2` config, CPU) is driven by the trained run's last 5000 recorded frames (`snake.py` option `replay`).
- Checked: replayed frames, task variables and bodies equal the recording.
- Analysed with `layer_probe_tests.py` (tests 1, 2, 4, 5, 6) → `layer_probe_tests_replay_reservoir.txt`.
- Trained network: `layer_probe_tests.txt`, last window. Same states; 4,903 vs 4,806 aligned samples.

**Trained vs reservoir on identical states (L1_E / L5_E / value_field):**

| Variable | Trained | Reservoir | Input frame |
|---|---|---|---|
| apple side, y (bal. acc. above chance) | 0.51 / 0.42 / 0.40 | 0.21 / 0.23 / 0.21 | 0.54 |
| path to apple (R²) | 0.49 / 0.31 / 0.31 | 0.09 / 0.07 / 0.07 | 0.64 |
| length (R²) | 0.61 / 0.21 / 0.20 | 0.30 / 0.01 / 0.07 | 0.61 |
| tail (R², mean y/x) | 0.68 / 0.56 / 0.56 | 0.50 / 0.49 / 0.47 | 0.69 |
| danger (mean of 4) | 0.48 / 0.45 / 0.44 | 0.44 / 0.41 / 0.41 | 0.48 |
| free area (mean of 4) | 0.86 / 0.67 / 0.66 | 0.71 / 0.61 / 0.66 | 0.88 |

**Caveats:**
- The memory test is not comparable: the reservoir's state includes adaptation A (`adapt=True`), the trained probe has V only.
- The reservoir is the same architecture family, not the run's exact initial network (fields 675 vs 1200, other connection scales).

## Screen-like structure (pixel_structure.py → pixel_structure.txt / .png)

| Population | Trained: median max\|r\| with a pixel / topography | Reservoir (same frames) |
|---|---|---|
| L1_E | 0.52 / 0.52 | 0.37 / 0.38 |
| L2_E | 0.47 / 0.40 | 0.44 / 0.61 |
| L3_E | 0.44 / 0.36 | 0.44 / 0.50 |
| L5_E | 0.41 / 0.28 | 0.40 / 0.43 |
| policy field | 0.40 / 0.28 | **0.62 / 0.70** |

**Apple-conditioned maps:**
- Trained: a sharp map in L1 (localised suppression at the apple's position), weaker in L2, none from L3 on (salt-and-pepper, non-topographic).
- Reservoir: large, smooth, blurred blobs at every depth, including the fields. It relays a low-pass copy of the screen.

**Body-conditioned maps** (`pixel_structure.py --body`, `pixel_structure_body.png`; body at a cell next to the head minus no body there):
- Trained: L1 shows raised activity roughly where the body is (noisier than the apple map); from L2 on, salt-and-pepper.
- Reservoir: large, smooth quadrant-scale patterns at every depth, including the fields. The same blurred screen relay as for the apple.

## Shift 3 and the user's architecture (reservoir replays on the same frames, 2026-10-30)

- `replay_res_v2_shift3`: `centre_shift` 3 on all `to_next` projections.
- `replay_user_arch`: also `sd_scale` + 0.1 on the `to_next` projections. The user's architecture except possibly the seed; fields and input unchanged (15×15 fields here vs 20×20 in the run).

**Effect of the shift:**
- The reservoir's screen image in the fields disappears.
- Topography falls with depth just as in the trained network (fields 0.30 vs trained 0.28). So the trained network's loss of topography is architectural, not learned.
- Pixel copying stays higher in the reservoir's deep layers (best |r| 0.49–0.51 vs trained 0.41–0.44). Trained L1 is sharper (0.52 vs 0.37).

**Decorrelation, matched (spike neighbour r; reservoir with the user's architecture vs trained, late window):**

| | L1 | L2 | L3 | L4 | L5 | fields |
|---|---|---|---|---|---|---|
| reservoir (user arch) | 0.134 | 0.114 | 0.071 | 0.055 | 0.048 | 0.136 |
| trained | 0.081 | 0.080 | 0.067 | 0.057 | 0.043 | 0.049 |

Training decorrelates L1, L2 and above all the fields. L3–L5 are already as decorrelated in the shifted reservoir.
