# Strictly Online Reinforcement Learning in Recurrent Spiking Networks

Paper, derivations and analysis code (formerly the `personal` repo).

- `paper.tex`, `references.bib`, `chapters/`, `images/`: the paper. Build with tectonic 0.17: `tectonic -X compile paper.tex`.
- `derivations/`: estimator toys and the `network_exp` rule experiments (`genn-src` is the GeNN 5.4.0 source; rebuild it).
- `analysis/`: analysis code and small results. Bulk result files (≤ 2 MB each, ~2 GB total) and large raw recordings are kept outside git.
- `RULE_VARIANT=*`, `FREEZE_READOUT=1`: example-script snapshots for the rule variants.

Depends on ml_genn: https://github.com/Nordest27/ml_genn
