"""Hybrid expand (our rule) / compress (e-prop) architectures vs the narrow_5l(5) reference (untrained tests)."""
import copy

BASE = {   # narrow_5l(5), from the user's snake_hparams (architecture keys only)
    "ei_layers": 5, "hid_e": 20, "hid_i": 15, "fan_in": 300, "toroidal": "local", "feedforward_targets": "both",
    "local": {"sigma_cells": 1.75, "p_max": 1.0, "input_sigma_cells": 1.813, "inhibition_scale": 3.972,
              "feedforward_inhibition": False, "p_global": 0.0, "inhibition_sigma_ratio": 1.0},
    "network": {
        "fields": {"conn": {"sigma_cells": 0.3, "mean_scale": 0.6205776944756594, "sd_scale": 0.12411553889513188},
                   "shape": 20},
        "layers": [{"to_next": {"sigma_cells": 0.45, "centre_shift": 3, "mean_scale": m, "sd_scale": m / 10}}
                   for m in (0.8962792218320149, 0.6946442290485155, 0.8631457274064128, 0.8468311991016344)] + [{}],
        "input": {"mean_scale": 0.16789556605999592, "sd_scale": 0.08394778302999796}}}


def hybrid(shapes, field_shape, relay_sigma):
    """shapes: [(e_shape, i_shape)] per layer; relay_sigma[k]: sigma (cells) of layer k -> k+1 (and last -> fields)."""
    hp = copy.deepcopy(BASE)
    lay = hp["network"]["layers"]
    for k, (e, i) in enumerate(shapes):
        lay[k]["e_shape"], lay[k]["i_shape"] = list(e), list(i)
        if k < len(shapes) - 1:
            lay[k]["to_next"]["sigma_cells"] = relay_sigma[k]
    hp["network"]["fields"]["shape"] = list(field_shape)
    hp["network"]["fields"]["conn"]["sigma_cells"] = relay_sigma[-1]
    return hp


CANDIDATES = {
    "base": BASE,
    # expand channels on the 20x20 sheet (our rule), then shrink the sheet (e-prop); compressing relays pool over the
    # source cells one target cell covers (sigma scaled by the size ratio)
    "hyb_sheet": hybrid([((20, 20, 3), (15, 15, 3)), ((20, 20, 4), (15, 15, 4)), ((20, 20, 6), (15, 15, 6)),
                         ((14, 14, 4), (10, 10, 4)), ((10, 10, 3), (8, 8, 3))], (10, 10, 3),
                        [0.45, 0.45, 0.45 * 20 / 14, 0.45 * 14 / 10, 0.3]),
    # expand channels, then compress channels on the same sheet (lateral resolution kept)
    "hyb_chan": hybrid([((20, 20, 3), (15, 15, 3)), ((20, 20, 4), (15, 15, 4)), ((20, 20, 6), (15, 15, 6)),
                        ((20, 20, 3), (15, 15, 3)), ((20, 20, 2), (15, 15, 2))], (20, 20, 2),
                       [0.45, 0.45, 0.45, 0.45, 0.3]),
}


# narrow_5l with more heterogeneous relays: sd_scale + 0.1 on every layer -> layer projection (the user's change)
_relay_sd = copy.deepcopy(BASE)
for _l in _relay_sd["network"]["layers"][:-1]:
    _l["to_next"]["sd_scale"] += 0.1
CANDIDATES["base_relay_sd"] = _relay_sd
