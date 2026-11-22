"""Distribution of the per-synapse single-trajectory signal-to-noise of the rule-averaged drift over many randomly
chosen eligible synapses of the recorded network layers (toy replay)."""
import glob, sys
import numpy as np
sys.path.insert(0, "/home/user/repos/ml_genn/examples/eprop")
import drift_toy as dt

thr = 3 / np.sqrt(6144)          # the search's "unmeasurable" bound at its largest sample size
for L in sys.argv[1:]:
    rec = dt.Recording(glob.glob(f"rec_L{L}/outputs/*drift_record.npz")[0])
    tg = next(gr for gr in rec.groups if gr["name"] == rec.target_group)
    rng = np.random.default_rng(7)
    snr, psi, txs = [], [], []
    for j in rng.choice(np.unique(tg["post"]), size=24, replace=False):
        src, pre, g0, sig, ids = rec.afferents(j)
        mine = [k for k, (name, _) in enumerate(ids) if name == rec.target_group]
        wr = np.repeat(np.arange(rec.W), 12)
        s, e, _ = dt.simulate(rec, j, wr, rng.standard_normal((len(wr), rec.T1)))
        f = dt.rule_filter(s[:, :, mine], rec.lam, rec.beta1)          # (trajectories, synapses)
        deliver, _ = rec.inputs(j)
        transmits = deliver[:, 1:, mine].any((0, 1))         # the presynaptic neuron spikes in the window
        elig = (np.abs(e[:, :, mine]).mean((0, 1)) > 0) & transmits & (f.std(0) > 0)
        frac_tx = transmits.mean()
        r = np.abs(f.mean(0)) / (f.std(0) + 1e-30)
        snr += list(r[elig]); psi.append(rec.state["Psi"][:, :, j].mean()); txs.append(frac_tx)
    snr = np.array(snr)
    print(f"L{L}: {len(snr)} eligible recurrent synapses on 24 neurons | per-trajectory SNR median {np.median(snr):.3f}, "
          f"quartiles {np.percentile(snr, 25):.3f}/{np.percentile(snr, 75):.3f} | below the unmeasurable bound "
          f"({thr:.3f}): {np.mean(snr < thr):.0%} | neuron mean psi median {np.median(psi):.3f} | recurrent synapses "
          f"transmitting in a window: {np.mean(txs):.0%}")
