"""Deterministic (Rao-Blackwell-style) prediction of the drift vs the true drift (many-rollout Stein mean), per
synapse, for sparse and dense presynaptic inputs. Prediction along the noise-free trajectory:
    P_t = kappa * ZFilter_t * smoothed_slope(V_t) * K_t,   K_t = sum_m alpha^(2m) c_{t-m} z_i(delivered at t-m)
i.e. each delivery of the synapse kicks the voltage by alpha^m and is weighted by alpha^m in the noise trace, times
the c afferents sharing the noise; the gate's slope is smoothed by the voltage noise sigma_V (SynSig x active
afferents, accumulated through the membrane). Ignores spike/reset effects of the kick and recurrent paths."""
import glob, sys
import numpy as np
from scipy.stats import norm, spearmanr
sys.path.insert(0, "/home/user/repos/ml_genn/examples/eprop")
import drift_toy as dt


def predict_neuron(rec, j, ks, sigs):
    """Rule-averaged-ready deterministic predictions (W, T, len(ks)) for afferents ks of neuron j, from one batched
    noise-free run over all windows."""
    deliver, trace = rec.inputs(j)
    W, T = rec.W, rec.T1 - 1
    _, _, V = dt.simulate(rec, j, np.arange(W), np.zeros((W, rec.T1)))        # (W, T) noise-free voltages
    A0 = rec.state["A"][:, 0, j][:, None]; beta = rec.state["Beta"][:, 0, j][:, None]
    th = rec.vth + beta * A0 * rec.rho ** np.arange(1, T + 1)[None, :]           # (W, T)
    a, kappa, c0 = rec.alpha, 1 - rec.alpha ** 2, 0.3 / rec.vth / rec.vth
    n_act = deliver.sum(-1).astype(float)                                         # (W, T+1)
    out = np.zeros((W, T, len(ks)))
    for m, (k, sig) in enumerate(zip(ks, sigs)):
        var = np.zeros((W, T + 1))
        for t in range(1, T + 1):
            var[:, t] = a * a * var[:, t - 1] + (sig * n_act[:, t - 1]) ** 2
        sd = np.sqrt(var[:, 1:]) + 1e-9
        lo, mid, hi = th - rec.vth, th, th + rec.vth
        slope = c0 * ((norm.cdf((V - lo) / sd) - norm.cdf((V - mid) / sd))
                      - (norm.cdf((V - mid) / sd) - norm.cdf((V - hi) / sd)))
        kick = deliver[:, :-1, k] * n_act[:, :-1]
        K = np.zeros((W, T)); acc = np.zeros(W)
        for t in range(T):
            acc = a * a * acc + kick[:, t]
            K[:, t] = acc
        out[:, :, m] = kappa * trace[:, :-1, k] * slope * K
    return out


L = sys.argv[1]
rec = dt.Recording(glob.glob(f"rec_L{L}/outputs/*drift_record.npz")[0])
tg = next(gr for gr in rec.groups if gr["name"] == rec.target_group)
rng = np.random.default_rng(7)
neurons = rng.choice(np.unique(tg["post"]), size=12, replace=False)
true_f, pred_f, nspk = [], [], []
for j in neurons:
    src, pre, g0, sig, ids = rec.afferents(j)
    mine = [k for k, (name, _) in enumerate(ids) if name == rec.target_group]
    deliver, _ = rec.inputs(j)
    wr = np.repeat(np.arange(rec.W), 24)
    s, e, _ = dt.simulate(rec, j, wr, rng.standard_normal((len(wr), rec.T1)))
    S = s[:, :, mine].reshape(rec.W, 24, rec.T1 - 1, len(mine)).mean(1)       # per window, Stein mean over rollouts
    P = predict_neuron(rec, j, mine, sig[mine])                                  # (W, T, n)
    for m, k in enumerate(mine):
        cnt = deliver[:, :-1, k].sum(1).mean()
        if cnt == 0:
            continue
        true_f.append(dt.rule_filter(S[:, :, m:m + 1].mean(0, keepdims=True)[0], rec.lam, rec.beta1)[0])
        pred_f.append(dt.rule_filter(P[:, :, m].mean(0)[:, None], rec.lam, rec.beta1)[0])
        nspk.append(cnt)
true_f, pred_f, nspk = map(np.array, (true_f, pred_f, nspk))
for name, m in [("sparse (<=1 spike/window)", nspk <= 1), ("dense (>1 spike/window)", nspk > 1), ("all", nspk >= 0)]:
    if m.sum() > 3:
        print(f"L{L} {name:26s}: n={m.sum():3d} | prediction vs true drift (rule-averaged): corr {np.corrcoef(pred_f[m], true_f[m])[0,1]:.2f}, "
              f"Spearman {spearmanr(pred_f[m], true_f[m])[0]:.2f}, sign agreement {np.mean(np.sign(pred_f[m]) == np.sign(true_f[m])):.2f}, "
              f"slope {np.polyfit(pred_f[m], true_f[m], 1)[0]:.2f}")
