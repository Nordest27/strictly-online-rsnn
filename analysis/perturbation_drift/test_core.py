"""Invariants the analyses rely on. Run: python3 -m pytest -q test_core.py   (or: python3 test_core.py)"""
import numpy as np
from core import Config, Learner, StaggeredTwins, NeuronState, Layout, build_weights, advance, score, f32

CFG = Config()


def _small_learner(variants=("original",), seeds=(0,)):
    return Learner(CFG, list(variants), list(seeds), rng_seed=0)


def test_twin_with_identical_noise_reproduces_network():
    """A twin given the same noise (sign +1) must reproduce the real network exactly, so the noise
    part of the decomposition is exactly zero."""
    L = _small_learner()
    L.twins = StaggeredTwins(CFG, L.lay, L.B, sign=+1.0)
    qmax = []
    L.hooks.append(lambda l, out, ex: qmax.append(np.abs(ex["Q"]).max()))
    L.run(30, block=1000, log=None)
    assert max(qmax) == 0.0


def test_twin_with_mirrored_noise_differs():
    L = _small_learner()
    L.twins = StaggeredTwins(CFG, L.lay, L.B, sign=-1.0)
    qmax = []
    L.hooks.append(lambda l, out, ex: qmax.append(np.abs(ex["Q"]).max()))
    L.run(30, block=1000, log=None)
    assert max(qmax) > 0.0


def test_backward_trace_equals_forward_advantage():
    """sum_t delta_t p_t (p_t = lam p_{t-1} + s_t) equals sum_t s_t A_t with A_t = sum_k lam^k delta_{t+k}."""
    r = np.random.default_rng(0); T = 200; lam = 0.97
    s, d = r.normal(size=T), r.normal(size=T)
    p, back = 0.0, 0.0
    for t in range(T):
        p = lam * p + s[t]; back += d[t] * p
    A = np.array([sum(lam ** k * d[t + k] for k in range(T - t)) for t in range(T)])
    assert np.isclose(back, (s * A).sum())


def test_shared_noise_reaches_membrane_as_sigma_n_c():
    """The noise delivered to neuron j in one step is sigma * n_j * (number of afferents that spiked)."""
    lay = Layout(50, 36, 16, 16)
    M, W, _, _ = build_weights(CFG, lay, 0)
    st = NeuronState(1, lay)
    st.z[:] = (np.random.default_rng(1).random(st.z.shape) < 0.3).astype(f32)
    zin = (np.random.default_rng(2).random((1, lay.NIN)) < 0.5).astype(f32)
    nz = np.random.default_rng(3).standard_normal((1, lay.POST)).astype(f32)
    out = advance(CFG, lay, st, W, M, zin, nz)
    c = out["zpre"] @ M
    assert np.allclose(out["applied"].sum(1), CFG.sigma * nz * c, atol=1e-7)


def test_gate_variants_differ_only_in_gate():
    """proposed = original - 2 * psibar * (presynaptic factor * noise trace)."""
    lay = Layout(50, 36, 16, 16)
    M, W, _, _ = build_weights(CFG, lay, 0)
    st = NeuronState(1, lay)
    rng = np.random.default_rng(0)
    for _ in range(50):
        zin = (rng.random((1, lay.NIN)) < 0.5).astype(f32)
        out = advance(CFG, lay, st, W, M, zin, rng.standard_normal((1, lay.POST)).astype(f32))
    s_orig = score(CFG, out["psi"], out["pi"], out["xi"])
    s_prop = score(CFG, out["psi"] - 2 * st.psibar, out["pi"], out["xi"])
    s_bar = score(CFG, st.psibar, out["pi"], out["xi"])
    assert np.allclose(s_prop, s_orig - 2 * s_bar, rtol=1e-5, atol=1e-6)


def test_stein_identity_single_neuron():
    """E[n f(n)] = E[f'(n)]: mean score equals the noise-direction derivative of the eligibility
    (within Monte-Carlo error) on a short window from a fixed state."""
    from experiments_lib import snapshot_rollouts, trained_snapshot
    snap = trained_snapshot(CFG, train_steps=300, seed=0)
    res = snapshot_rollouts(CFG, snap, R=2000, S=10, T=20, noise_fd=True, weight_fd=False)
    a, b, se = res["A"][-1], res["B"], res["A_se"][-1]
    assert np.median(np.abs(a - b) / (se + 1e-12)) < 2.0


def test_light_dynamics_match_full_network():
    """The lightweight rollout dynamics (targets only) reproduce the full network's psi and eligibility."""
    from experiments_lib import advance_light, _Light
    lay = Layout(50, 36, 16, 16)
    M, W, _, _ = build_weights(CFG, lay, 0)
    full, light = NeuronState(1, lay), _Light(1, lay)
    rng = np.random.default_rng(0)
    epsA = np.zeros((lay.PRE, lay.POST), f32)
    for _ in range(80):
        zin = (rng.random((1, lay.NIN)) < 0.5).astype(f32); nz = rng.standard_normal((1, lay.POST)).astype(f32)
        of = advance(CFG, lay, full, W, M, zin, nz)
        ol = advance_light(CFG, lay, light, W, M, zin, nz)
        pi_l = light.zbar[0][:, None] - CFG.beta * epsA
        epsA = ol["psi"][0][None, :] * light.zbar[0][:, None] + (CFG.rho - ol["psi"][0][None, :] * CFG.beta) * epsA
        assert np.allclose(of["psi"], ol["psi"]) and np.allclose(of["pi"][0], pi_l, atol=1e-5)


if __name__ == "__main__":
    import inspect, sys
    fails = 0
    for name, fn in sorted(inspect.getmembers(sys.modules[__name__], inspect.isfunction)):
        if name.startswith("test_"):
            try:
                fn(); print(f"ok    {name}")
            except AssertionError as e:
                fails += 1; print(f"FAIL  {name} {e}")
    sys.exit(fails)
