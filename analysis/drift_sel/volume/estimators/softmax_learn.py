"""Phase 3: can the teacher's architecture learn the task with each feedback signal?
Contextual bandit: 8 contexts (binary input patterns over D = 20 inputs), one correct action each; reward 1 if the
chosen pool is correct. The competition (pools, shared inhibition) is fixed; the input weights W_in (E neurons x D)
start small and random (all pools equally likely). Per trial dW_in[i] = eta * credit_i * u (u the context pattern),
per-learner RMS-normalised. 16 independent learners in parallel.
Credit signals:
  own       : delta * sum_t xi_i
  pool      : own averaged over the neuron's pool
  win_share : delta * (s_k - sbar_k), s_k the pool's spike share in the window, sbar_k its running mean (rate 0.02)
  mixed     : own + pool-relayed
delta = r - rbar (running mean of the learner's reward, rate 0.02).
"""
import sys, numpy as np, softmax_net as S

K, NE, N = S.K, S.N_E, S.N
D, NCTX, M = 20, 8, 16
pool = np.repeat(np.arange(K), NE)


def run(rule, trials=3000, eta=0.003, seed=0, b=0.085, gin=0.02):
    rng = np.random.default_rng(seed); W = S.weights()
    task = np.random.default_rng(100 + seed)
    pats = (task.random((NCTX, D)) < 0.3).astype(float); correct = task.integers(0, K, NCTX)
    Win = 0.05 * rng.standard_normal((M, K * NE, D)) / np.sqrt(D)
    rbar = np.full(M, 0.25); sbar = np.full((M, K), 0.25); ms = np.full(M, 1e-12); rew = []
    for t in range(trials):
        c = rng.integers(0, NCTX, M); u = pats[c]                         # (M, D)
        drive = np.zeros((M, N)); drive[:, :K * NE] = b + gin * np.einsum("mid,md->mi", Win, u)
        a, cnt = S.simulate(np.zeros((M, K)), W, rng, drive_override=drive); xs = S.simulate.last_xisum[:, :K * NE]
        r = (a == correct[c]).astype(float); delta = r - rbar; rbar += 0.02 * (r - rbar); rew.append(r)
        share = cnt / np.maximum(cnt.sum(1, keepdims=True), 1)
        own = delta[:, None] * xs
        pl = np.repeat(own.reshape(M, K, NE).mean(2), NE, axis=1)
        if rule == "own":
            cr = own
        elif rule == "pool":
            cr = pl
        elif rule == "win_share":
            cr = np.repeat(delta[:, None] * (share - sbar), NE, axis=1)
        elif rule == "mixed":
            cr = own / (np.sqrt((own ** 2).mean(1, keepdims=True)) + 1e-12) + pl / (np.sqrt((pl ** 2).mean(1, keepdims=True)) + 1e-12)
        sbar += 0.02 * (share - sbar)
        G = cr[:, :, None] * u[:, None, :]
        ms = 0.99 * ms + 0.01 * (G ** 2).mean((1, 2))
        Win += eta * G / (np.sqrt(ms / (1 - 0.99 ** (t + 1)))[:, None, None] + 1e-12)
    R = np.array(rew)                                                   # (trials, M)
    return R.reshape(-1, 250, M).mean(1)                                # blocks of 250 trials


if __name__ == "__main__":
    rules = sys.argv[1:] or ["own", "pool", "win_share", "mixed"]
    for rl in rules:
        B = run(rl)
        m = B.mean(1)
        print(f"{rl:10s} reward per 250-trial block: " + " ".join(f"{v:.2f}" for v in m)
              + f"   (last block sd over learners {B[-1].std():.2f})", flush=True)
