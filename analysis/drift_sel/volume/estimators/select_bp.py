"""Rate-network reference for the selection task: 20 -> H ReLU -> 4 softmax policy, REINFORCE with exact backprop."""
import numpy as np, sys
K = 4
def run(H=80, trials=30000, lr=0.05, seed=0, beta=1.0):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((H, 20)) / np.sqrt(20); b1 = np.zeros(H); W2 = rng.standard_normal((K, H)) / np.sqrt(H)
    base = 0.25; rew = []
    for t in range(trials):
        cue = rng.integers(K); perm = rng.permutation(K)
        u = np.concatenate([np.eye(K)[cue], np.eye(K)[perm].ravel()]); target = int(np.argmax(perm == cue))
        h = np.maximum(W1 @ u + b1, 0); z = beta * (W2 @ h); p = np.exp(z - z.max()); p /= p.sum()
        a = rng.choice(K, p=p); r = float(a == target); d = r - base; base += 0.01 * (r - base); rew.append(r)
        g = beta * (np.eye(K)[a] - p) * d
        gh = (W2.T @ g) * (h > 0)
        W2 += lr * np.outer(g, h); W1 += lr * np.outer(gh, u); b1 += lr * gh
    return np.array(rew).reshape(-1, 1000).mean(1)
for H in (80, 20):
    print(f"H {H}: reward per 1000 trials", " ".join(f"{x:.2f}" for x in run(H)[::3]))
