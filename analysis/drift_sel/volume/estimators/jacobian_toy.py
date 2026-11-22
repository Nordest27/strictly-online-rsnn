"""Toy: can a network learn the feedback of its own computation (local Jacobians from noise-activity covariance) and
learn faster than with perturbation credit alone?

Task: contextual bandit. Inputs are noisy samples of N_CL random cluster centres (dim D); each cluster has a random
correct action among K (not linearly separable); reward 1 if the sampled action is correct.
Network (rates): h1 = relu(W1 x + s1 n1); v = W2 h1 + s2 n2; h2 = softmax(v) (a competition computed in the activity);
u = W3 h2 + s3 n3; action ~ softmax(beta u). Baseline: running mean reward; delta = r - baseline.
Rules (same network, noise and task stream):
  perturbation : every layer dW = eta * delta * n_l / s_l * input_l^T (node perturbation; our rule's logic)
  jacobian     : output layer as above; earlier layers get credit backward through learned Jacobians,
                 C_v = J3^T C_u, C_h1 = J2^T C_v (masked by relu), C_u = delta * n3 / s3, where J3 ~ d u / d v and
                 J2 ~ d v / d h1pre are learned reward-free from noise-activity covariance (running averages of
                 (u - ubar) n2^T / s2 and (v - vbar) n1^T / s1); dW_l = eta * C_l input_l^T
  feedback_al  : as jacobian but with fixed random feedback matrices (feedback alignment)
  backprop     : exact gradient of delta * log pi(a) through the network (upper bound)
Reported: mean reward over training blocks, mean of several seeds.

Usage: python jacobian_toy.py [--trials 40000] [--seeds 5]
"""
import argparse
import numpy as np


def softmax(z, axis=-1):
    z = z - z.max(axis, keepdims=True); e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


def run(rule, seed, trials=40000, D=20, N_CL=8, K=4, H1=64, H2=16, eta=0.003, s=(0.3, 0.3, 0.3), beta=3.0,
        j_rate=0.01, block=2000, clip=None):
    rng = np.random.default_rng(seed)
    task = np.random.default_rng(1000 + seed)              # same task/stream for every rule at a given seed
    centres = task.standard_normal((N_CL, D)); labels = task.integers(0, K, N_CL)
    W1 = rng.standard_normal((H1, D)) / np.sqrt(D)
    W2 = rng.standard_normal((H2, H1)) / np.sqrt(H1)
    W3 = rng.standard_normal((K, H2)) / np.sqrt(H2)
    J2 = np.zeros((H2, H1)); J3 = np.zeros((K, H2))         # learned Jacobians (start at zero: nothing known)
    vbar = np.zeros(H2); ubar = np.zeros(K)
    B2 = rng.standard_normal((H2, H1)) / np.sqrt(H1); B3 = rng.standard_normal((K, H2)) / np.sqrt(H2)
    base = 0.0; rewards = []; s1, s2, s3 = s
    ms = {"W1": 0.0, "W2": 0.0, "W3": 0.0}

    def apply(name, W, G):
        """W += eta * G / RMS(G) (running mean square per layer): only the direction of a rule's credit matters."""
        ms[name] = 0.999 * ms[name] + 0.001 * float(np.mean(G * G))
        W += eta * G / (np.sqrt(ms[name] / (1 - 0.999 ** (t + 1))) + 1e-12)
    for t in range(trials):
        c = task.integers(N_CL); x = centres[c] + 0.3 * task.standard_normal(D)
        n1, n2, n3 = rng.standard_normal(H1), rng.standard_normal(H2), rng.standard_normal(K)
        a1 = W1 @ x + s1 * n1; h1 = np.maximum(a1, 0.0)
        v = W2 @ h1 + s2 * n2; h2 = softmax(v)
        u = W3 @ h2 + s3 * n3; pi = softmax(beta * u)
        if not np.all(np.isfinite(pi)):
            raise FloatingPointError(f"{rule} seed {seed}: diverged at trial {t}")
        act = rng.choice(K, p=pi); r = float(act == labels[c])
        delta = r - base; base += 0.01 * (r - base); rewards.append(r)
        relu_mask = (a1 > 0).astype(float)
        if rule == "perturbation":
            apply("W3", W3, delta * np.outer(n3 / s3, h2))
            apply("W2", W2, delta * np.outer(n2 / s2, h1))
            apply("W1", W1, delta * np.outer(n1 / s1 * relu_mask, x))
        elif rule in ("jacobian", "feedback_al", "jacobian_clean", "mixed_clean", "mixed_fa", "mixed_plain"):
            Cu = delta * n3 / s3
            F3, F2 = (B3, B2) if rule in ("feedback_al", "mixed_fa") else (J3, J2)
            Cv = F3.T @ Cu
            Ch = (F2.T @ Cv) * relu_mask
            if rule.startswith("mixed"):         # own perturbation credit + credit routed back (learned J or fixed B)
                Cv = Cv + delta * n2 / s2
                Ch = Ch + delta * n1 / s1 * relu_mask
            if clip is not None:                 # per-layer credit norm no larger than the output's
                lim = clip * np.linalg.norm(Cu) + 1e-12
                Cv = Cv * min(1.0, lim / (np.linalg.norm(Cv) + 1e-12))
                Ch = Ch * min(1.0, lim / (np.linalg.norm(Ch) + 1e-12))
            apply("W3", W3, np.outer(Cu, h2)); apply("W2", W2, np.outer(Cv, h1)); apply("W1", W1, np.outer(Ch, x))
            if rule in ("jacobian", "mixed_plain"):   # reward-free: how each layer's noise moves the next pre-activation
                J3 += j_rate * (np.outer(u - ubar, n2 / s2) - J3)
                J2 += j_rate * (np.outer(v - vbar, n1 / s1) - J2)
                ubar += 0.01 * (u - ubar); vbar += 0.01 * (v - vbar)
            elif rule in ("jacobian_clean", "mixed_clean"):   # against a noise-free pass of the same input (a shadow)
                h1c = np.maximum(W1 @ x, 0.0); vc = W2 @ h1c; uc = W3 @ softmax(vc)
                J3 += j_rate * (np.outer(u - uc, n2 / s2) - J3)
                J2 += j_rate * (np.outer(v - vc, n1 / s1) - J2)
        elif rule == "backprop":            # exact gradient of delta * log pi(a)
            g_u = beta * (np.eye(K)[act] - pi) * delta
            g_h2 = W3.T @ g_u
            g_v = h2 * (g_h2 - h2 @ g_h2)    # softmax Jacobian
            g_h1 = (W2.T @ g_v) * relu_mask
            apply("W3", W3, np.outer(g_u, h2)); apply("W2", W2, np.outer(g_v, h1)); apply("W1", W1, np.outer(g_h1, x))
    r = np.array(rewards)
    return r[: len(r) // block * block].reshape(-1, block).mean(1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=40000)
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--eta", type=float, default=0.003)
    ap.add_argument("--j-rate", type=float, default=0.001); ap.add_argument("--clip", type=float, default=None)
    ap.add_argument("--rules", nargs="+", default=["perturbation", "feedback_al", "jacobian", "backprop"])
    a = ap.parse_args()
    rules = a.rules
    curves = {}
    for rl in rules:
        try:
            curves[rl] = np.mean([run(rl, sd, a.trials, eta=a.eta, j_rate=a.j_rate, clip=a.clip)
                                  for sd in range(a.seeds)], 0)
        except FloatingPointError as e:
            print("DIVERGED:", e)
    n = len(next(iter(curves.values())))
    marks = [0, n // 8, n // 4, n // 2, n - 1]
    print("mean reward per 2000-trial block (chance 0.25), averaged over seeds")
    print(f"{'rule':14s}" + "".join(f"  block {m + 1:2d}" for m in marks))
    for rl, cv in curves.items():
        print(f"{rl:14s}" + "".join(f"  {cv[m]:8.3f}" for m in marks))


if __name__ == "__main__":
    main()
