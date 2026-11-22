import sys, numpy as np
d = np.load(sys.argv[1]); n_in, n_hid = 4, 3
off = ~np.eye(n_hid, dtype=bool)
post = np.concatenate([np.repeat(np.arange(n_hid), n_in), np.nonzero(off)[0]])
kind = np.array(["in"] * (n_in * n_hid) + ["rec"] * int(off.sum()))
p = post.size; k = d["fixed"].shape[0]
for name in ("null", "fixed", "track"):
    jac = d[name]; anti = jac - jac.transpose(0, 2, 1)
    z = anti.mean(0) / (anti.std(0, ddof=1) / np.sqrt(k))
    m = jac.mean(0); sym = 0.5 * (m + m.T)
    print(f"--- {name}")
    for label, sel in [("same post, in-in", lambda a, b: post[a] == post[b] and kind[a] == kind[b] == "in"),
                       ("diff post, in-in", lambda a, b: post[a] != post[b] and kind[a] == kind[b] == "in"),
                       ("involves rec", lambda a, b: "rec" in (kind[a], kind[b]))]:
        pairs = [(a, b) for a in range(p) for b in range(a + 1, p) if sel(a, b)]
        zz = np.array([abs(z[a, b]) for a, b in pairs])
        r = np.array([abs(m[a, b] - m[b, a]) / (abs(sym[a, b]) * 2 + 1e-30) for a, b in pairs])
        print(f"  {label:18s} n={len(pairs):3d}  |z|>3: {np.sum(zz > 3):3d}  median|z| {np.median(zz):5.2f}"
              f"  median |J_ab-J_ba|/|J_ab+J_ba| among |z|>3: {np.median(r[zz > 3]) if np.any(zz > 3) else float('nan'):.2f}")
