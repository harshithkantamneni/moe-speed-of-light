"""h=1 only: recall@k and recall@(k+3) of the full-stack ridge forecaster (same setup as ridge_full.py) and of the
decayed-count and persistence forecasts, for comparison with SeqMoE's recall@(k+3) metric."""
import json, sys
import numpy as np
sys.path.insert(0, "/home/claude/moe-speed-of-light")
from mosl.traces import load_pack
model, pack, lam = sys.argv[1], sys.argv[2], float(sys.argv[3])
p = load_pack(pack); routes = p["routes"]; L, Ttot, k = routes.shape; E = int(p["E"])
st, pl, sl = p["starts"], p["prompt_lens"], p["seq_lens"]
resp = np.concatenate([np.arange(s + a, s + n) for s, a, n in zip(st, pl, sl)])
conv = np.concatenate([[i] * (n - a) for i, (a, n) in enumerate(zip(pl, sl))])
T = len(resp); R = routes[:, resp].astype(np.int64)
X = np.zeros((T, L * E), np.float32)
for l in range(L): X[np.arange(T)[:, None], l * E + R[l]] = 1.0
Xd = np.zeros_like(X); acc = np.zeros(L * E, np.float32); dec = np.float32(0.5 ** (1 / 8))
for t in range(T):
    if t == 0 or conv[t] != conv[t - 1]: acc[:] = 0
    acc = acc * dec + X[t]; Xd[t] = acc
h = 1; m = k + 3
hits = {f"{n}@{w}": 0 for n in ("ridge", "freq") for w in (k, m)}; tot = 0
for fold in (0, 1):
    tr = np.where(conv % 2 != fold)[0]; te = np.where(conv % 2 == fold)[0]
    mu = X[tr].mean(0); Xs = X[tr] - mu
    G = (Xs.T @ Xs).astype(np.float64); G[np.diag_indices_from(G)] += lam
    ok = (tr + h < T); ok[ok] = conv[tr[ok] + h] == conv[tr[ok]]
    Wt = np.linalg.solve(G, (Xs[ok].T @ (X[tr[ok] + h] - mu)).astype(np.float64)).astype(np.float32); del G, Xs
    valid = (te + h < T); valid[valid] = conv[te[valid] + h] == conv[te[valid]]; tv = te[valid]
    S_ = (X[tv] - mu) @ Wt + mu
    for l in range(L):
        tgt = R[l, tv + 1]
        o_r = np.argsort(-S_[:, l * E:(l + 1) * E], 1); o_f = np.argsort(-Xd[tv, l * E:(l + 1) * E], 1)
        for n, o in (("ridge", o_r), ("freq", o_f)):
            for w in (k, m):
                hits[f"{n}@{w}"] += (o[:, :w][:, :, None] == tgt[:, None, :]).any(1).sum()
        tot += len(tv) * k
print(model, "h=1 recall:", {a: round(v / tot, 3) for a, v in hits.items()})
