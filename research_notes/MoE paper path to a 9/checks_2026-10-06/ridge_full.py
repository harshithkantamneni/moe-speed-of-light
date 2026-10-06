"""Full-stack linear forecaster, no projection: ridge from the all-layer binary routing vector X_t (L*E) to X_{t+h},
h=1..8, folds by conversation parity (T-arm traces). Forecast = top-k per layer of (a) the ridge score, (b) a blend of
the ridge score's within-layer rank and the decayed-count rank (half-life 8). Baselines: persistence, decayed-count
top-k. Scored with pilot_forecast_sim.pol_fc (always-admit beyond the window) against always-admit and MIN."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, "/home/claude/moe-speed-of-light")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # pilot_forecast_sim.py
from pilot_forecast_sim import pol_fc, opt_reads
from mosl.traces import load_pack
model, pack, out = sys.argv[1], sys.argv[2], sys.argv[3]; lam = float(sys.argv[4])
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
H = 8
pr = np.full((L, T, H, k), -1, np.int64); pb = np.full((L, T, H, k), -1, np.int64)
hits = {n: np.zeros(H) for n in ("ridge", "blend", "persist", "freq")}; cnt = np.zeros(H)
t0 = time.time()
for fold in (0, 1):
    tr = np.where(conv % 2 != fold)[0]; te = np.where(conv % 2 == fold)[0]
    mu = X[tr].mean(0); Xs_tr = X[tr] - mu
    G = (Xs_tr.T @ Xs_tr).astype(np.float64); G[np.diag_indices_from(G)] += lam
    Lc = np.linalg.cholesky(G); del G
    for h in range(1, H + 1):
        ok = (tr + h < T); ok[ok] = conv[tr[ok] + h] == conv[tr[ok]]
        B = (Xs_tr[ok].T @ (X[tr[ok] + h] - mu)).astype(np.float64)
        Wt = np.linalg.solve(Lc.T, np.linalg.solve(Lc, B)).astype(np.float32); del B
        valid = (te + h < T); valid[valid] = conv[te[valid] + h] == conv[te[valid]]
        tv = te[valid]
        S_ = (X[tv] - mu) @ Wt + mu; del Wt
        for l in range(L):
            sl_ = S_[:, l * E:(l + 1) * E]; dl = Xd[tv, l * E:(l + 1) * E]
            top = np.argsort(-sl_, 1)[:, :k]
            rr = np.argsort(np.argsort(-sl_, 1), 1); rd = np.argsort(np.argsort(-(dl + 1e-6 * sl_), 1), 1)
            btop = np.argsort(rr + rd, 1)[:, :k]
            ftop = np.argsort(-dl, 1)[:, :k]
            pr[l, tv, h - 1] = top; pb[l, tv, h - 1] = btop
            tgt = R[l, tv + h]
            for n, fc in (("ridge", top), ("blend", btop), ("persist", R[l, tv]), ("freq", ftop)):
                hits[n][h - 1] += (fc[:, :, None] == tgt[:, None, :]).any(2).sum()
            cnt[h - 1] += len(tv) * k
        del S_
        print(model, "fold", fold, "h", h, f"{time.time() - t0:.0f}s", flush=True)
    del Lc, Xs_tr
res = {"model": model, "E": E, "k": k, "L": L, "T": T, "lambda": lam, "prec_at_k": {n: (v / cnt).round(4).tolist() for n, v in hits.items()}, "budgets": {}}
print(model, "precision@k by h:", {n: np.round(v / cnt, 3).tolist() for n, v in hits.items()}, flush=True)
for C in (E // 8, E // 4):
    Bd = {"opt": 0, "AA": 0}
    for l in range(L):
        Rl = np.ascontiguousarray(R[l])
        c, cp = opt_reads(Rl, E, C); Bd["opt"] += c + cp
        c, cp = pol_fc(Rl, np.full((T, 1, k), -1, np.int64), E, C, 1, 16.0, -1e18); Bd["AA"] += c + cp
        for W in (1, 2, 4, 8):
            for n, P_ in (("ridge", pr), ("blend", pb)):
                c, cp = pol_fc(Rl, np.ascontiguousarray(P_[l, :, :W, :]), E, C, W, 16.0, -1e18); Bd[f"{n}@W{W}"] = Bd.get(f"{n}@W{W}", 0) + c + cp
    Bd = {a: v / T for a, v in Bd.items()}; gap = Bd["AA"] - Bd["opt"]
    Bd["closed_vs_AA"] = {a: round((Bd["AA"] - v) / gap, 3) for a, v in Bd.items() if "@" in a}
    res["budgets"][C] = Bd
    print(model, "C", C, "AA", round(Bd["AA"], 2), "opt", round(Bd["opt"], 2), Bd["closed_vs_AA"], flush=True)
json.dump(res, open(out, "w"), indent=1)
