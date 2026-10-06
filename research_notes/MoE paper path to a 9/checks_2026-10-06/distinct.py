import json, sys
import numpy as np
from numba import njit
sys.path.insert(0, "/home/claude/moe-speed-of-light")
from scripts.foresight_exact import jobs_from
from scripts.provenance import MODELS
WS = np.array([1, 2, 4, 8, 16, 32, 64])
@njit(cache=True)
def distinct(R, E, Wmax):
    T, k = R.shape
    out = np.zeros(Wmax + 1)
    seen = np.zeros(E, np.int64)
    stamp = 0
    n = 0
    for t in range(0, T - Wmax, 3):
        stamp += 1
        d = 0
        for w in range(1, Wmax + 1):
            for j in range(k):
                e = R[t + w - 1, j]
                if seen[e] != stamp:
                    seen[e] = stamp
                    d += 1
            out[w] += d
        n += 1
    return out / max(n, 1)
w50 = json.load(open(sys.argv[1]))
pts = {(p["model"], p["C"]): p["w50"] for p in w50["points"]}
res = {}
for name, routes, segs, E, k, kappa in jobs_from("/home/claude/gpu-branch/results", None, "S", list(MODELS)):
    L = routes.shape[0]
    D = np.zeros(65)
    for l in range(L):
        D += distinct(np.ascontiguousarray(routes[l], dtype=np.int64), int(E), 64)
    D /= L
    beta = np.polyfit(np.log(WS[:5]), np.log(D[WS[:5]]), 1)[0]
    res[name] = dict(E=int(E), k=int(k), beta=float(beta), D=[float(D[w]) for w in WS])
    for (m, C), w in pts.items():
        if m != name: continue
        # D at W50 by log-linear interpolation on integer W (W50 < 1 -> scale k*W50)
        if w < 1: d = k * w
        else:
            lw = np.log(w); xs = np.log(np.arange(1, 65)); d = float(np.exp(np.interp(lw, xs, np.log(D[1:65]))))
        res[name].setdefault("pts", []).append(dict(C=C, Ck=C / k, W50=w, D_at_W50=d, D_over_C=d / C, newD_over_C=(d - k) / C))
    print(name, "k", k, "beta(W1-16)", round(beta, 3), "D(W)", [round(D[w], 1) for w in WS], [(p["C"], round(p["W50"], 2), round(p["D_over_C"], 2)) for p in res[name].get("pts", [])], flush=True)
allr = [p["D_over_C"] for v in res.values() for p in v.get("pts", [])]
alln = [p["newD_over_C"] for v in res.values() for p in v.get("pts", [])]
allw = [p["W50"] for v in res.values() for p in v.get("pts", [])]
print("D(W50)/C: median %.2f  IQR %.2f-%.2f  CV %.2f | (D-k)/C median %.2f CV %.2f | W50 CV %.2f | betas %s" % (np.median(allr), *np.percentile(allr, [25, 75]), np.std(allr) / np.mean(allr), np.median(alln), np.std(alln)/np.mean(alln), np.std(allw)/np.mean(allw), [round(v['beta'], 2) for v in res.values()]))
json.dump(res, open(sys.argv[2], "w"), indent=1)
