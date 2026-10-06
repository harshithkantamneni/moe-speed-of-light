import json, sys, numpy as np
d = json.load(open(sys.argv[1]))
WS = np.array([1, 2, 4, 8, 16, 32, 64])
rows = [(m, p) for m, v in d.items() for p in v.get("pts", [])]
def inv(v, target):
    k = v["k"]; D = np.array(v["D"])
    if target <= k: return target / k
    return float(np.exp(np.interp(np.log(target), np.log(D), np.log(WS))))
errs = {"power": [], "drule": [], "linear": []}
for hold in d:
    tr = [(m, p) for m, p in rows if m != hold]; te = [(m, p) for m, p in rows if m == hold]
    x = np.log([p["Ck"] for _, p in tr]); y = np.log([p["W50"] for _, p in tr])
    b, a = np.polyfit(x, y, 1)
    a1 = np.median([p["W50"] / p["Ck"] for _, p in tr])
    c = np.median([p["D_over_C"] for _, p in tr])
    for m, p in te:
        errs["power"].append(abs(np.log(np.exp(a) * p["Ck"] ** b / p["W50"])))
        errs["linear"].append(abs(np.log(a1 * p["Ck"] / p["W50"])))
        errs["drule"].append(abs(np.log(inv(d[m], c * p["C"]) / p["W50"])))
for k, v in errs.items():
    v = np.array(v); print(f"{k:7s} LOMO |log error|: median {np.median(v):.3f} (x{np.exp(np.median(v)):.2f}), p90 {np.percentile(v,90):.3f} (x{np.exp(np.percentile(v,90)):.2f}), max x{np.exp(v.max()):.2f}, n={len(v)}")
