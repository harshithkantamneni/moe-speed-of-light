"""Per-layer slot allocation for the expert cache (LLAMA_EC_ALLOC).

Every MoE layer gets one of: 0 slots (experts run on the CPU, as in llama.cpp layer
offload), E slots (the whole layer on the GPU), or C slots in between (a cached layer:
resident experts on the GPU, the rest on the CPU, both paths every step). A dynamic
program picks the per-layer sizes that minimise the predicted step time under a total
slot budget, so the stock layer-offload layout is always in the search space.

Per-layer expert time (microseconds), k experts selected per token:
    E slots : k*a
    0 slots : k*a + s                        (s: measured cost of offloading one layer)
    C slots : o + comb(p*k*a, (1-p)*k*b) + q*c (cached layer)
where b = a + s/k is the CPU cost of one expert consistent with s, p is the fraction of
the layer's requests the C slots serve, o the cached layer's fixed overhead, comb = max
(CPU/GPU overlap) or + (serial), q admissions per step in the layer and c the critical-
path cost of one admission.

    python scripts/ec_alloc.py --stats profile.json --budget 192 --a 38 --s 395 --o 84 \
        [--hits hits.json] [--serial] [--c 0] --out alloc.txt
"""
import argparse
import json

import numpy as np


def coverage(counts, C):
    c = np.sort(np.asarray(counts, float))[::-1]
    return float(c[:C].sum() / c.sum()) if c.sum() > 0 else 0.0


def layer_cost(C, E, k, a, s, o, p, serial, q=0.0, c=0.0):
    if C == E:
        return k * a
    if C == 0:
        return k * a + s
    b = a + s / k
    g, cpu = p * k * a, (1 - p) * k * b
    return o + (g + cpu if serial else max(g, cpu)) + q * c


def allocate(layers, budget, a, s, o, serial=False, choices=None, hit=None, adm=None, c=0.0):
    """layers: list of dict(il, E, k, counts). hit/adm: optional {il: {C: value}} from a
    simulated dynamic policy; otherwise the static top-C coverage of the profile counts."""
    L = len(layers)
    INF = float("inf")
    # dp[i][b] = best cost of the first i layers using b slots
    dp = np.full((L + 1, budget + 1), INF)
    dp[0, 0] = 0.0
    arg = [[None] * (budget + 1) for _ in range(L + 1)]
    for i, ly in enumerate(layers):
        E, k = ly["E"], ly["k"]
        opts = sorted(set([0, E] + [C for C in (choices or range(1, E)) if 0 < C < E]))
        costs = {}
        for C in opts:
            if hit is not None and C not in (0, E) and C in hit.get(ly["il"], {}):
                p = hit[ly["il"]][C]
                q = adm[ly["il"]][C] if adm else 0.0
            else:
                p, q = coverage(ly["counts"], C), 0.0
            costs[C] = layer_cost(C, E, k, a, s, o, p, serial, q, c)
        for b0 in range(budget + 1):
            if dp[i, b0] == INF:
                continue
            for C, t in costs.items():
                b1 = b0 + C
                if b1 <= budget and dp[i, b0] + t < dp[i + 1, b1]:
                    dp[i + 1, b1] = dp[i, b0] + t
                    arg[i + 1][b1] = (b0, C)
    b_best = int(np.argmin(dp[L]))
    alloc, b = [], b_best
    for i in range(L, 0, -1):
        b0, C = arg[i][b]
        alloc.append(C)
        b = b0
    alloc = alloc[::-1]
    return dict(alloc={ly["il"]: C for ly, C in zip(layers, alloc)}, predicted_us=float(dp[L, b_best]), slots=b_best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--a", type=float, required=True)
    ap.add_argument("--s", type=float, required=True)
    ap.add_argument("--o", type=float, required=True)
    ap.add_argument("--c", type=float, default=0.0)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--serial", action="store_true")
    ap.add_argument("--hits", help="json {il: {C: hit_fraction}} (and optional adm) from a simulated policy")
    ap.add_argument("--choices", default="", help="comma list of cached-layer sizes to consider")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    st = json.load(open(args.stats))
    layers = [dict(il=l["il"], E=len(l["count"]), k=args.k, counts=l["count"]) for l in st["layers"]]
    hit = adm = None
    if args.hits:
        h = json.load(open(args.hits))
        hit = {int(il): {int(C): v for C, v in d.items()} for il, d in h["hit"].items()}
        adm = {int(il): {int(C): v for C, v in d.items()} for il, d in h.get("adm", {}).items()} or None
    choices = [int(x) for x in args.choices.split(",") if x] or None
    r = allocate(layers, args.budget, args.a, args.s, args.o, args.serial, choices, hit, adm, args.c)
    with open(args.out, "w") as f:
        for il, C in r["alloc"].items():
            f.write(f"{il} {C}\n")
    kinds = {"gpu": sum(C == ly["E"] for ly, C in zip(layers, r["alloc"].values())),
             "cpu": sum(C == 0 for C in r["alloc"].values())}
    print(json.dumps(dict(predicted_ms=r["predicted_us"] / 1000, slots=r["slots"], **kinds,
                          cached=len(layers) - kinds["gpu"] - kinds["cpu"], alloc=list(r["alloc"].values()))))


if __name__ == "__main__":
    main()
