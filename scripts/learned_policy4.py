"""The learned predictor of scripts/learned_policy3.py trained on Belady's decisions instead of reuse within a horizon
(the Hawkeye recipe): the label of expert e after step t is 1 if MIN with bypass at the same capacity keeps e resident
after step t. The cache then admits and evicts by the predicted probability that MIN would keep the expert, so the
policy imitates the optimum rather than a fixed-horizon reuse test.

    python scripts/learned_policy4.py --models gpt-oss-120b --budgets 8,4 --features x
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl.traces import load_pack  # noqa: E402
from scripts.provenance import MODELS, find, per_seq  # noqa: E402
from scripts.foresight import _pol, INF  # noqa: E402
from scripts.learned_policy import _trans  # noqa: E402
from scripts.learned_policy2 import features_all, labels_all, sim_p, make_model, NF as NF2  # noqa: E402
from scripts.learned_policy3 import trans_all, features_x, NF as NF3  # noqa: E402


@njit(cache=True)
def min_resident(R, E, C):
    """Y[t, e] = 1 if MIN with bypass (capacity C, single read) holds e after step t's requests; also (cpu, copies)."""
    T, k = R.shape
    nxt = np.full(E, INF, np.int64)
    nu_t = np.empty((T, k), np.int64)
    for t in range(T - 1, -1, -1):
        for j in range(k):
            nu_t[t, j] = nxt[R[t, j]]
        for j in range(k):
            nxt[R[t, j]] = t
    res = np.full(C, -1, np.int64)
    slot = np.full(E, -1, np.int64)
    nu = np.full(E, INF, np.int64)
    Y = np.zeros((T, E), np.float32)
    cpu = 0
    cp = 0
    for t in range(T):
        for j in range(k):
            nu[R[t, j]] = nu_t[t, j]
        for j in range(k):
            e = R[t, j]
            if slot[e] >= 0:
                continue
            fr = -1
            for s in range(C):
                if res[s] < 0:
                    fr = s
                    break
            if fr >= 0:
                res[fr] = e; slot[e] = fr; cp += 1
                continue
            v = -1; vkey = -1
            for s in range(C):
                c = res[s]
                inrow = False
                for jj in range(k):
                    if R[t, jj] == c:
                        inrow = True
                        break
                if inrow:
                    continue
                if nu[c] > vkey:
                    vkey = nu[c]; v = s
            if v < 0 or nu[e] >= vkey:
                cpu += 1
                continue
            old = res[v]
            slot[old] = -1
            res[v] = e; slot[e] = v; cp += 1
        for s in range(C):
            if res[s] >= 0:
                Y[t, res[s]] = 1.0
    return Y, cpu, cp


def run_model(name, routes, segs, E, k, kappa, C, kind, feats, H_mult=1.0, seed=0, n_train=300000):
    L, T, _ = routes.shape
    cut = segs[len(segs) // 2][0]
    rng = np.random.default_rng(seed)
    H = max(1, int(round(H_mult * C / k)))
    R3 = np.ascontiguousarray(routes, dtype=np.int64)
    PX = trans_all(R3, E, H, cut, 1) if feats == "x" else None
    nf = NF3 if feats == "x" else NF2
    Ms, pops, Xs, Ys = [], [], [], []
    per_layer = max(2000, n_train // L)
    for l in range(L):
        Rtr = np.ascontiguousarray(R3[l, :cut])
        M = _trans(Rtr, E, H)
        cnt = np.bincount(Rtr.ravel(), minlength=E).astype(np.float64)
        pop = np.log1p(cnt / max(1, cut))
        Ms.append(M); pops.append(pop)
        F = features_x(R3[:, :cut], l, E, M, PX, pop, 0, cut) if feats == "x" else features_all(Rtr, E, M, pop, 0, cut)
        Y, _, _ = min_resident(Rtr, E, C)
        tt = rng.integers(64, cut - 1, size=per_layer); ee = rng.integers(0, E, size=per_layer)
        tq = rng.integers(64, cut - 1, size=per_layer // 2); eq = Rtr[tq, rng.integers(0, k, size=per_layer // 2)]
        tt = np.concatenate([tt, tq]); ee = np.concatenate([ee, eq])
        Xs.append(F[tt, ee]); Ys.append(Y[tt, ee])
        del F, Y
    model = make_model(kind)
    model.fit(np.concatenate(Xs), np.concatenate(Ys))
    lr = 0; acc = {}; mn = 0
    for l in range(L):
        R = np.ascontiguousarray(R3[l])
        P = np.empty((T - cut, E), np.float32)
        for a in range(cut, T, 4000):
            b = min(T, a + 4000)
            F = features_x(R3[:, :b], l, E, Ms[l], PX, pops[l], a, b) if feats == "x" else features_all(R, E, Ms[l], pops[l], a, b)
            P[a - cut:b - cut] = model.predict_proba(F.reshape(-1, nf))[:, 1].reshape(b - a, E)
        c, f = sim_p(R, E, C, P, 0.0, cut)
        lr += c + f
        Rte = np.ascontiguousarray(R[cut:])
        _, c2, f2 = min_resident(Rte, E, C)
        mn += c2 + f2
        for key, W in (("dfa-fetch", -1), ("W1", 1), ("W4", 4), ("opt", INF)):
            for kap in sorted({0.0, kappa}):
                a_ = acc.setdefault(f"{key}@k{kap:g}", [0, 0])
                cc, pp = _pol(Rte, E, C, W, 16.0, kap)
                a_[0] += cc; a_[1] += pp
    Tte = T - cut
    row = {key: (c + f) / Tte for key, (c, f) in acc.items()}
    best = {key: min(v for kk, v in row.items() if kk.startswith(key + "@k")) for key in ("dfa-fetch", "W1", "W4", "opt")}
    lr /= Tte
    gap = (best["dfa-fetch"] - lr) / max(1e-9, best["dfa-fetch"] - best["opt"])
    print(f"   {name} C={C} {kind} feats={feats} (MIN labels): dfa-fetch {best['dfa-fetch']:7.2f} learned {lr:7.2f} W1 {best['W1']:7.2f}"
          f" W4 {best['W4']:7.2f} opt {best['opt']:7.2f} (min check {mn / Tte:7.2f}) | closed {100 * gap:5.1f}%", flush=True)
    return dict(learned=lr, H=H, kind=kind, feats=feats, gap_closed=gap, min_check=mn / Tte, **best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results"))
    ap.add_argument("--models", default="gpt-oss-120b")
    ap.add_argument("--budgets", default="8,4")
    ap.add_argument("--model", default="logit")
    ap.add_argument("--features", default="h")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = json.load(open(a.out)) if a.out and os.path.exists(a.out) else {}
    for key in a.models.split(","):
        ck, kappa, _ = MODELS[key]
        pk = load_pack(find(a.results, f"{ck}_S.npz"))
        win = per_seq(pk)
        idx = np.concatenate([np.arange(x, y) for _, x, y in win])
        segs, s0 = [], 0
        for _, x, y in win:
            segs.append((s0, s0 + (y - x))); s0 += y - x
        routes = pk["routes"][:, idx]
        E, k = int(pk["E"]), int(routes.shape[2])
        for d in a.budgets.split(","):
            C = max(k, E // int(d))
            for kind in a.model.split(","):
                for fs in a.features.split(","):
                    out[f"{key} C{C} {kind} {fs} min"] = run_model(key, routes, segs, E, k, float(kappa), C, kind, fs)
                    if a.out:
                        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
