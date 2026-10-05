"""The learned reuse predictor of scripts/learned_policy2.py with the current token's routing at every layer as
evidence: besides the same-layer transition score, a cross-layer score for expert e of layer l, the mean over the
L*k experts the token requested at all layers of P(e is requested at layer l within the next H steps | that expert
was requested now), counted on the training conversations. The routing of the whole token is known to the engine at
the end of the step, so the score is available online.

    python scripts/learned_policy3.py --models gpt-oss-120b --budgets 8,4
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
from scripts.learned_policy2 import labels_all, sim_p, make_model  # noqa: E402

HL = np.array([1.0, 4.0, 16.0, 64.0, 256.0])
NF = 11


@njit(cache=True)
def trans_all(routes, E, H, cut, stride):
    """P[l', i, l, e] = P(e at layer l within (t, t+H] | expert i requested at layer l' at t), t < cut - H."""
    L, T, k = routes.shape
    num = np.zeros((L, E, L, E), np.float32)
    den = np.zeros((L, E), np.float32)
    stamp = np.zeros((L, E), np.int64)
    ys_l = np.empty(L * E, np.int64); ys_e = np.empty(L * E, np.int64)
    for t in range(0, cut - 1, stride):
        ny = 0
        for tt in range(t + 1, min(cut, t + 1 + H)):
            for l in range(L):
                for j in range(k):
                    e = routes[l, tt, j]
                    if stamp[l, e] != t + 1:
                        stamp[l, e] = t + 1
                        ys_l[ny] = l; ys_e[ny] = e; ny += 1
        for lp in range(L):
            for j in range(k):
                i = routes[lp, t, j]
                den[lp, i] += 1
                for q in range(ny):
                    num[lp, i, ys_l[q], ys_e[q]] += 1
    for lp in range(L):
        for i in range(E):
            if den[lp, i] > 0:
                for l in range(L):
                    for e in range(E):
                        num[lp, i, l, e] /= den[lp, i]
    return num


@njit(cache=True)
def features_x(routes, l, E, M, PX, pop, t0, t1):
    L, T, k = routes.shape
    nh = HL.shape[0]
    d = np.empty(nh)
    for h in range(nh):
        d[h] = 0.5 ** (1.0 / HL[h])
    sc = np.zeros((nh, E)); last = np.full(E, -1, np.int64)
    F = np.zeros((t1 - t0, E, NF), np.float32)
    for t in range(t1):
        for j in range(k):
            e = routes[l, t, j]
            for h in range(nh):
                sc[h, e] = (sc[h, e] * d[h] ** (t - last[e]) if last[e] >= 0 else 0.0) + 1.0
            last[e] = t
        if t < t0:
            continue
        ts = np.zeros(E); tx = np.zeros(E); tmax = np.zeros(E)
        for j in range(k):
            i = routes[l, t, j]
            for e in range(E):
                ts[e] += M[i, e]
        for lp in range(L):
            for j in range(k):
                i = routes[lp, t, j]
                for e in range(E):
                    v = PX[lp, i, l, e]
                    tx[e] += v
                    if v > tmax[e]:
                        tmax[e] = v
        for e in range(E):
            for h in range(nh):
                F[t - t0, e, h] = np.log1p(sc[h, e] * d[h] ** (t - last[e])) if last[e] >= 0 else 0.0
            F[t - t0, e, nh] = np.log1p(t - last[e]) if last[e] >= 0 else 12.0
            F[t - t0, e, nh + 1] = 1.0 if last[e] == t else 0.0
            F[t - t0, e, nh + 2] = ts[e] / k
            F[t - t0, e, nh + 3] = pop[e]
            F[t - t0, e, nh + 4] = tx[e] / (L * k)
            F[t - t0, e, nh + 5] = tmax[e]
    return F


def run_model(name, routes, segs, E, k, kappa, C, kind, H_mult, seed=0, n_train=300000, stride=1):
    L, T, _ = routes.shape
    cut = segs[len(segs) // 2][0]
    rng = np.random.default_rng(seed)
    H = max(1, int(round(H_mult * C / k)))
    R3 = np.ascontiguousarray(routes, dtype=np.int64)
    t0 = time.time()
    PX = trans_all(R3, E, H, cut, stride)
    tpx = time.time() - t0
    Ms, pops, Xs, Ys = [], [], [], []
    per_layer = max(2000, n_train // L)
    for l in range(L):
        Rtr = np.ascontiguousarray(R3[l, :cut])
        M = _trans(Rtr, E, H)
        cnt = np.bincount(Rtr.ravel(), minlength=E).astype(np.float64)
        pop = np.log1p(cnt / max(1, cut))
        Ms.append(M); pops.append(pop)
        F = features_x(R3[:, :cut], l, E, M, PX, pop, 0, cut)
        Y = labels_all(Rtr, E, H, 0, cut)
        tt = rng.integers(64, cut - 1, size=per_layer); ee = rng.integers(0, E, size=per_layer)
        tq = rng.integers(64, cut - 1, size=per_layer // 2); eq = Rtr[tq, rng.integers(0, k, size=per_layer // 2)]
        tt = np.concatenate([tt, tq]); ee = np.concatenate([ee, eq])
        Xs.append(F[tt, ee]); Ys.append(Y[tt, ee])
        del F, Y
    model = make_model(kind)
    model.fit(np.concatenate(Xs), np.concatenate(Ys))
    lr = 0; acc = {}
    for l in range(L):
        R = np.ascontiguousarray(R3[l])
        P = np.empty((T - cut, E), np.float32)
        for a in range(cut, T, 4000):
            b = min(T, a + 4000)
            F = features_x(R3[:, :b], l, E, Ms[l], PX, pops[l], a, b)
            P[a - cut:b - cut] = model.predict_proba(F.reshape(-1, NF))[:, 1].reshape(b - a, E)
        c, f = sim_p(R, E, C, P, 0.0, cut)
        lr += c + f
        Rte = np.ascontiguousarray(R[cut:])
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
    coef = getattr(model, "coef_", None)
    print(f"   {name} C={C} {kind} H={H} (cross-layer): dfa-fetch {best['dfa-fetch']:7.2f} learned {lr:7.2f} W1 {best['W1']:7.2f} W4 {best['W4']:7.2f}"
          f" opt {best['opt']:7.2f} | closed {100 * gap:5.1f}% (PX {tpx:.0f} s){'' if coef is None else ' coef ' + str(np.round(coef[0], 2))}", flush=True)
    return dict(learned=lr, H=H, kind=kind, gap_closed=gap, **best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results"))
    ap.add_argument("--models", default="gpt-oss-120b")
    ap.add_argument("--budgets", default="8,4")
    ap.add_argument("--model", default="logit")
    ap.add_argument("--hmult", default="1")
    ap.add_argument("--stride", type=int, default=1)
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
                for hm in a.hmult.split(","):
                    out[f"{key} C{C} {kind} H{hm} x"] = run_model(key, routes, segs, E, k, float(kappa), C, kind, float(hm), stride=a.stride)
                    if a.out:
                        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
