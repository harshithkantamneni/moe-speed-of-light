"""Pilot: how much of the online->MIN read gap do *realisable-style* routing forecasts close?

Uses the T-arm traces (dataset text, token ids known) of gpt-oss-120b / qwen3-30b-a3b / gpt-oss-20b.
Policy = scripts/foresight.py::_pol, but "next use within the window" is taken from a forecast F[t, h] (h = 1..W)
made at step t, instead of the exact future routing.

Forecasters (per layer):
  exact      F[t,h] = R[t+h]                                  (sanity: should equal foresight.py's W rows)
  tok        F[t,h] = top-k experts of P(expert | token id of step t+h), table fit on the other half of the
             conversations (prompt+response tokens); unseen ids -> global top-k.  = a perfect drafter + a token table
  tokA{a}    tok, but the drafter's tokens are right i.i.d. with prob a per step; forecast empty after first miss
  persist    F[t,h] = R[t]
  noisy{r}   F[t,h] = exact with each true expert kept w.p. r, plus (k - kept) random false experts (precision = r)
"""
import json
import sys

import numpy as np
from numba import njit

sys.path.insert(0, "/home/claude/moe-speed-of-light")
from mosl.traces import load_pack  # noqa: E402

INF = 1 << 40


@njit(cache=True)
def pol_fc(R, F, E, C, W, hl, kappa):
    # R [T,k] actual; F [T,W,m] forecast for steps t+1..t+W (-1 = none)
    T, k = R.shape
    m = F.shape[2]
    decay = 0.5 ** (1.0 / hl)
    res = np.full(C, -1, np.int64)
    slot = np.full(E, -1, np.int64)
    sc = np.zeros(E)
    last = np.zeros(E, np.int64)
    nu = np.full(E, INF, np.int64)
    cpu = 0
    cp = 0
    for t in range(T):
        for j in range(k):
            e = R[t, j]
            sc[e] = sc[e] * decay ** (t - last[e]) + 1.0
            last[e] = t
        # forecast next use (within window) for every expert
        for e in range(E):
            nu[e] = INF
        for h in range(W - 1, -1, -1):
            for j in range(m):
                e = F[t, h, j]
                if e >= 0:
                    nu[e] = t + 1 + h
        lim = t + W
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
                res[fr] = e
                slot[e] = fr
                cp += 1
                continue
            v = -1
            vbeyond = False
            vkey = 0.0
            for s in range(C):
                c = res[s]
                inrow = False
                for jj in range(k):
                    if R[t, jj] == c:
                        inrow = True
                        break
                if inrow:
                    continue
                beyond = nu[c] > lim
                if beyond:
                    key = sc[c] * decay ** (t - last[c])
                    if (not vbeyond) or key < vkey:
                        v = s
                        vbeyond = True
                        vkey = key
                elif not vbeyond:
                    key = float(nu[c])
                    if v < 0 or key > vkey:
                        v = s
                        vkey = key
            if v < 0:
                cpu += 1
                continue
            if nu[e] > lim:
                if not (vbeyond and sc[e] > vkey + kappa):
                    cpu += 1
                    continue
            elif (not vbeyond) and float(nu[e]) >= vkey:
                cpu += 1
                continue
            old = res[v]
            slot[old] = -1
            res[v] = e
            slot[e] = v
            cp += 1
    return cpu, cp


@njit(cache=True)
def opt_reads(R, E, C):
    # MIN with bypass == _pol with whole-future foresight; implemented via exact forecast of all future is too
    # big, so use the standard next-use formulation
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
                res[fr] = e
                slot[e] = fr
                cp += 1
                continue
            v = -1
            vkey = -1
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
                    vkey = nu[c]
                    v = s
            if v < 0 or nu[e] >= vkey:
                cpu += 1
                continue
            old = res[v]
            slot[old] = -1
            res[v] = e
            slot[e] = v
            cp += 1
    return cpu, cp


def main(model, packpath, tokpath, kappa):
    p = load_pack(packpath)
    rows = [json.loads(l) for l in open(tokpath)]
    ids_all = np.concatenate([np.array(r["ids"], np.int64) for r in rows])
    routes = p["routes"]  # [L, Ttot, k]
    L, Ttot, k = routes.shape
    E = p["E"]
    assert len(ids_all) == Ttot
    starts = p["starts"]
    conv = np.concatenate([[i] * n for i, n in enumerate(p["seq_lens"])])
    resp = np.concatenate([np.arange(s + pl, s + n) for s, pl, n in zip(starts, p["prompt_lens"], p["seq_lens"])])
    fold = conv % 2
    rng = np.random.default_rng(0)
    V = int(ids_all.max()) + 1
    out = {"model": model, "E": E, "k": k, "L": L, "T": int(len(resp)), "budgets": {}}
    Ws = (1, 2, 4, 8)
    Wmax = max(Ws)
    budgets = (E // 8, E // 4, E // 2)
    # token tables per layer per fold: counts [V, E] is too big for V=200k x E=128 x L ... use dict of top-k
    tok_fc = np.full((L, Ttot, k), -1, np.int64)  # predicted top-k for each position given its true token
    for l in range(L):
        for f in (0, 1):
            tr = np.where(fold != f)[0]
            te = np.where(fold == f)[0]
            ids_tr = ids_all[tr]
            r_tr = routes[l, tr]
            # counts via sparse key token*E+expert
            keys = (ids_tr[:, None] * E + r_tr).ravel()
            uk, cnt = np.unique(keys, return_counts=True)
            tok_of = uk // E
            ex_of = uk % E
            gl = np.bincount(r_tr.ravel(), minlength=E)
            gtop = np.argsort(-gl)[:k]
            order = np.lexsort((-cnt, tok_of))
            tok_s, ex_s = tok_of[order], ex_of[order]
            first = np.searchsorted(tok_s, np.unique(tok_s))
            table = {}
            ut = np.unique(tok_s)
            bounds = np.append(first, len(tok_s))
            for i, tkn in enumerate(ut):
                exs = ex_s[bounds[i]:bounds[i + 1]][:k]
                if len(exs) < k:
                    exs = np.concatenate([exs, [g for g in gtop if g not in exs][: k - len(exs)]])
                table[int(tkn)] = exs
            for pos in te:
                tok_fc[l, pos] = table.get(int(ids_all[pos]), gtop)
    for C in budgets:
        B = {}
        for l in range(L):
            R = np.ascontiguousarray(routes[l, resp])
            T = len(R)
            TF = tok_fc[l, resp]
            o_cpu, o_cp = opt_reads(R, E, C)
            B.setdefault("opt", 0)
            B["opt"] += o_cpu + o_cp
            c0, cp0 = pol_fc(R, np.full((T, 1, k), -1, np.int64), E, C, 1, 16.0, kappa)
            B.setdefault("W0", 0)
            B["W0"] += c0 + cp0
            for W in Ws:
                idx = np.arange(T)[:, None] + 1 + np.arange(W)[None, :]
                valid = idx < T
                idxc = np.minimum(idx, T - 1)
                Fex = np.where(valid[:, :, None], R[idxc], -1)
                Ftok = np.where(valid[:, :, None], TF[idxc], -1)
                Fper = np.where(valid[:, :, None], np.broadcast_to(R[:, None, :], (T, W, k)), -1)
                fcs = {"exact": Fex, "tok": Ftok, "persist": Fper}
                for a in (0.5, 0.75, 0.9):
                    ok = rng.random((T, W)) < a
                    alive = np.cumprod(ok, axis=1).astype(bool)
                    fcs[f"tokA{a}"] = np.where(alive[:, :, None], Ftok, -1)
                for r in (0.5, 0.7, 0.9):
                    keep = rng.random(Fex.shape) < r
                    fake = rng.integers(0, E, Fex.shape)
                    fcs[f"noisy{r}"] = np.where(Fex >= 0, np.where(keep, Fex, fake), -1)
                for name, F in fcs.items():
                    cpu, cp = pol_fc(R, np.ascontiguousarray(F), E, C, W, 16.0, kappa)
                    key = f"{name}@W{W}"
                    B[key] = B.get(key, 0) + cpu + cp
        Tn = len(resp)
        B = {kk: v / Tn for kk, v in B.items()}
        gap = B["W0"] - B["opt"]
        B["closed"] = {kk: round((B["W0"] - v) / gap, 3) for kk, v in B.items() if "@" in kk}
        out["budgets"][C] = B
        print(model, "C", C, "W0", round(B["W0"], 2), "opt", round(B["opt"], 2), flush=True)
        for W in Ws:
            print("  W", W, {n: B["closed"][f"{n}@W{W}"] for n in
                             ["exact", "tok", "tokA0.9", "tokA0.75", "tokA0.5", "persist", "noisy0.9", "noisy0.7", "noisy0.5"]},
                  flush=True)
    # token-table precision at h (fraction of true experts in predicted top-k), response positions
    prec = []
    for l in range(L):
        R = routes[l, resp]
        TF = tok_fc[l, resp]
        prec.append(np.mean([len(set(a) & set(b)) / k for a, b in zip(R[::7], TF[::7])]))
    out["tok_table_precision_by_layer"] = [round(float(x), 3) for x in prec]
    print(model, "token-table precision@k mean", round(float(np.mean(prec)), 3), "first/last layers",
          [round(float(x), 2) for x in prec[:3]], [round(float(x), 2) for x in prec[-3:]], flush=True)
    return out


if __name__ == "__main__":
    model, pack, tok, kappa, dst = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
    o = main(model, pack, tok, kappa)
    json.dump(o, open(dst, "w"), indent=1, default=float)
