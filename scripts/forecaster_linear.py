"""A realisable forecaster on the value map: ridge regression from the all-layer routing of step t (a binary vector of
L*E) to the routing of step t+h, h = 1..8, fitted on the model's mixed-domain own-text trace (arm S; no AIME text) and
applied to the AIME-25 routing the engine runs (jobs 084b/084c). Forecast = the top-k experts of each layer. Baselines
on the same footing: persistence (step t's experts for every h) and the decayed request count (half-life 8, top-k).
Each forecast drives the window policy of scripts/value_map.py (replay_fc, admit-every-miss beyond the window), and is
scored by precision at k and by the share of the admit-every-miss -> MIN read gap it closes.
Writes prereg/forecaster_linear.json and paper/wsg_forecaster.tex.

    python scripts/forecaster_linear.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl.traces import load_pack  # noqa: E402
from scripts.provenance import MODELS, find  # noqa: E402
from scripts.value_map import replay, replay_fc  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
RES = "/home/claude/gpu-branch/results"
AIME = {"gpt-oss-120b": ("084c_gptoss_trace@vast/route_aime25_gptoss.npz", (14, 32)),
        "qwen3-30b-a3b": ("084b_vram_rerun@vast/route_aime25_qwen3.npz", (16, 32))}
H = 8
LAM = 100.0
MAXTRAIN = {"gpt-oss-120b": 40000, "qwen3-30b-a3b": 24000}   # all of gpt-oss's S text; Qwen3's first 24,000 tokens (memory)


def onehot(R, E):
    """R [L, T, k] -> X [T, L*E] float32"""
    L, T, k = R.shape
    X = np.zeros((T, L * E), np.float32)
    for l in range(L):
        X[np.arange(T)[:, None], l * E + R[l]] = 1.0
    return X


def main():
    pj = P("prereg", "forecaster_linear.json")
    out = json.load(open(pj)) if os.path.exists(pj) else {}
    M = {}
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else list(AIME)
    for key, (rel, Cs) in AIME.items():
        if key not in only:
            continue
        t0 = time.time()
        ck = MODELS[key][0]
        pk = load_pack(find(RES, f"{ck}_S.npz"))
        E = int(pk["E"])
        resp = np.concatenate([np.arange(s + a, s + n) for s, a, n in zip(pk["starts"], pk["prompt_lens"], pk["seq_lens"])])
        conv = np.concatenate([[i] * (int(n) - int(a)) for i, (a, n) in enumerate(zip(pk["prompt_lens"], pk["seq_lens"]))])
        if len(resp) > MAXTRAIN[key]:
            resp, conv = resp[:MAXTRAIN[key]], conv[:MAXTRAIN[key]]
        Rtr = pk["routes"][:, resp].astype(np.int64)
        L, Ttr, k = Rtr.shape
        Xtr = onehot(Rtr, E)
        mu = Xtr.mean(0)
        # centred Gram without a centred copy: X'X - n mu mu'
        G = (Xtr.T @ Xtr).astype(np.float64) - Ttr * np.outer(mu, mu)
        G[np.diag_indices_from(G)] += LAM
        Lc = np.linalg.cholesky(G)
        del G
        z = np.load(os.path.join(RES, rel))
        act = z["act"].astype(np.int64)            # [T, L, k]
        seqs = z["seq"].astype(np.int64)
        T = act.shape[0]
        Xte = onehot(act.transpose(1, 0, 2), E)
        Fr = np.full((L, T, H, k), -1, np.int64)
        for h in range(1, H + 1):
            ok = np.arange(Ttr - h)
            ok = ok[conv[ok + h] == conv[ok]]
            Y = Xtr[ok + h]
            Xo = Xtr[ok]
            # (Xo - mu)'(Y - mu) = Xo'Y - mu (sum Y)' - (sum Xo) mu' + n mu mu'
            B = (Xo.T @ Y).astype(np.float64) - np.outer(mu, Y.sum(0)) - np.outer(Xo.sum(0), mu) + len(ok) * np.outer(mu, mu)
            del Y, Xo
            Wt = np.linalg.solve(Lc.T, np.linalg.solve(Lc, B)).astype(np.float32)
            del B
            idx = np.arange(T - h)
            idx = idx[seqs[idx + h] == seqs[idx]]
            S = (Xte[idx] - mu) @ Wt + mu
            for l in range(L):
                Fr[l, idx, h - 1] = np.argsort(-S[:, l * E:(l + 1) * E], 1)[:, :k]
            del Wt, S
            print(key, "h", h, f"{time.time() - t0:.0f}s", flush=True)
        # baselines: persistence and decayed request counts (half-life 8), within each problem
        Fp = np.full_like(Fr, -1)
        Fd = np.full_like(Fr, -1)
        acc = np.zeros((L, E))
        dec = 0.5 ** (1 / 8)
        for t in range(T):
            if t == 0 or seqs[t] != seqs[t - 1]:
                acc[:] = 0
            acc *= dec
            for l in range(L):
                acc[l, act[t, l]] += 1
            for h in range(1, H + 1):
                if t + h < T and seqs[t + h] == seqs[t]:
                    Fp[:, t, h - 1] = act[t].copy()
                    Fd[:, t, h - 1] = np.argsort(-acc, 1)[:, :k]
        res = {"E": E, "k": k, "L": L, "T_train": int(Ttr), "T_test": int(T), "lambda": LAM, "precision_at_k": {}, "cells": {}}
        for nm, F in (("ridge", Fr), ("persist", Fp), ("decayed", Fd)):
            pr = []
            for h in range(1, H + 1):
                idx = np.arange(T - h)
                idx = idx[seqs[idx + h] == seqs[idx]]
                f = F[:, idx, h - 1]                       # [L, n, k]
                tgt = act[idx + h].transpose(1, 0, 2)      # [L, n, k]
                pr.append(float((f[..., :, None] == tgt[..., None, :]).any(-1).mean()))
            res["precision_at_k"][nm] = pr
        for C in Cs:
            aa = replay(act, seqs, E, C, -1, True, kappa=-1e9)[0]
            opt = replay(act, seqs, E, C, 0, False)[0]
            cell = {"aa": aa, "min": opt}
            for nm, F in (("ridge", Fr), ("persist", Fp), ("decayed", Fd)):
                for W in (1, 2, 4, 8):
                    v = replay_fc(act, seqs, E, C, F, W)
                    cell[f"{nm}_W{W}"] = v
                    cell[f"{nm}_W{W}_share"] = (aa - v) / (aa - opt)
            res["cells"][str(C)] = cell
            print(key, C, {kk: round(vv, 3) for kk, vv in cell.items() if kk.endswith("share")}, flush=True)
        print(key, "precision@k", {kk: [round(x, 3) for x in vv] for kk, vv in res["precision_at_k"].items()}, flush=True)
        out[key] = res
        json.dump(out, open(pj, "w"), indent=1)
    if not all(k_ in out for k_ in AIME):
        return
    for key, nm in (("gpt-oss-120b", "Gpt"), ("qwen3-30b-a3b", "Qwen")):
        r = out[key]
        M[f"lfPrecOne{nm}"] = f"{r['precision_at_k']['ridge'][0]:.2f}"
        M[f"lfPrecFour{nm}"] = f"{r['precision_at_k']['ridge'][3]:.2f}"
        M[f"lfPrecDecOne{nm}"] = f"{r['precision_at_k']['decayed'][0]:.2f}"
        M[f"lfPrecDecFour{nm}"] = f"{r['precision_at_k']['decayed'][3]:.2f}"
        sh = [r["cells"][c][f"ridge_W{W}_share"] for c in r["cells"] for W in (1, 2, 4, 8)]
        M[f"lfShareMax{nm}"] = f"{100 * max(sh):.0f}"
        M[f"lfShareMin{nm}"] = f"{100 * min(sh):.0f}"
    with open(P("paper", "wsg_forecaster.tex"), "w") as f:
        f.write("% generated by scripts/forecaster_linear.py\n")
        for kk in sorted(M):
            f.write(f"\\newcommand{{\\{kk}}}{{{M[kk]}}}\n")
    print(M)


if __name__ == "__main__":
    main()
