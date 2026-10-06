"""A nonlinear sequence forecaster, in the spirit of SeqMoE (a recurrent model over each step's full-stack expert
activation vector): a GRU reads the binary all-layer routing vector of steps .. t and predicts, with one head per
horizon, which experts each layer will request at t+1 .. t+H. Fitted on the model's mixed-domain own-text trace (arm
S; no AIME text), evaluated on the AIME-25 routing the engine runs: precision at k (top-k per layer), recall at k+3
(SeqMoE's metric), and the share of the admit-every-miss -> MIN read gap its forecasts close when they drive the
window policy (scripts/value_map.py, replay_fc). Writes prereg/forecaster_gru.json and paper/wsg_forecaster_gru.tex.

    python scripts/forecaster_gru.py [--epochs 6] [--hidden 512]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl.traces import load_pack  # noqa: E402
from scripts.provenance import MODELS, find  # noqa: E402
from scripts.value_map import replay, replay_fc  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
RES = "/home/claude/gpu-branch/results"
AIME = {"gpt-oss-120b": ("084c_gptoss_trace@vast/route_aime25_gptoss.npz", (14, 32)),
        "qwen3-30b-a3b": ("084b_vram_rerun@vast/route_aime25_qwen3.npz", (16, 32))}
H = 4


class Net(torch.nn.Module):
    def __init__(self, D, hid):
        super().__init__()
        self.inp = torch.nn.Linear(D, hid)
        self.gru = torch.nn.GRU(hid, hid, batch_first=True)
        self.out = torch.nn.Linear(hid, D * H)

    def forward(self, x, h0=None):
        z, h = self.gru(torch.relu(self.inp(x)), h0)
        return self.out(z), h


def onehot(R, E):
    """R [L, T, k] -> [T, L*E] float32"""
    L, T, k = R.shape
    X = np.zeros((T, L * E), np.float32)
    for l in range(L):
        X[np.arange(T)[:, None], l * E + R[l]] = 1.0
    return X


def segments(seq_ids):
    out, s = [], 0
    for t in range(1, len(seq_ids) + 1):
        if t == len(seq_ids) or seq_ids[t] != seq_ids[s]:
            out.append((s, t))
            s = t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--models", default="gpt-oss-120b,qwen3-30b-a3b")
    ap.add_argument("--maxtrain", type=int, default=32000)
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(2)
    out, M = {}, {}
    for key in a.models.split(","):
        rel, Cs = AIME[key]
        t0 = time.time()
        pk = load_pack(find(RES, f"{MODELS[key][0]}_S.npz"))
        E = int(pk["E"])
        resp = np.concatenate([np.arange(s + p, s + n) for s, p, n in zip(pk["starts"], pk["prompt_lens"], pk["seq_lens"])])
        conv = np.concatenate([[i] * (int(n) - int(p)) for i, (p, n) in enumerate(zip(pk["prompt_lens"], pk["seq_lens"]))])
        resp, conv = resp[: a.maxtrain], conv[: a.maxtrain]
        Rtr = pk["routes"][:, resp].astype(np.int64)
        L, T, k = Rtr.shape
        D = L * E
        X = torch.from_numpy(onehot(Rtr, E))
        net = Net(D, a.hidden)
        opt = torch.optim.Adam(net.parameters(), lr=a.lr)
        lossf = torch.nn.BCEWithLogitsLoss(reduction="none")
        # training chunks within conversations
        chunks = []
        for s, e in segments(conv):
            for c in range(s, e - H - 1, a.chunk):
                chunks.append((c, min(c + a.chunk, e - H)))
        for ep in range(a.epochs):
            perm = np.random.default_rng(ep).permutation(len(chunks))
            tot, nb = 0.0, 0
            for g in opt.param_groups:   # cosine decay over the epochs
                g["lr"] = a.lr * 0.5 * (1 + np.cos(np.pi * ep / a.epochs))
            for i in range(0, len(perm), a.batch):
                xs, ys, ms = [], [], []
                for j in perm[i:i + a.batch]:
                    c, d = chunks[j]
                    n = d - c
                    x = torch.zeros(a.chunk, D)
                    y = torch.zeros(a.chunk, D * H)
                    m = torch.zeros(a.chunk, 1)
                    x[:n] = X[c:d]
                    for h in range(1, H + 1):
                        y[:n, (h - 1) * D:h * D] = X[c + h:d + h]
                    m[:n] = 1
                    xs.append(x); ys.append(y); ms.append(m)
                x, y, m = torch.stack(xs), torch.stack(ys), torch.stack(ms)
                logit, _ = net(x)
                loss = (lossf(logit, y) * m).sum() / (m.sum() * D * H)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss); nb += 1
            print(key, "epoch", ep, f"loss {tot / nb:.5f}", f"{time.time() - t0:.0f}s", flush=True)
        # evaluate on the AIME routing, one problem at a time from a fresh state
        z = np.load(os.path.join(RES, rel))
        act = z["act"].astype(np.int64)
        seqs = z["seq"].astype(np.int64)
        Tt = act.shape[0]
        Xt = torch.from_numpy(onehot(act.transpose(1, 0, 2), E))
        F = np.full((L, Tt, H, k), -1, np.int64)
        F3 = np.full((L, Tt, k + 3), -1, np.int64)
        net.eval()
        with torch.no_grad():
            for s, e in segments(seqs):
                logit, _ = net(Xt[s:e].unsqueeze(0))
                lg = logit[0].numpy()
                for h in range(1, H + 1):
                    blk = lg[:, (h - 1) * D:h * D].reshape(e - s, L, E)
                    top = np.argsort(-blk, axis=2)
                    valid = np.arange(e - s) + h < (e - s)
                    for l in range(L):
                        F[l, s:e][valid, h - 1] = top[valid, l, :k]
                        if h == 1:
                            F3[l, s:e][valid] = top[valid, l, :k + 3]
        prec, rec3 = [], None
        for h in range(1, H + 1):
            idx = np.arange(Tt - h)
            idx = idx[seqs[idx + h] == seqs[idx]]
            f = F[:, idx, h - 1]
            tgt = act[idx + h].transpose(1, 0, 2)
            prec.append(float((f[..., :, None] == tgt[..., None, :]).any(-1).mean()))
            if h == 1:
                f3 = F3[:, idx]
                rec3 = float((tgt[..., :, None] == f3[..., None, :]).any(-1).mean())
        res = {"E": E, "k": k, "L": L, "T_train": int(T), "hidden": a.hidden, "epochs": a.epochs, "precision_at_k": prec,
               "recall_at_k_plus_3_h1": rec3, "cells": {}}
        for C in Cs:
            aa = replay(act, seqs, E, C, -1, True, kappa=-1e9)[0]
            mn = replay(act, seqs, E, C, 0, False)[0]
            cell = {"aa": aa, "min": mn}
            for W in (1, 2, 4):
                v = replay_fc(act, seqs, E, C, F, W)
                cell[f"gru_W{W}"] = v
                cell[f"gru_W{W}_share"] = (aa - v) / (aa - mn)
            res["cells"][str(C)] = cell
        out[key] = res
        print(key, "precision@k", [round(p, 3) for p in prec], "recall@k+3", round(rec3, 3),
              {c: {kk: round(vv, 3) for kk, vv in v.items() if kk.endswith("share")} for c, v in res["cells"].items()}, flush=True)
        nm = "Gpt" if key.startswith("gpt") else "Qwen"
        M[f"grPrecOne{nm}"] = f"{prec[0]:.2f}"
        M[f"grPrecFour{nm}"] = f"{prec[3]:.2f}"
        M[f"grRecThree{nm}"] = f"{rec3:.2f}"
        sh = [v[f"gru_W{W}_share"] for v in res["cells"].values() for W in (1, 2, 4)]
        M[f"grShareMin{nm}"] = f"{100 * min(sh):.0f}"
        M[f"grShareMax{nm}"] = f"{100 * max(sh):.0f}"
    json.dump(out, open(P("prereg", "forecaster_gru.json"), "w"), indent=1)
    with open(P("paper", "wsg_forecaster_gru.tex"), "w") as f:
        f.write("% generated by scripts/forecaster_gru.py\n")
        for kk in sorted(M):
            f.write(f"\\newcommand{{\\{kk}}}{{{M[kk]}}}\n")
    print(M)


if __name__ == "__main__":
    main()
