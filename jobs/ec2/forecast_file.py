"""Forecast files for the engine's oracle (LLAMA_EC_ORACLE_FORECAST, patch oracle6), from shadow_runahead.py's output.

shadow.npz holds, per decode step r (problems in order, 256 steps each), F_<variant> [T, L, H, m] uint8: the experts
each layer's router picked (unmasked top-m, best first) when the shadow ran h + 1 tokens past step r, i.e. the forecast,
made at the end of step r, of step r + 1 + h (255 = none). The engine reads int16 rows of 1 + H * L * k: the sequence
id, then for h = 0 .. H-1 and every layer the first k experts (-1 = none); FILE.json gives the shape.

shifted() is the same forecast as the engine sees it at lag d (LLAMA_EC_ORACLE_FC_LAG): at step g, row g - d, so step
g + 1 + h comes from row g - d at h + d (unknown past the horizon or across a sequence boundary), in the [L, T, H, k]
layout of vmap.replay_fc.

    python forecast_file.py shadow.npz OUT.bin [self|orac]
"""
import json
import sys

import numpy as np


def as_rows(F, k):
    """[T, L, H, m] uint8 (255 = none) -> [T, H, L, k] int16 (-1 = none)"""
    Fx = F[:, :, :, :k].astype(np.int16)
    Fx[F[:, :, :, :k] == 255] = -1
    return np.ascontiguousarray(Fx.transpose(0, 2, 1, 3))


def write(path, F, seq, k):
    rows = as_rows(F, k)
    T, H, L, _ = rows.shape
    out = np.empty((T, 1 + H * L * k), np.int16)
    out[:, 0] = seq.astype(np.int16)
    out[:, 1:] = rows.reshape(T, -1)
    out.tofile(path)
    json.dump(dict(n_rec=int(T), H=int(H), n_layer=int(L), k=int(k)), open(path + ".json", "w"))
    return out.shape


def read(path):
    meta = json.load(open(path + ".json"))
    T, H, L, k = meta["n_rec"], meta["H"], meta["n_layer"], meta["k"]
    raw = np.fromfile(path, np.int16).reshape(T, 1 + H * L * k)
    return raw[:, 1:].reshape(T, H, L, k), raw[:, 0].astype(np.int64)


def shifted(rows, fseq, tseq, lag):
    """rows [Tf, H, L, k] int16 -> [L, T, H, k] int64 as the engine sees it at lag d on a trace with sequence ids tseq"""
    Tf, H, L, k = rows.shape
    T = len(tseq)
    out = np.full((T, H, L, k), -1, np.int64)
    for g in range(lag, T):
        g0 = g - lag
        if g0 >= Tf or fseq[g0] != tseq[g]:
            continue
        if H > lag:
            out[g, : H - lag] = rows[g0, lag:]
    return np.ascontiguousarray(out.transpose(2, 0, 1, 3))


def main(src, dst, variant="self"):
    z = np.load(src)
    k = int(z["k"])
    shape = write(dst, z[f"F_{variant}"], z["seq"], k)
    print(f"{dst}: {shape[0]} rows of {shape[1]} int16 ({variant}, k = {k}, H = {int(z['H'])})")


if __name__ == "__main__":
    main(*sys.argv[1:])
