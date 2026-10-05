"""Write the learned admission order's model file for the engine (LLAMA_EC_LEARNED), on the rented machine.

The logistic weights were fitted offline (scripts/learned_policy6.py on main; jobs/ec2/learned_weights.json); this
script recomputes, from the model's mixed-domain routing trace (results/03x_*/<model>_S.npz, response tokens only),
the per-layer popularity, the same-layer transition matrices M[l][i][e] = P(e within the next H steps | i now) and,
for the cross-layer variant, PX[l'][i][l][e] = P(e at layer l within (t, t+H] | i at layer l' at t) counted every
4th step (cross = 2: quantised to 8 bits, q = round(255 p)), exactly as the offline script did, and writes the binary
the engine reads. Its sha256 is printed and must
equal the one in the job header (the file the offline replay used).

    python learned_export.py <S.npz> <weights.json> <key> <out.bin>
"""
import hashlib
import json
import struct
import sys

import numpy as np
from numba import njit

HL = np.array([1.0, 4.0, 16.0, 64.0, 256.0])
NF = 11


@njit(cache=True)
def trans_all(routes, E, H, cut, stride):
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
def trans(R, E, H):
    T, k = R.shape
    num = np.zeros((E, E)); den = np.zeros(E)
    seen = np.zeros(E, np.int64)
    for t in range(T - 1):
        stamp = t + 1
        for tt in range(t + 1, min(T, t + 1 + H)):
            for j in range(k):
                seen[R[tt, j]] = stamp
        for j in range(k):
            i = R[t, j]
            den[i] += 1
            for e in range(E):
                if seen[e] == stamp:
                    num[i, e] += 1
    for i in range(E):
        if den[i] > 0:
            for e in range(E):
                num[i, e] /= den[i]
    return num


def main():
    npz, wpath, key, out = sys.argv[1:5]
    W = json.load(open(wpath))[key]
    z = np.load(npz, allow_pickle=False)
    routes = z["routes"].astype(np.int64)
    E = int(z["num_experts"])
    seq_lens, prompt_lens = z["seq_lens"], z["prompt_lens"]
    starts = np.concatenate([[0], np.cumsum(seq_lens)[:-1]])
    idx = np.concatenate([np.arange(s0 + P, s0 + L_) for s0, L_, P in zip(starts, seq_lens, prompt_lens) if L_ - P >= 2])
    R3 = np.ascontiguousarray(routes[:, idx])
    L, T, k = R3.shape
    H, cross = int(W["H"]), int(W["cross"])
    PX = trans_all(R3, E, H, T, 4) if cross else None
    pops = np.stack([np.log1p(np.bincount(R3[l].ravel(), minlength=E).astype(np.float64) / T) for l in range(L)])
    Ms = np.stack([trans(np.ascontiguousarray(R3[l]), E, H) for l in range(L)])
    with open(out, "wb") as f:
        f.write(b"ECLP")
        f.write(struct.pack("<7i", 1, L, E, k, NF, H, cross))
        f.write(np.asarray(HL, np.float32).tobytes())
        f.write(np.asarray(W["coef"], np.float32).tobytes())
        f.write(np.asarray([W["intercept"]], np.float32).tobytes())
        f.write(np.asarray(pops, np.float32).tobytes())
        f.write(np.asarray(Ms, np.float32).tobytes())
        if cross == 2:
            f.write(np.ascontiguousarray(np.round(PX * 255.0).astype(np.uint8)).tobytes())
        elif cross:
            f.write(np.asarray(PX, np.float32).tobytes())
    print(key, "sha256", hashlib.sha256(open(out, "rb").read()).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
