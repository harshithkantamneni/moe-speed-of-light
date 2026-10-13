"""Forecast files for the CPU test: the trace itself (exact) and a random one, in shadow.npz's [T, L, H, m] layout."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forecast_file as FF  # noqa: E402
import vmap as V  # noqa: E402

O = sys.argv[1]
act, seq, meta = V.load_la(f"{O}/la.bin")
T, L, k = act.shape
E = int(meta["n_expert"])
H, m = 4, k + 2
F = np.full((T, L, H, m), 255, np.uint8)
for g in range(T):
    for h in range(H):
        t = g + 1 + h
        if t < T and seq[t] == seq[g]:
            F[g, :, h, :k] = act[t]
FF.write(f"{O}/fc_exact.bin", F, seq, k)
rng = np.random.default_rng(5)
R = np.full((T, L, H, m), 255, np.uint8)
for g in range(T):
    for l in range(L):
        for h in range(H):
            if rng.random() < 0.1:
                continue                                  # no forecast at all for this (row, layer, h)
            ex = rng.permutation(E)[:m].astype(np.uint8)
            keep = rng.random(m) < 0.5                    # half the time the true expert of step g + 1 + h
            t = g + 1 + h
            if t < T and seq[t] == seq[g]:
                tr = [e for e in act[t, l] if e not in ex[:k]]
                for j in range(k):
                    if keep[j] and tr:
                        ex[j] = tr.pop()
            ex = np.array(list(dict.fromkeys(ex.tolist())), np.uint8)   # distinct, as a router's top-m
            if rng.random() < 0.1:
                ex[rng.integers(0, len(ex))] = 255        # a hole
            R[g, l, h, : len(ex)] = ex
FF.write(f"{O}/fc_rand.bin", R, seq, k)
print("forecasts written", T, L, k, E)
