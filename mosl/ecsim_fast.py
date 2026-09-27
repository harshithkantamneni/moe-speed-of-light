"""Numba port of mosl.ecsim.simulate_layer (the llama.cpp expert cache's policy), for long traces.
Identical semantics; tests/test_ecsim_fast.py checks equality with the reference on random traces."""
import numpy as np
from numba import njit

_POL = {"dfa": 0, "lru": 1, "lfu": 2, "static": 3}


@njit(cache=True)
def _layer(R, E, C, pol, half_life, kappa, delay, init):
    T, k = R.shape
    decay = 0.5 ** (1.0 / half_life)
    slot_of = np.full(E, -1, np.int64)
    resident = np.full(C, -1, np.int64)
    loading = np.full(C, -1, np.int64)
    batch = np.zeros(C, np.int64)
    score = np.zeros(E)
    last = np.zeros(E, np.int64)
    stamp = np.full(E, -1, np.int64)
    for s in range(min(C, init.shape[0])):
        if init[s] >= 0:
            resident[s] = init[s]
            slot_of[init[s]] = s
    hits = np.zeros(T, np.int32)
    misses = np.zeros(T, np.int32)
    admits = np.zeros(T, np.int32)
    missed = np.empty(k, np.int64)
    for t in range(T):
        for s in range(C):
            if loading[s] >= 0 and t - batch[s] >= delay:
                e = loading[s]
                loading[s] = -1
                resident[s] = e
                slot_of[e] = s
        nm = 0
        for j in range(k):
            e = R[t, j]
            score[e] = score[e] * decay ** (t - last[e]) + 1.0
            last[e] = t
            if slot_of[e] >= 0:
                hits[t] += 1
                stamp[e] = t
            else:
                misses[t] += 1
                missed[nm] = e
                nm += 1
        if pol == 3:
            continue
        for i in range(nm):
            e = missed[i]
            busy = False
            for s2 in range(C):
                if loading[s2] == e:
                    busy = True
                    break
            if busy:
                continue
            s = -1
            for s2 in range(C):
                if resident[s2] < 0 and loading[s2] < 0:
                    s = s2
                    break
            if s < 0:
                victim = -1
                best = 0.0
                for s2 in range(C):
                    c = resident[s2]
                    if c < 0 or loading[s2] >= 0:
                        continue
                    insel = False
                    for j in range(k):
                        if R[t, j] == c:
                            insel = True
                            break
                    if insel:
                        continue
                    if pol == 1:
                        key = float(stamp[c])
                    else:
                        key = score[c] * decay ** (t - last[c])
                    if victim < 0 or key < best:
                        best = key
                        victim = s2
                if victim < 0:
                    continue
                if pol == 0 and not (score[e] > best + kappa):
                    continue
                v = resident[victim]
                resident[victim] = -1
                slot_of[v] = -1
                s = victim
            loading[s] = e
            batch[s] = t
            admits[t] += 1
    return hits, misses, admits


def simulate_layer(R, E, C, policy="dfa", half_life=16.0, kappa=0.0, init=None, delay=2):
    init_a = np.full(0, -1, np.int64) if init is None else np.asarray(init, np.int64)
    return _layer(np.ascontiguousarray(R, np.int64), int(E), int(C), _POL[policy], float(half_life), float(kappa),
                  int(delay), init_a)


def simulate(R_layers, E, C, policy="dfa", init=None, **kw):
    """Stacked [T, L] hits, misses, admits (no per-step admission budget)."""
    res = [simulate_layer(R, E, C, policy, init=None if init is None else init[i], **kw) for i, R in enumerate(R_layers)]
    return tuple(np.stack([r[j] for r in res], 1) for j in range(3))
