"""Trace-driven expert-cache simulation for batch-1 MoE decode.

A trace for one MoE layer is an int array R[T, k]: the k experts selected at
decode step t. The GPU holds a cache of `cap` experts for that layer (per-layer
partitioning, as in Mixtral-offloading / HOBBIT / llama.cpp), or a single
pool across layers (global variants).

Two miss semantics, which matter for both optimal policy and cost:
  * fetch   - a miss must be copied over PCIe and then runs on the GPU, so the
              expert is always admitted (classic demand paging).
  * bypass  - a miss can be executed where it lives (CPU); admission into the
              GPU cache is optional and costs a background PCIe transfer.
Belady's MIN is optimal for `fetch`; MIN-with-bypass (admit only if the new
expert is re-used before the furthest-reused resident) is optimal for
`bypass` with unit-size items (verified against exhaustive search in tests).

Every policy returns, per step, the number of misses and admissions, so the
performance model can use the full per-(token, layer) distribution rather
than a mean hit rate (layer time is a max() of GPU and CPU paths, which is
nonlinear in the miss count).
"""
from __future__ import annotations

import numpy as np
from numba import njit

INF = np.int64(1 << 62)


@njit(cache=True)
def next_use(R, E):
    """nxt[t, j] = next step > t at which expert R[t, j] is requested (INF if never)."""
    T, k = R.shape
    last = np.full(E, INF, dtype=np.int64)
    nxt = np.empty((T, k), dtype=np.int64)
    for t in range(T - 1, -1, -1):
        for j in range(k):
            nxt[t, j] = last[R[t, j]]
        for j in range(k):
            last[R[t, j]] = t
    return nxt


@njit(cache=True)
def _sim(R, E, cap, policy, bypass, static_set):
    """policy: 0=LRU 1=LFU 2=Belady(MIN) 3=static. Returns (miss[T], adm[T])."""
    T, k = R.shape
    incache = np.zeros(E, dtype=np.bool_)
    stamp = np.zeros(E, dtype=np.int64)     # LRU recency
    freq = np.zeros(E, dtype=np.int64)      # LFU counts
    nu = np.full(E, INF, dtype=np.int64)    # next use of each expert (for MIN)
    nxt = next_use(R, E) if policy == 2 else np.zeros((1, 1), dtype=np.int64)
    miss = np.zeros(T, dtype=np.int32)
    adm = np.zeros(T, dtype=np.int32)
    size = 0
    if policy == 3:
        for e in range(E):
            if static_set[e]:
                incache[e] = True
    req = np.zeros(E, dtype=np.bool_)
    missed = np.zeros(k, dtype=np.int64)
    for t in range(T):
        for j in range(k):
            req[R[t, j]] = True
            freq[R[t, j]] += 1
            if policy == 2:
                nu[R[t, j]] = nxt[t, j]
        # pass 1: serve hits (they run before any eviction this step)
        nm = 0
        for j in range(k):
            e = R[t, j]
            if incache[e]:
                stamp[e] = t
            else:
                missed[nm] = e
                nm += 1
        miss[t] = nm
        # pass 2: admission decisions for the misses
        for q in range(nm):
            e = missed[q]
            if policy == 3 or cap == 0:
                continue
            victim = -1
            if size >= cap:
                best = -1
                for c in range(E):
                    # with bypass, experts already served this step may be evicted
                    if incache[c] and (not req[c] or (bypass and policy == 2)):
                        if policy == 0:
                            key = -stamp[c]
                        elif policy == 1:
                            key = -freq[c] * (1 << 20) - stamp[c]  # LFU, LRU tiebreak
                        else:
                            key = nu[c]
                        if victim == -1 or key > best:
                            best, victim = key, c
                if victim == -1:
                    continue  # everything resident is in use this step
                if bypass and policy == 2 and nu[e] >= nu[victim]:
                    continue  # admitting would not improve the MIN objective
                incache[victim] = False
                size -= 1
            elif bypass and policy == 2 and nu[e] >= INF:
                continue  # never reused: do not waste a transfer
            incache[e] = True
            size += 1
            stamp[e] = t
            adm[t] += 1
        for j in range(k):
            req[R[t, j]] = False
    return miss, adm


POLICIES = {"lru": 0, "lfu": 1, "min": 2, "static": 3}


def simulate(R, E, cap, policy="lru", bypass=False, static_set=None):
    R = np.ascontiguousarray(R, dtype=np.int64)
    if static_set is None:
        static_set = np.zeros(E, dtype=np.bool_)
    return _sim(R, E, int(cap), POLICIES[policy], bool(bypass), static_set.astype(np.bool_))


def top_frequency_set(R, E, cap):
    """Static pinning: the `cap` most frequently used experts in trace R."""
    cnt = np.bincount(R.ravel(), minlength=E)
    s = np.zeros(E, dtype=np.bool_)
    s[np.argsort(-cnt, kind="stable")[:cap]] = True
    return s


def brute_force_opt(R, E, cap, bypass):
    """Exact minimum-miss schedule by exhaustive search over cache states.
    Only for tiny instances (tests)."""
    from itertools import combinations
    T, k = R.shape
    states = [frozenset(c) for n in range(cap + 1) for c in combinations(range(E), n)]
    best = {s: 0 for s in states if len(s) == 0}
    for t in range(T):
        reqs = set(int(x) for x in R[t])
        new = {}
        for s, cost in best.items():
            m = len(reqs - s)
            # after serving step t, any state reachable: resident set may add
            # requested experts (admission) and drop anything; with fetch
            # semantics every requested expert must be resident after step t.
            for s2 in states:
                if not bypass and not reqs <= s2:
                    continue
                if not (s2 - s) <= reqs:   # can only admit what was requested
                    continue
                c = cost + m
                if new.get(s2, 1 << 30) > c:
                    new[s2] = c
        best = new
    return min(best.values())


@njit(cache=True)
def _sim_rstar(R, E, cap, half_life, theta, evict_lfu):
    """Online r*-aware admission (bypass semantics: misses run on the CPU).

    Each expert keeps an exponentially decayed request count (half-life in
    decode steps) - an online estimate of its near-future reuse. A missed
    expert is copied to the GPU only if its score exceeds the victim's score by
    `theta`; set theta ~ kappa * r*, the number of extra reuses needed to pay
    for the copy on the target platform. Victim: least-recently used, or
    lowest decayed score if evict_lfu. Returns (miss[T], adm[T])."""
    T, k = R.shape
    decay = 0.5 ** (1.0 / half_life)
    score = np.zeros(E)
    last = np.zeros(E, dtype=np.int64)
    stamp = np.zeros(E, dtype=np.int64)
    incache = np.zeros(E, dtype=np.bool_)
    req = np.zeros(E, dtype=np.bool_)
    miss = np.zeros(T, dtype=np.int32)
    adm = np.zeros(T, dtype=np.int32)
    missed = np.zeros(k, dtype=np.int64)
    size = 0
    for t in range(T):
        for j in range(k):
            e = R[t, j]
            score[e] = score[e] * decay ** (t - last[e]) + 1.0
            last[e] = t
            req[e] = True
        nm = 0
        for j in range(k):
            e = R[t, j]
            if incache[e]:
                stamp[e] = t
            else:
                missed[nm] = e
                nm += 1
        miss[t] = nm
        for q in range(nm):
            e = missed[q]
            if cap == 0:
                continue
            if size < cap:
                incache[e] = True; size += 1; stamp[e] = t; adm[t] += 1
                continue
            victim = -1
            best = 0.0
            for c in range(E):
                if incache[c] and not req[c]:
                    if evict_lfu:
                        key = -score[c] * decay ** (t - last[c])
                    else:
                        key = -float(stamp[c])
                    if victim == -1 or key > best:
                        best, victim = key, c
            if victim == -1:
                continue
            sv = score[victim] * decay ** (t - last[victim])
            if score[e] > sv + theta:
                incache[victim] = False
                incache[e] = True
                stamp[e] = t
                adm[t] += 1
        for j in range(k):
            req[R[t, j]] = False
    return miss, adm


def simulate_rstar(R, E, cap, half_life=16.0, theta=1.0, evict_lfu=True):
    return _sim_rstar(np.ascontiguousarray(R, dtype=np.int64), E, int(cap), float(half_life), float(theta), bool(evict_lfu))
