"""Exact re-implementation of the llama.cpp expert cache's policy (src/llama-expert-cache.cpp)
for trace-driven prediction: same victim rules, same decayed scores, and the same
two-step publication delay (a copy issued after step t is used from step t+2 on)."""
import json
import os

import numpy as np


def decode_windows(rows, seqs, n_prefill, n_decode):
    """(start, stop) token ranges of the decode steps ec-bench runs for each sequence."""
    out = []
    for si in seqs:
        r = rows[si]
        L = len(r["ids"])
        P = max(1, min(n_prefill, r["prompt_len"] if r["prompt_len"] > 0 else n_prefill, L - 1))
        N = min(n_decode, L - P)
        if N > 0:
            out.append((si, P, P + N))
    return out


def load_steps(trace_dir, tok_file, seqs, n_prefill=128, n_decode=192):
    """Routing [T, k] per layer for the decode steps of `seqs`, in run order."""
    rows = [json.loads(l) for l in open(tok_file)]
    lens = np.load(os.path.join(trace_dir, "seq_lens.npy"))
    assert [len(r["ids"]) for r in rows] == lens.tolist()
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    win = decode_windows(rows, seqs, n_prefill, n_decode)
    idx = np.concatenate([np.arange(starts[si] + a, starts[si] + b) for si, a, b in win])
    meta = json.load(open(os.path.join(trace_dir, "meta.json")))
    layers = sorted(int(k) for k in meta["layers"])
    R = [np.load(os.path.join(trace_dir, f"layer{l:03d}.npy")).astype(np.int64)[idx] for l in layers]
    E = int(meta["layers"][str(layers[0])]["num_experts"])
    return R, E, win


def counts_init(R_profile, E, C):
    """Top-C experts by request count per layer (static hot set)."""
    return [list(np.argsort(-np.bincount(R.ravel(), minlength=E), kind="stable")[:C]) for R in R_profile]


def simulate_layer(R, E, C, policy="dfa", half_life=16.0, kappa=0.0, init=None):
    """Returns hits[T], misses[T], admits[T] for one layer."""
    T, k = R.shape
    decay = 0.5 ** (1.0 / half_life)
    slot_of = np.full(E, -1)
    resident = np.full(C, -1)
    loading = np.full(C, -1)
    batch = np.zeros(C, dtype=np.int64)
    score = np.zeros(E)
    last = np.zeros(E, dtype=np.int64)
    stamp = np.full(E, -1, dtype=np.int64)
    if init is not None:
        for s, e in enumerate(init[:C]):
            resident[s] = e
            slot_of[e] = s
    hits = np.zeros(T, dtype=np.int32)
    misses = np.zeros(T, dtype=np.int32)
    admits = np.zeros(T, dtype=np.int32)
    for t in range(T):
        b = t & 1
        for s in range(C):  # publish copies issued two steps ago
            if loading[s] >= 0 and batch[s] == b:
                e = loading[s]
                loading[s] = -1
                resident[s] = e
                slot_of[e] = s
        sel = R[t]
        missed = []
        for e in sel:
            score[e] = score[e] * decay ** (t - last[e]) + 1.0
            last[e] = t
            if slot_of[e] >= 0:
                hits[t] += 1
                stamp[e] = t
            else:
                misses[t] += 1
                missed.append(e)
        if policy == "static":
            continue
        for e in missed:
            if (loading == e).any():
                continue
            free = np.where((resident < 0) & (loading < 0))[0]
            if len(free):
                s = free[0]
            else:
                victim, best = -1, 0.0
                for s2 in range(C):
                    c = resident[s2]
                    if c < 0 or loading[s2] >= 0 or c in sel:
                        continue
                    key = stamp[c] if policy == "lru" else score[c] * decay ** (t - last[c])
                    if victim < 0 or key < best:
                        best, victim = key, s2
                if victim < 0:
                    continue
                if policy == "dfa" and not (score[e] > best + kappa):
                    continue
                v = resident[victim]
                resident[victim] = -1
                slot_of[v] = -1
                s = victim
            loading[s] = e
            batch[s] = b
            admits[t] += 1
    return hits, misses, admits


def simulate(R_layers, E, C, policy="dfa", init=None, **kw):
    """Stacked [T, L] hits, misses, admits."""
    res = [simulate_layer(R, E, C, policy, init=None if init is None else init[i], **kw) for i, R in enumerate(R_layers)]
    return tuple(np.stack([r[j] for r in res], 1) for j in range(3))
