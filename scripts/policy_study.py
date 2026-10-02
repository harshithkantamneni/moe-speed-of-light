"""Trace-driven expert-cache policy study across nine MoE models, a drift-rate characterisation, and confidence
intervals for the W50 lookahead law (Tier 2 items 1-3 of the paper plan).

Traces: own sampled text (arm S) of the 9 models in data/traces_manifest.json (response tokens of every
conversation back to back, cache carried across conversations, as in scripts/foresight.py).

1. Policy study. Per model and per budget C = E/8, E/4, 3E/8 slots per layer (rounded half up; Mixtral, E = 8,
   uses C = 2, 3, 4), each layer is replayed event-atomically: a token requests its k experts as one set, hits are
   served first, and an expert of the current token is never evicted to serve that token. Policies:
     lru, lfu            classic (mosl/cachesim.py, fetch semantics: every miss is copied and serves the token)
     df0, dfk            decayed frequency (half-life 16 tokens) with hysteresis kappa = 0 / the deployed kappa
                         (scripts/foresight.py _pol with W < 0: a miss is copied into the lowest-scored slot if its
                         score beats the victim's by kappa, else run where it lives; one read either way)
     dfa-deployed        the engine's policy as shipped (mosl/ecsim_fast.py: the CPU runs every miss, admissions land
                         two steps later as separate background copies; reads = misses + copies)
     arc, s3fifo         ARC (Megiddo & Modha 2003) and S3-FIFO (Yang et al. 2023), fetch semantics, adapted to atomic
                         replay: a resident requested by the current token is skipped as a victim
     static              the C most requested experts of the first 10% of the trace, never replaced
     static-hindsight    reference: the C most requested experts of the whole trace, never replaced
     W1 ... W16          Belady within a window of W future tokens, decayed frequency (kappa 0) beyond it (_pol's
                         rule, with the optimum's freedom to evict a resident already served this step)
     opt                 the exact Belady MIN with bypass over the whole future (mosl/cachesim.py, verified against
                         exhaustive search; it may evict an expert already served this step, which _pol forbids and
                         which made _pol's W = infinity read 0.1-1.4% more than the optimum): the fewest reads possible
   Every policy is scored on the last 90% of the trace (the online ones warm up on the first 10%, which is also
   static's training set; opt sees the whole future). Metric: expert reads per token from host memory (a bypassed
   miss is a read), bytes per token with the model's expert size from data/gguf_bytes.json where a GGUF header was
   read (olmoe and qwen1.5-moe: none; a Q4_K_M-equivalent estimate from the parameter count is given separately),
   hit rate, and the union-to-capacity ratio (distinct experts per layer in 16-token windows / C).
2. Drift rate. Per model, budget and layer: the fraction of the top-C experts (by request count) that changes
   between consecutive windows of 100 and of 1,000 tokens, averaged over layers; and the share of all requests on the
   top-C experts of the whole trace.
3. W50 intervals. The exponent of W50 = a (C/k)^b (scripts/fig_foresight.py, 26 points) refitted under a bootstrap
   over models (2,000 resamples), leave-one-model-out, and with the interpolated points (W50 < 1) excluded.

    python scripts/policy_study.py --results /home/claude/gpu-branch/results --out prereg/policy_study.json
    python scripts/policy_study.py --selftest
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl import cachesim, ecsim_fast  # noqa: E402
from mosl.traces import load_pack  # noqa: E402
from scripts.fig_foresight import W as FS_W, curves, w_at  # noqa: E402
from scripts.foresight import INF, _pol  # noqa: E402
from scripts.provenance import MODELS, find, per_seq  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
WINDOWS = (1, 2, 4, 8, 16)
HALF_LIFE = 16.0
DRIFT_WINDOWS = (100, 1000)
# model key -> data/gguf_bytes.json entry (the file whose header gave the expert bytes), or None
GGUF_FILE = {
    "gpt-oss-20b": "ggml-org/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf",
    "gpt-oss-120b": "ggml-org/gpt-oss-120b-GGUF/gpt-oss-120b-MXFP4.gguf",
    "qwen3-30b-a3b": "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
    "mixtral-8x7b": "mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF/Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf",
    "phi3.5-moe": "bartowski/Phi-3.5-MoE-instruct-GGUF/Phi-3.5-MoE-instruct-Q4_K_M.gguf",
    "qwen2-57b": "Qwen/Qwen2-57B-A14B-Instruct-GGUF/qwen2-57b-a14b-instruct-q4_k_m.gguf",
    "deepseek-v2-lite": "mradermacher/DeepSeek-V2-Lite-Chat-GGUF/DeepSeek-V2-Lite-Chat.Q8_0.gguf",
    "olmoe": None,
    "qwen1.5-moe": None,
}
SHAPE_REPO = {"olmoe": "allenai/OLMoE-1B-7B-0125-Instruct", "qwen1.5-moe": "Qwen/Qwen1.5-MoE-A2.7B-Chat"}


def budgets_for(key, E):
    if key == "mixtral-8x7b":
        return [2, 3, 4]
    return [int(math.floor(E * f + 0.5)) for f in (1 / 8, 1 / 4, 3 / 8)]


# ----------------------------------------------------------------------------------------------------------------
# decayed frequency / windowed Belady / Belady with bypass: scripts.foresight._pol with per-step outputs
# ----------------------------------------------------------------------------------------------------------------
@njit(cache=True)
def _pol_steps(R, E, C, W, hl, kappa, atomic):
    """scripts.foresight._pol, returning per-step (cpu reads, copies) instead of totals. Same semantics:
    W < 0 no foresight (decayed frequency with hysteresis kappa), W >= INF the whole future (Belady with bypass).
    atomic=True is _pol's rule (a resident requested by the current token is never the victim); atomic=False lets a
    resident already served this step be evicted, the freedom of the exact MIN with bypass in mosl/cachesim.py."""
    T, k = R.shape
    decay = 0.5 ** (1.0 / hl)
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
    sc = np.zeros(E)
    last = np.zeros(E, np.int64)
    cpu = np.zeros(T, np.int32)
    cp = np.zeros(T, np.int32)
    ishit = np.zeros(k, np.bool_)
    for t in range(T):
        for j in range(k):
            e = R[t, j]
            nu[e] = nu_t[t, j]
            sc[e] = sc[e] * decay ** (t - last[e]) + 1.0
            last[e] = t
            ishit[j] = slot[e] >= 0        # hits are served before any eviction this step
        lim = t + W if W >= 0 else t
        for j in range(k):
            e = R[t, j]
            if ishit[j]:
                continue
            fr = -1
            for s in range(C):
                if res[s] < 0:
                    fr = s
                    break
            if fr >= 0:
                res[fr] = e
                slot[e] = fr
                cp[t] += 1
                continue
            v = -1
            vbeyond = False
            vkey = 0.0
            for s in range(C):
                c = res[s]
                if atomic:
                    inrow = False
                    for jj in range(k):
                        if R[t, jj] == c:
                            inrow = True
                            break
                    if inrow:
                        continue
                beyond = W < 0 or nu[c] > lim
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
                cpu[t] += 1
                continue
            if W < 0 or nu[e] > lim:
                if not (vbeyond and sc[e] > vkey + kappa):
                    cpu[t] += 1
                    continue
            elif (not vbeyond) and float(nu[e]) >= vkey:
                cpu[t] += 1
                continue
            old = res[v]
            slot[old] = -1
            res[v] = e
            slot[e] = v
            cp[t] += 1
    return cpu, cp


# ----------------------------------------------------------------------------------------------------------------
# ARC (Megiddo & Modha, FAST 2003), unit-size items, fetch semantics, event-atomic
# ----------------------------------------------------------------------------------------------------------------
@njit(cache=True)
def _oldest(lst, seq, cur, which, t, E):
    """index of the least recently placed expert in list `which` that is not requested by the current token"""
    best = -1
    bs = 0
    for e in range(E):
        if lst[e] == which and cur[e] != t:
            if best < 0 or seq[e] < bs:
                best = e
                bs = seq[e]
    return best


@njit(cache=True)
def _arc_replace(lst, seq, cur, n, clk, t, E, was_b2, p):
    """ARC's REPLACE: move the LRU of T1 (if |T1| > p, or |T1| == p and x came from B2) to B1, else the LRU of T2 to
    B2; skips experts requested by the current token, falling back to the other list. Returns True if one was
    evicted."""
    first = 1
    if not (n[1] >= 1 and ((was_b2 and n[1] == p) or n[1] > p)):
        first = 2
    for which in (first, 3 - first):
        v = _oldest(lst, seq, cur, which, t, E)
        if v >= 0:
            n[which] -= 1
            lst[v] = which + 2      # T1 -> B1, T2 -> B2
            n[which + 2] += 1
            seq[v] = clk[0]
            clk[0] += 1
            return True
    return False


@njit(cache=True)
def _arc(R, E, C):
    T, k = R.shape
    lst = np.zeros(E, np.int64)          # 0 none, 1 T1, 2 T2, 3 B1 (ghost), 4 B2 (ghost)
    seq = np.zeros(E, np.int64)
    cur = np.full(E, -1, np.int64)
    n = np.zeros(5, np.int64)
    clk = np.zeros(1, np.int64)
    p = 0.0
    miss = np.zeros(T, np.int32)
    for t in range(T):
        for j in range(k):
            cur[R[t, j]] = t
        for j in range(k):                       # hits first: move to the MRU end of T2
            e = R[t, j]
            if lst[e] == 1 or lst[e] == 2:
                n[lst[e]] -= 1
                lst[e] = 2
                n[2] += 1
                seq[e] = clk[0]
                clk[0] += 1
        for j in range(k):
            e = R[t, j]
            if lst[e] == 1 or lst[e] == 2:
                continue
            miss[t] += 1
            was = lst[e]
            if was == 3:                          # ghost hit in B1: favour recency
                d = 1.0 if n[3] == 0 else max(n[4] / n[3], 1.0)
                p = min(float(C), p + d)
                if not _arc_replace(lst, seq, cur, n, clk, t, E, False, p):
                    continue
                n[3] -= 1
                lst[e] = 2
                n[2] += 1
            elif was == 4:                        # ghost hit in B2: favour frequency
                d = 1.0 if n[4] == 0 else max(n[3] / n[4], 1.0)
                p = max(0.0, p - d)
                if not _arc_replace(lst, seq, cur, n, clk, t, E, True, p):
                    continue
                n[4] -= 1
                lst[e] = 2
                n[2] += 1
            else:                                 # never seen recently
                if n[1] + n[3] == C:
                    if n[1] < C:
                        g = _oldest(lst, seq, cur, 3, -1, E)
                        lst[g] = 0
                        n[3] -= 1
                        if not _arc_replace(lst, seq, cur, n, clk, t, E, False, p):
                            continue
                    else:
                        v = _oldest(lst, seq, cur, 1, t, E)
                        if v < 0:
                            continue
                        lst[v] = 0
                        n[1] -= 1
                elif n[1] + n[3] < C and n[1] + n[2] + n[3] + n[4] >= C:
                    if n[1] + n[2] + n[3] + n[4] == 2 * C:
                        g = _oldest(lst, seq, cur, 4, -1, E)
                        lst[g] = 0
                        n[4] -= 1
                    if not _arc_replace(lst, seq, cur, n, clk, t, E, False, p):
                        continue
                lst[e] = 1
                n[1] += 1
            seq[e] = clk[0]
            clk[0] += 1
            assert n[1] + n[2] <= C
    return miss


# ----------------------------------------------------------------------------------------------------------------
# S3-FIFO (Yang et al., SOSP 2023): small FIFO (10%), main FIFO (90%), ghost FIFO (as many as main); freq 0..3,
# promotion from small at freq > 1, lazy demotion in main; fetch semantics, event-atomic
# ----------------------------------------------------------------------------------------------------------------
@njit(cache=True)
def _s3_evict_main(q, seq, freq, cur, n, clk, t, E):
    while n[2] > 0:
        v = _oldest(q, seq, cur, 2, t, E)
        if v < 0:
            return False
        if freq[v] > 0:
            freq[v] -= 1
            seq[v] = clk[0]
            clk[0] += 1
        else:
            q[v] = 0
            n[2] -= 1
            return True
    return False


@njit(cache=True)
def _s3_evict_small(q, seq, freq, cur, n, clk, t, E, capM, capG):
    while n[1] > 0:
        v = _oldest(q, seq, cur, 1, t, E)
        if v < 0:
            return False
        if freq[v] > 1:
            q[v] = 2
            n[1] -= 1
            n[2] += 1
            freq[v] = 0
            seq[v] = clk[0]
            clk[0] += 1
            if n[2] > capM:
                _s3_evict_main(q, seq, freq, cur, n, clk, t, E)
        else:
            if n[3] >= capG:
                g = _oldest(q, seq, cur, 3, -1, E)
                q[g] = 0
                n[3] -= 1
            q[v] = 3
            n[1] -= 1
            n[3] += 1
            seq[v] = clk[0]
            clk[0] += 1
            return True
    return False


@njit(cache=True)
def _s3fifo(R, E, C):
    T, k = R.shape
    capS = max(1, int(math.floor(0.1 * C + 0.5)))
    capM = C - capS
    capG = max(1, capM)
    q = np.zeros(E, np.int64)            # 0 none, 1 small, 2 main, 3 ghost
    seq = np.zeros(E, np.int64)
    freq = np.zeros(E, np.int64)
    cur = np.full(E, -1, np.int64)
    n = np.zeros(4, np.int64)
    clk = np.zeros(1, np.int64)
    miss = np.zeros(T, np.int32)
    for t in range(T):
        for j in range(k):
            cur[R[t, j]] = t
        for j in range(k):
            e = R[t, j]
            if q[e] == 1 or q[e] == 2:
                freq[e] = min(freq[e] + 1, 3)
        for j in range(k):
            e = R[t, j]
            if q[e] == 1 or q[e] == 2:
                continue
            miss[t] += 1
            ok = True
            while n[1] + n[2] >= C:
                # the reference implementation's rule (libCacheSim): evict from main only when it is over its share
                # or small is empty, otherwise from small
                if n[2] > capM or n[1] == 0:
                    ok = _s3_evict_main(q, seq, freq, cur, n, clk, t, E)
                    if not ok:
                        ok = _s3_evict_small(q, seq, freq, cur, n, clk, t, E, capM, capG)
                else:
                    ok = _s3_evict_small(q, seq, freq, cur, n, clk, t, E, capM, capG)
                    if not ok:
                        ok = _s3_evict_main(q, seq, freq, cur, n, clk, t, E)
                if not ok:
                    break
            if not ok:
                continue
            if q[e] == 3:
                n[3] -= 1
                q[e] = 2
                n[2] += 1
            else:
                q[e] = 1
                n[1] += 1
            freq[e] = 0
            seq[e] = clk[0]
            clk[0] += 1
            assert n[1] + n[2] <= C
    return miss


# ----------------------------------------------------------------------------------------------------------------
def static_misses(R, E, C, t0):
    """top-C experts of R[:t0] by request count; per-step misses over the whole trace (scored from t0 on)"""
    cnt = np.bincount(R[:t0].ravel(), minlength=E)
    top = np.zeros(E, np.bool_)
    top[np.argsort(-cnt, kind="stable")[:C]] = True
    return (~top[R]).sum(1).astype(np.int32)


def union_size(routes, segs, w=16):
    """mean over layers and non-overlapping w-token windows within conversations of the number of distinct experts
    requested (divide by C for the union-to-capacity ratio)"""
    L = routes.shape[0]
    tot, nwin = 0.0, 0
    for a, b in segs:
        for w0 in range(a, b - w + 1, w):
            blk = routes[:, w0:w0 + w].reshape(L, -1)
            tot += np.mean([len(np.unique(blk[l])) for l in range(L)])
            nwin += 1
    return tot / max(1, nwin)


def drift(routes, E, C, w):
    """fraction of the top-C set (by request count) that changes between consecutive windows of w tokens,
    averaged over consecutive pairs and layers"""
    L, T, k = routes.shape
    nw = T // w
    if nw < 2:
        return float("nan")
    out = []
    for l in range(L):
        r = routes[l, :nw * w].reshape(nw, w * k)
        cnt = np.zeros((nw, E), np.int64)
        for i in range(nw):
            cnt[i] = np.bincount(r[i], minlength=E)
        top = np.argsort(-cnt, axis=1, kind="stable")[:, :C]
        mem = np.zeros((nw, E), np.bool_)
        np.put_along_axis(mem, top, True, axis=1)
        keep = (mem[1:] & mem[:-1]).sum(1)
        out.append(1.0 - keep.mean() / C)
    return float(np.mean(out))


def topc_share(routes, E, C):
    L = routes.shape[0]
    sh = []
    for l in range(L):
        cnt = np.sort(np.bincount(routes[l].ravel(), minlength=E))[::-1]
        sh.append(cnt[:C].sum() / cnt.sum())
    return float(np.mean(sh))


def expert_bytes(key, L):
    """per-layer bytes of one routed expert from the GGUF header cache, or None; plus a Q4_K_M-equivalent estimate"""
    gg = json.load(open(os.path.join(ROOT, "data", "gguf_bytes.json")))
    shapes = json.load(open(os.path.join(ROOT, "data", "shapes.json")))
    q4 = [gg[v]["expert_bytes_per_layer"] for v in GGUF_FILE.values() if v and "Q4_K_M" in v.upper()]
    q4p = [shapes[r]["expert_params"] for r in ("Qwen/Qwen3-30B-A3B-Instruct-2507", "mistralai/Mixtral-8x7B-Instruct-v0.1",
                                                 "microsoft/Phi-3.5-MoE-instruct", "Qwen/Qwen2-57B-A14B-Instruct")]
    ratio = float(np.mean([np.mean(b) / p for b, p in zip(q4, q4p)]))   # bytes per parameter of a Q4_K_M expert
    f = GGUF_FILE[key]
    if f is None:
        est = shapes[SHAPE_REPO[key]]["expert_params"] * ratio
        return None, None, est, ratio
    eb = gg[f]["expert_bytes_per_layer"]
    assert len(eb) == L, (key, len(eb), L)
    return np.array(eb, float), f, None, ratio


def run_model(key, routes, segs, E, k, kappa, budgets, eb, log=print):
    L, T, _ = routes.shape
    t0 = T // 10
    Te = T - t0
    out = {}
    usize = union_size(routes, segs)
    for C in budgets:
        tt = time.time()
        reads_l = {}          # policy -> per-layer reads over the scored tokens
        miss_l = {}           # policy -> per-layer misses (dfa-deployed differs from reads)

        def add(name, m_step, extra=None):
            reads_l.setdefault(name, []).append(int(m_step[t0:].sum()))
            miss_l.setdefault(name, []).append(int(m_step[t0:].sum()) if extra is None else int(extra[t0:].sum()))

        for l in range(L):
            R = np.ascontiguousarray(routes[l], dtype=np.int64)
            m, _ = cachesim.simulate(R, E, C, "lru", bypass=False)
            add("lru", m)
            m, _ = cachesim.simulate(R, E, C, "lfu", bypass=False)
            add("lfu", m)
            for name, W, kap, atomic in [("df0", -1, 0.0, True), ("dfk", -1, kappa, True), ("opt-pol", INF, 0.0, True)] + \
                    [(f"W{w}", w, 0.0, False) for w in WINDOWS] + [(f"W{w}@kdep", w, kappa, False) for w in WINDOWS]:
                c, cp = _pol_steps(R, E, C, W, HALF_LIFE, kap, atomic)
                add(name, c + cp)
            m, _ = cachesim.simulate(R, E, C, "min", bypass=True)      # the exact optimum: reads = misses
            add("opt", m)
            h, m, ad = ecsim_fast.simulate_layer(R, E, C, "dfa", half_life=HALF_LIFE, kappa=kappa)
            add("dfa-deployed", m + ad, extra=m)
            add("arc", _arc(R, E, C))
            add("s3fifo", _s3fifo(R, E, C))
            add("static", static_misses(R, E, C, t0))
            add("static-hindsight", static_misses(R, E, C, T))
        cell = {}
        opt_reads = sum(reads_l["opt"]) / Te
        for name in reads_l:
            rl = np.array(reads_l[name], float)
            reads = rl.sum() / Te
            cell[name] = dict(reads=reads, hit=1.0 - sum(miss_l[name]) / (Te * L * k), rel_opt=reads / opt_reads,
                              bytes=None if eb is None else float((rl * eb).sum() / Te))
        out[C] = dict(C=C, C_over_k=C / k, union_to_capacity=usize / C, union_size=usize,
                      drift={f"w{w}": drift(routes, E, C, w) for w in DRIFT_WINDOWS},
                      topC_share=topc_share(routes, E, C), random_topC_change=1.0 - C / E, policies=cell)
        log(f"   C={C:3d} (C/k={C / k:.1f}, union/C={out[C]['union_to_capacity']:.2f}, drift100={out[C]['drift']['w100']:.2f}, "
            f"drift1000={out[C]['drift']['w1000']:.2f}, topC share={out[C]['topC_share']:.2f}) reads/token: "
            + " ".join(f"{n}={cell[n]['reads']:.1f}" for n in ("lru", "lfu", "df0", "dfk", "arc", "s3fifo", "static", "W1", "W4", "W16", "opt-pol", "opt"))
            + f"  [{time.time() - tt:.0f}s]")
    return out, T, Te


# ----------------------------------------------------------------------------------------------------------------
def fit(x, y):
    b, c0 = np.polyfit(np.log(x), np.log(y), 1)
    r = np.corrcoef(np.log(x), np.log(y))[0, 1]
    return float(b), float(np.exp(c0)), float(r)


def w50_intervals(path, nboot=2000, seed=0):
    rows = [r for r in curves(path) if "job 063" not in r["model"]]
    for r in rows:
        r["w50"] = w_at(r["g"], 0.5)
        r["interp"] = bool(r["g"][0] >= 0.5)       # half the gap closes before W = 1: W50 interpolated on [0, 1]
    rows = [r for r in rows if np.isfinite(r["w50"])]
    models = sorted({r["model"] for r in rows})
    x = np.array([r["ck"] for r in rows])
    y = np.array([r["w50"] for r in rows])
    interp = np.array([r["interp"] for r in rows])
    by_model = {m: np.array([i for i, r in enumerate(rows) if r["model"] == m]) for m in models}
    res = dict(n_points=len(rows), n_models=len(models), n_interpolated=int(interp.sum()),
               windows=list(FS_W), source=os.path.relpath(path, ROOT),
               points=[dict(model=r["model"], C=r["C"], C_over_k=r["ck"], w50=r["w50"], interpolated=r["interp"]) for r in rows])
    b, a, r = fit(x, y)
    res["full"] = dict(exponent=b, prefactor=a, r=r, n=len(rows))
    rng = np.random.default_rng(seed)

    def boot(idx_pool, label):
        bs, ps = [], []
        ms = [m for m in models if len(np.intersect1d(by_model[m], idx_pool))]
        while len(bs) < nboot:
            pick = rng.choice(len(ms), len(ms), replace=True)
            idx = np.concatenate([np.intersect1d(by_model[ms[i]], idx_pool) for i in pick])
            if len(np.unique(x[idx])) < 3:
                continue
            bb, aa, _ = fit(x[idx], y[idx])
            bs.append(bb)
            ps.append(aa)
        bs, ps = np.array(bs), np.array(ps)
        return dict(method=label, resamples=nboot, seed=seed, exponent_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    exponent_median=float(np.median(bs)), exponent_sd=float(bs.std(ddof=1)),
                    prefactor_ci95=[float(np.percentile(ps, 2.5)), float(np.percentile(ps, 97.5))])

    allidx = np.arange(len(rows))
    res["bootstrap_models"] = boot(allidx, "bootstrap over models (resample the 9 models with replacement, refit on their points)")
    # (d) reference: bootstrap over points
    bs = []
    while len(bs) < nboot:
        idx = rng.choice(len(rows), len(rows), replace=True)
        if len(np.unique(x[idx])) < 3:
            continue
        bs.append(fit(x[idx], y[idx])[0])
    bs = np.array(bs)
    res["bootstrap_points"] = dict(method="bootstrap over the 26 points (reference; ignores the model grouping)", resamples=nboot,
                                   exponent_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], exponent_median=float(np.median(bs)))
    loo = {}
    for m in models:
        idx = np.array([i for i in allidx if i not in set(by_model[m].tolist())], dtype=int)
        if len(np.unique(x[idx])) < 3:
            continue
        bb, aa, rr = fit(x[idx], y[idx])
        loo[m] = dict(exponent=bb, prefactor=aa, r=rr, n=int(len(idx)))
    ex = [v["exponent"] for v in loo.values()]
    res["leave_one_model_out"] = dict(fits=loo, exponent_range=[float(min(ex)), float(max(ex))],
                                      jackknife_se=float(np.sqrt((len(ex) - 1) / len(ex) * ((np.array(ex) - np.mean(ex)) ** 2).sum())))
    keep = np.where(~interp)[0]
    bb, aa, rr = fit(x[keep], y[keep])
    res["no_interpolated"] = dict(exponent=bb, prefactor=aa, r=rr, n=int(len(keep)), excluded=int(interp.sum()),
                                  C_over_k_range=[float(x[keep].min()), float(x[keep].max())],
                                  bootstrap_models=boot(keep, "bootstrap over models, interpolated points excluded"))
    return res


ONLINE = ("lru", "lfu", "df0", "dfk", "arc", "s3fifo", "static")


def summarize(res):
    """medians of reads relative to opt per budget class, the best online policy per cell, and the sign of LRU
    against decayed frequency (kappa 0) per model and budget"""
    models = list(res["models"])
    pols = ONLINE + ("dfa-deployed", "static-hindsight", "opt-pol") + tuple(f"W{w}" for w in WINDOWS)
    cls = {}
    for i, name in enumerate(("E/8", "E/4", "3E/8")):
        vals = {p: [] for p in pols}
        for m in models:
            v = res["models"][m]
            pc = v["cells"][str(v["budgets"][i])]["policies"]
            for p in pols:
                vals[p].append(pc[p]["rel_opt"])
        cls[name] = {p: dict(median=float(np.median(vals[p])), min=float(min(vals[p])), max=float(max(vals[p]))) for p in pols}
    allv = {p: [res["models"][m]["cells"][str(C)]["policies"][p]["rel_opt"] for m in models for C in res["models"][m]["budgets"]] for p in pols}
    cls["all"] = {p: dict(median=float(np.median(allv[p])), min=float(min(allv[p])), max=float(max(allv[p]))) for p in pols}
    best, within1 = {}, {}
    sign = {}
    for m in models:
        v = res["models"][m]
        sign[m] = []
        for C in v["budgets"]:
            pc = v["cells"][str(C)]["policies"]
            b = min(ONLINE, key=lambda p: pc[p]["reads"])
            best[b] = best.get(b, 0) + 1
            for p in ONLINE:
                if pc[p]["reads"] <= 1.01 * pc[b]["reads"]:
                    within1[p] = within1.get(p, 0) + 1
            d = pc["lru"]["reads"] / pc["df0"]["reads"] - 1
            sign[m].append(dict(C=C, lru_over_df0_pct=100 * d, sign="+" if d > 0.005 else ("-" if d < -0.005 else "0")))
    res["summary"] = dict(
        rel_opt_by_budget_class=cls,
        best_online_policy_cells=best, online_within_1pct_of_best_cells=within1,
        lru_vs_df0=dict(note="+: LRU reads more than decayed frequency (kappa 0) by > 0.5%; -: fewer; 0: within 0.5%", per_model=sign),
        drift_at_E4={m: res["models"][m]["cells"][str(res["models"][m]["budgets"][1])]["drift"] for m in models},
        below_opt=dict(note="cells where any policy reads fewer than opt (the exact MIN with bypass); expected empty",
                       cells=[dict(model=m, C=C, policy=p, rel=res["models"][m]["cells"][str(C)]["policies"][p]["rel_opt"])
                              for m in models for C in res["models"][m]["budgets"]
                              for p in res["models"][m]["cells"][str(C)]["policies"]
                              if res["models"][m]["cells"][str(C)]["policies"][p]["rel_opt"] < 1.0 - 1e-9]),
        opt_pol_over_opt={m: {str(C): res["models"][m]["cells"][str(C)]["policies"]["opt-pol"]["rel_opt"] for C in res["models"][m]["budgets"]}
                          for m in models})
    return res


def selftest():
    """_pol_steps totals equal scripts.foresight._pol; ARC and S3-FIFO respect capacity and atomicity on random
    traces (asserts inside), miss counts are sane against LRU and Belady."""
    rng = np.random.default_rng(0)
    for trial in range(40):
        E = int(rng.choice([8, 16, 32, 64]))
        k = int(rng.choice([2, 4, 8]))
        C = int(rng.integers(k, max(k + 1, E // 2)))
        T = int(rng.integers(50, 400))
        p = rng.dirichlet(np.full(E, 0.3))
        R = np.stack([rng.choice(E, k, replace=False, p=p) for _ in range(T)]).astype(np.int64)
        R[1::3] = R[0:-1:3][: len(R[1::3])]
        for W in (-1, 1, 4, INF):
            for kap in (0.0, 1.5):
                c, cp = _pol_steps(R, E, C, W, HALF_LIFE, kap, True)
                assert (int(c.sum()), int(cp.sum())) == _pol(R, E, C, W, HALF_LIFE, kap), (trial, W, kap)
        lru, _ = cachesim.simulate(R, E, C, "lru", bypass=False)
        opt, _ = cachesim.simulate(R, E, C, "min", bypass=True)
        # the exact optimum never reads more than any windowed policy, atomic or relaxed, and the relaxed W = infinity
        # policy reads no fewer than it
        for W in (1, 4, 16, INF):
            for atomic in (True, False):
                c, cp = _pol_steps(R, E, C, W, HALF_LIFE, 0.0, atomic)
                assert c.sum() + cp.sum() >= opt.sum(), (trial, W, atomic, c.sum() + cp.sum(), opt.sum())
        arc, s3 = _arc(R, E, C), _s3fifo(R, E, C)
        assert opt.sum() <= min(arc.sum(), s3.sum()) <= T * k, (trial, opt.sum(), arc.sum(), s3.sum())
        assert arc.sum() <= 1.6 * lru.sum() + 20 and s3.sum() <= 1.6 * lru.sum() + 20, (trial, lru.sum(), arc.sum(), s3.sum())
        # a full-cache LRU/ARC/S3 miss must be one of the k requested experts: at most k misses per step
        assert arc.max() <= k and s3.max() <= k
    # ARC == LRU-like sanity on a cyclic scan larger than the cache: everything misses
    E, C, T = 16, 4, 200
    R = (np.arange(T) % 8).reshape(T, 1).astype(np.int64)
    assert _arc(R, E, C).sum() == T
    # a repeated working set that fits: only compulsory misses after warm-up
    R = (np.arange(T) % 3).reshape(T, 1).astype(np.int64)
    assert _arc(R, E, C).sum() == 3 and _s3fifo(R, E, C).sum() == 3
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results"))
    ap.add_argument("--arm", default="S")
    ap.add_argument("--out", default=os.path.join(ROOT, "prereg", "policy_study.json"))
    ap.add_argument("--w50-out", default=os.path.join(ROOT, "prereg", "foresight", "w50_intervals.json"))
    ap.add_argument("--foresight", default=os.path.join(ROOT, "prereg", "foresight", "foresight_S.json"))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--max-tokens", type=int, default=0, help="score only the first N response tokens (0: all)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--summarize-only", action="store_true", help="recompute the summary block of an existing --out")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.summarize_only:
        res = summarize(json.load(open(a.out)))
        json.dump(res, open(a.out, "w"), indent=1)
        print(json.dumps(res["summary"]["rel_opt_by_budget_class"]["all"], indent=None))
        return
    t_all = time.time()
    res = dict(meta=dict(
        description=__doc__.split("\n\n")[0],
        arm=a.arm, half_life=HALF_LIFE, windows=list(WINDOWS), drift_windows=list(DRIFT_WINDOWS), union_window=16,
        scoring="reads per token over the last 90% of each trace (all policies; the first 10% warms the online policies "
                "and is static's training set; opt sees the whole future)",
        budgets="C = round(E/8), round(E/4), round(3E/8) slots per layer, half up; Mixtral (E = 8): C = 2, 3, 4",
        replay="event-atomic: a token's k experts per layer are one request set; hits are served first; an expert of "
               "the current token is never evicted to serve that token",
        policies=dict(lru="classic LRU, every miss copied (mosl/cachesim.py)", lfu="classic LFU (whole-history counts, LRU tie-break), every miss copied",
                      df0="decayed frequency, half-life 16, kappa 0: a miss is copied into the lowest-scored slot if its score exceeds the victim's, else run where it lives (scripts/foresight.py _pol, W < 0)",
                      dfk="the same with the deployed hysteresis kappa (per model, scripts/provenance.py)",
                      **{"dfa-deployed": "the engine's policy as shipped: CPU runs every miss, background admission lands 2 steps later as a separate copy (mosl/ecsim_fast.py); reads = misses + copies"},
                      arc="ARC (Megiddo & Modha 2003), every miss copied; a resident of the current token is never the victim",
                      s3fifo="S3-FIFO (Yang et al. 2023): small FIFO max(1, round(0.1 C)), main C - small, ghost as many as main, freq capped at 3, promotion at freq > 1, lazy demotion in main, eviction from main only when it is over its share or small is empty (the reference implementation's rule); every miss copied; current-token residents skipped as victims",
                      static="the C most requested experts of the first 10% of the trace, never replaced",
                      **{"static-hindsight": "reference: the C most requested experts of the whole trace (chosen in hindsight), never replaced"},
                      W="Belady within the next W tokens, decayed frequency (kappa 0) beyond (scripts/foresight.py _pol's rule, with the exact optimum's freedom to evict a resident already served this step); W{w}@kdep: the same with the deployed kappa",
                      opt="the exact Belady MIN with bypass over the whole future (mosl/cachesim.py, verified against exhaustive search): a resident already served this step may be evicted",
                      **{"opt-pol": "reference: scripts/foresight.py _pol at W = infinity (job 084's optimum), which never evicts a resident of the current token and so reads slightly more than opt"}),
        metrics=dict(reads="expert reads per token from host memory (a bypassed miss counts as a read)",
                     hit="1 - misses / (tokens * L * k)", bytes="reads x bytes of one expert (per layer where the GGUF mixes types)",
                     rel_opt="reads / opt reads", union_to_capacity="mean over layers and 16-token windows (within conversations) of distinct experts / C",
                     drift="fraction of the per-layer top-C set (by count) that changes between consecutive windows of 100 / 1000 tokens (whole trace)",
                     topC_share="share of a layer's requests on its C most requested experts over the whole trace, mean over layers",
                     random_topC_change="1 - C/E: the drift of two unrelated top-C sets"),
        expert_bytes_source="data/gguf_bytes.json (GGUF headers); olmoe and qwen1.5-moe have none: bytes are not reported for them "
                            "(expert_bytes_est_q4km is the parameter count x the Q4_K_M bytes/param of the four Q4_K_M headers)",
        max_tokens=a.max_tokens), models={})
    for key in a.models.split(","):
        ck, kappa, _ = MODELS[key]
        p = find(a.results, f"{ck}_{a.arm}.npz")
        if not p:
            print(f"{key}: no {a.arm} pack")
            continue
        pk = load_pack(p)
        win = per_seq(pk)
        idx = np.concatenate([np.arange(x, y) for _, x, y in win])
        segs, s0 = [], 0
        for _, x, y in win:
            segs.append((s0, s0 + (y - x)))
            s0 += y - x
        routes = pk["routes"][:, idx]
        if a.max_tokens and routes.shape[1] > a.max_tokens:
            routes = routes[:, :a.max_tokens]
            segs = [(x, min(y, a.max_tokens)) for x, y in segs if x < a.max_tokens]
        E, k, L = int(pk["E"]), int(routes.shape[2]), int(routes.shape[0])
        eb, ebfile, est, ratio = expert_bytes(key, L)
        budgets = budgets_for(key, E)
        print(f"== {key}: E={E} k={k} L={L} T={routes.shape[1]} conversations={len(segs)} kappa={kappa} budgets={budgets} "
              f"expert bytes={'%.3g' % eb.mean() if eb is not None else 'n/a'}", flush=True)
        cells, T, Te = run_model(key, routes, segs, E, k, kappa, budgets, eb, log=lambda s: print(s, flush=True))
        res["models"][key] = dict(E=E, k=k, L=L, T=int(T), T_scored=int(Te), convs=len(segs), kappa=kappa, budgets=budgets,
                                  expert_bytes=None if eb is None else float(eb.mean()), expert_bytes_file=ebfile,
                                  expert_bytes_est_q4km=est, q4km_bytes_per_param=ratio, trace=os.path.relpath(p, a.results),
                                  cells={str(C): v for C, v in cells.items()})
        json.dump(res, open(a.out, "w"), indent=1)
    res["meta"]["runtime_s"] = time.time() - t_all
    res = summarize(res)
    json.dump(res, open(a.out, "w"), indent=1)
    w50 = w50_intervals(a.foresight)
    os.makedirs(os.path.dirname(a.w50_out), exist_ok=True)
    json.dump(w50, open(a.w50_out, "w"), indent=1)
    print(f"W50 exponent: full {w50['full']['exponent']:.3f} (r={w50['full']['r']:.3f}, n={w50['full']['n']}); "
          f"bootstrap over models 95% {w50['bootstrap_models']['exponent_ci95']}; LOO range {w50['leave_one_model_out']['exponent_range']}; "
          f"no interpolated {w50['no_interpolated']['exponent']:.3f} (n={w50['no_interpolated']['n']}, r={w50['no_interpolated']['r']:.3f}, "
          f"95% {w50['no_interpolated']['bootstrap_models']['exponent_ci95']})")
    print(f"done in {time.time() - t_all:.0f}s -> {a.out}, {a.w50_out}")


if __name__ == "__main__":
    main()
