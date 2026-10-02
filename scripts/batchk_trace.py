"""Batch-K verification: what speculative decoding's batched verification buys an expert cache, on routing traces.

Speculative decoding verifies several draft tokens in one target forward pass, so at every MoE layer the expert sets
of K positions are requested together, and a missed expert is read from host memory once for all of them. This
converts foresight (the next tokens' routing) into batching. Before any engineering, this script asks how many host
reads per committed token that saves, on the same S-arm traces as scripts/foresight.py (nine models, the models' own
sampled text, response tokens of every conversation back to back, the cache carried across conversations).

Convention (K, acceptance, committed tokens).
  K        positions per verification forward pass: the token committed by the previous step (whose forward pass is
           due) plus K - 1 draft tokens. K = 1 is plain decoding. Papers that count K as draft tokens would call our
           K their K + 1.
  alpha    per-draft acceptance probability; drafts are accepted independently until the first rejection, so the
           number of accepted drafts is A = min(K - 1, G) with G ~ Geometric, P(G = g) = alpha^g (1 - alpha).
  commit   a step commits A + 1 tokens: the accepted drafts and one more token (the token resampled at the first
           rejection, or the bonus token after K - 1 accepted drafts). Mean committed tokens per step =
           (1 - alpha^K) / (1 - alpha), which is Leviathan et al.'s (1 - alpha^(gamma+1)) / (1 - alpha) with
           gamma = K - 1 drafts. "Accepted token" below means committed token; every trace position is committed
           exactly once, so reads per accepted token = total reads / trace length.
  routing  the trace is walked as the target model's true tokens. In a step starting at position p, the positions
           p .. p + A carry the trace's routing (the carried-over token and the accepted drafts equal the target's
           tokens). The K - 1 - A rejected drafts are wrong tokens; their routing is unknown, and three variants
           bracket it:
             random  each rejected draft routes like a token drawn from a uniformly random other position of the
                     trace (the same position in every layer, so it is one token's routing; no correlation with
                     the true token: the pessimistic end);
             same    each rejected draft routes like the true token at the position it occupies (p + A + 1 + j; full
                     correlation, the reads only come earlier: an intermediate reference);
             free    rejected drafts cost nothing (as if the verifier knew the outcome in advance: the optimistic end).
           Steps never straddle a conversation; a 'same' draft past the conversation's end is dropped.
  cache    C slots per layer. The per-layer demand of a step is the union of the K positions' expert sets. Every
           expert of the union that is not resident costs one host read (it is copied into a slot and run on the GPU,
           or run on the CPU: the same read count, the accounting of the speed limit and of foresight.py's
           'dfa-fetch' and 'opt' rows). Event-atomic: no member of the current union is evicted during the step; when
           every resident is in the union a miss is bypassed. Policies:
             belady  MIN with bypass at step granularity: a miss is admitted only if its next use (in steps) precedes
                     the evicted resident's, the victim being the resident with the farthest next use; a miss never
                     requested again is not admitted. At K = 1 this is exactly foresight.py's 'opt'.
             online  decayed frequency (half-life 16 committed tokens; the clock is the committed-token position of
                     the step; each requesting position of the batch adds 1 to the expert's score): the victim is the
                     resident with the lowest decayed score, the miss is admitted if its score beats the victim's by
                     kappa (kappa 0 and the model's kappa from scripts/provenance.py are both run, fewer reads kept).
                     At K = 1 this is exactly foresight.py's 'dfa-fetch'.
Metrics per (model, C, K, alpha, variant, policy): reads per committed token and its ratio to K = 1 (K = 1 does not
depend on alpha or the variant); bytes per committed token at BF16 expert size (data/shapes.json) and, for models
with a GGUF entry in data/gguf_bytes.json, at the served quantisation; the mean union size per step, as a fraction of
K*k and of (positions in the batch)*k; committed tokens per step.
Budgets: C = E/8, E/4, 3E/8 per layer (floor), Mixtral 2, 3, 4. Runtime: the first 40,000 response tokens of each
trace (gpt-oss-120b has 32,601); domains are interleaved in corpus order, so the prefix mixes them.

    python scripts/batchk_trace.py --results /home/claude/gpu-branch/results --out prereg/batchk_trace.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl.traces import load_pack  # noqa: E402
from scripts.provenance import MODELS, find, per_seq  # noqa: E402

K_LIST = (1, 2, 4, 8)
ALPHAS = (0.6, 0.8, 0.9)
VARIANTS = ("random", "same", "free")
TOKEN_CAP = 40000
HALF_LIFE = 16.0
SEED = 0
BIG = 1 << 40
REPO = {"olmoe": "allenai/OLMoE-1B-7B-0125-Instruct", "gpt-oss-20b": "openai/gpt-oss-20b",
        "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B-Instruct-2507", "gpt-oss-120b": "openai/gpt-oss-120b",
        "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1", "deepseek-v2-lite": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "qwen1.5-moe": "Qwen/Qwen1.5-MoE-A2.7B-Chat", "qwen2-57b": "Qwen/Qwen2-57B-A14B-Instruct",
        "phi3.5-moe": "microsoft/Phi-3.5-MoE-instruct"}
GGUF = {"gpt-oss-20b": "ggml-org/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf",
        "gpt-oss-120b": "ggml-org/gpt-oss-120b-GGUF/gpt-oss-120b-MXFP4.gguf",
        "qwen3-30b-a3b": "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"}


def budgets(key, E):
    if key == "mixtral-8x7b":
        return (2, 3, 4)
    return tuple(max(1, (E * n) // 8) for n in (1, 2, 3))


def make_steps(T, bounds, K, alpha, variant, rng):
    """Verification steps over T committed tokens: start positions, committed counts (A + 1), and the trace positions
    whose routing the rejected drafts take (flat list with offsets; empty for 'free')."""
    starts, ntrue, rej = [], [], []
    p, b = 0, 0
    while p < T:
        while bounds[b + 1] <= p:
            b += 1
        end = int(bounds[b + 1])
        A = min(K - 1, int(rng.geometric(1.0 - alpha) - 1)) if K > 1 else 0
        A = min(A, end - 1 - p)
        n_rej = K - 1 - A
        if variant == "random":
            r = rng.integers(0, T, size=n_rej)
        elif variant == "same":
            r = np.arange(p + A + 1, min(end, p + A + 1 + n_rej))
        else:
            r = np.zeros(0, np.int64)
        starts.append(p)
        ntrue.append(A + 1)
        rej.append(np.asarray(r, np.int64))
        p += A + 1
    off = np.zeros(len(starts) + 1, np.int64)
    off[1:] = np.cumsum([len(r) for r in rej])
    flat = np.concatenate(rej).astype(np.int64) if off[-1] > 0 else np.zeros(0, np.int64)
    return np.array(starts, np.int64), np.array(ntrue, np.int64), flat, off


@njit
def _unions(R, E, starts, ntrue, rej, rej_off):
    """Per step, the distinct experts requested by its positions (flat list, per-expert request counts, offsets)."""
    S = starts.shape[0]
    k = R.shape[1]
    tot = 0
    for s in range(S):
        tot += (ntrue[s] + rej_off[s + 1] - rej_off[s]) * k
    flat = np.empty(tot, np.int32)
    cnt = np.empty(tot, np.int32)
    off = np.zeros(S + 1, np.int64)
    mark = np.full(E, -1, np.int64)
    where = np.zeros(E, np.int64)
    n = 0
    for s in range(S):
        p = starts[s]
        for i in range(ntrue[s] + rej_off[s + 1] - rej_off[s]):
            u = p + i if i < ntrue[s] else rej[rej_off[s] + i - ntrue[s]]
            for j in range(k):
                e = R[u, j]
                if mark[e] != s:
                    mark[e] = s
                    flat[n] = e
                    cnt[n] = 1
                    where[e] = n
                    n += 1
                else:
                    cnt[where[e]] += 1
        off[s + 1] = n
    return flat[:n], cnt[:n], off


@njit
def _next_use(flat, off, E):
    S = off.shape[0] - 1
    nxt = np.empty(flat.shape[0], np.int64)
    last = np.full(E, BIG, np.int64)
    for s in range(S - 1, -1, -1):
        for q in range(off[s], off[s + 1]):
            nxt[q] = last[flat[q]]
        for q in range(off[s], off[s + 1]):
            last[flat[q]] = s
    return nxt


@njit
def _belady(flat, off, nxt, E, C):
    """MIN with bypass at step granularity, event-atomic. Returns host reads."""
    S = off.shape[0] - 1
    res = np.full(C, -1, np.int64)
    slot = np.full(E, -1, np.int64)
    nu = np.full(E, BIG, np.int64)
    inU = np.full(E, -1, np.int64)
    reads = 0
    for s in range(S):
        a, b = off[s], off[s + 1]
        for q in range(a, b):
            e = flat[q]
            nu[e] = nxt[q]
            inU[e] = s
        for q in range(a, b):
            e = flat[q]
            if slot[e] >= 0:
                continue
            reads += 1
            if nu[e] >= BIG:
                continue                     # never requested again: do not spend a slot
            fr = -1
            for c in range(C):
                if res[c] < 0:
                    fr = c
                    break
            if fr >= 0:
                res[fr] = e
                slot[e] = fr
                continue
            v = -1
            vkey = -1
            for c in range(C):
                r = res[c]
                if inU[r] == s:
                    continue                 # needed by this step: not evictable
                if nu[r] > vkey:
                    vkey = nu[r]
                    v = c
            if v < 0 or nu[e] >= vkey:
                continue                     # no victim, or admitting would not bridge a sooner reuse: bypass
            slot[res[v]] = -1
            res[v] = e
            slot[e] = v
    return reads


@njit
def _online(flat, cnt, off, starts, E, C, hl, kappa):
    """Decayed-frequency online policy, one read per miss, event-atomic. Returns host reads."""
    S = off.shape[0] - 1
    decay = 0.5 ** (1.0 / hl)
    res = np.full(C, -1, np.int64)
    slot = np.full(E, -1, np.int64)
    sc = np.zeros(E)
    last = np.zeros(E, np.int64)
    inU = np.full(E, -1, np.int64)
    reads = 0
    for s in range(S):
        a, b = off[s], off[s + 1]
        t = starts[s]
        for q in range(a, b):
            e = flat[q]
            sc[e] = sc[e] * decay ** (t - last[e]) + cnt[q]
            last[e] = t
            inU[e] = s
        for q in range(a, b):
            e = flat[q]
            if slot[e] >= 0:
                continue
            reads += 1
            fr = -1
            for c in range(C):
                if res[c] < 0:
                    fr = c
                    break
            if fr >= 0:
                res[fr] = e
                slot[e] = fr
                continue
            v = -1
            vkey = 0.0
            for c in range(C):
                r = res[c]
                if inU[r] == s:
                    continue
                key = sc[r] * decay ** (t - last[r])
                if v < 0 or key < vkey:
                    vkey = key
                    v = c
            if v < 0 or not (sc[e] > vkey + kappa):
                continue
            slot[res[v]] = -1
            res[v] = e
            slot[e] = v
    return reads


def simulate(routes, E, C, starts, ntrue, rej, rej_off, kappa):
    """Sum over layers of host reads (belady; online at kappa 0 and at the model's kappa) and of union sizes."""
    rb = ro0 = rok = usum = 0
    for l in range(routes.shape[0]):
        flat, cnt, off = _unions(np.ascontiguousarray(routes[l]), E, starts, ntrue, rej, rej_off)
        rb += _belady(flat, off, _next_use(flat, off, E), E, C)
        ro0 += _online(flat, cnt, off, starts, E, C, HALF_LIFE, 0.0)
        rok += _online(flat, cnt, off, starts, E, C, HALF_LIFE, kappa) if kappa != 0.0 else ro0
        usum += flat.shape[0]
    return rb, ro0, rok, usum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results"))
    ap.add_argument("--out", default="prereg/batchk_trace.json")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--cap", type=int, default=TOKEN_CAP)
    a = ap.parse_args()
    root = os.path.join(os.path.dirname(__file__), "..")
    shapes = json.load(open(os.path.join(root, "data", "shapes.json")))
    gguf = json.load(open(os.path.join(root, "data", "gguf_bytes.json")))
    out = dict(convention=__doc__, K=list(K_LIST), alphas=list(ALPHAS), variants=list(VARIANTS), token_cap=a.cap,
               seed=SEED, half_life=HALF_LIFE, models={})
    t_all = time.time()
    for key in a.models.split(","):
        ck, kappa, _ = MODELS[key]
        p = find(a.results, f"{ck}_S.npz")
        if not p:
            print(f"{key}: no S pack")
            continue
        pk = load_pack(p)
        win = per_seq(pk)
        idx = np.concatenate([np.arange(x, y) for _, x, y in win])
        T_full = len(idx)
        idx = idx[:a.cap]
        T = len(idx)
        bounds = np.concatenate([[0], np.cumsum([y - x for _, x, y in win])])
        bounds = np.concatenate([bounds[bounds < T], [T]])
        routes = np.ascontiguousarray(pk["routes"][:, idx])
        L, _, k = routes.shape
        E = int(pk["E"])
        sh = shapes[REPO[key]]
        S_bf16 = sh["expert_params"] * 2.0
        S_gguf = float(np.mean(gguf[GGUF[key]]["expert_bytes_per_layer"])) if key in GGUF else None
        rec = dict(repo=REPO[key], E=E, k=k, L=L, T=T, T_full=T_full, subsampled=T < T_full, conversations=int(len(bounds) - 1),
                   kappa=kappa, expert_bytes_bf16=S_bf16, expert_bytes_gguf=S_gguf, gguf=GGUF.get(key), budgets={})
        print(f"== {key}: E={E} k={k} L={L} T={T} of {T_full} ({rec['conversations']} conversations), kappa={kappa}")
        t_model = time.time()
        for C in budgets(key, E):
            rows = []
            base = {}
            for K in K_LIST:
                for alpha in (ALPHAS if K > 1 else (None,)):
                    for variant in (VARIANTS if K > 1 else ("none",)):
                        rng = np.random.default_rng([SEED, K, int(round(100 * (alpha or 0))), VARIANTS.index(variant) if K > 1 else 0])
                        starts, ntrue, rej, rej_off = make_steps(T, bounds, K, alpha or 0.0, variant, rng)
                        rb, ro0, rok, usum = simulate(routes, E, C, starts, ntrue, rej, rej_off, kappa)
                        S = len(starts)
                        width = int(ntrue.sum() + rej_off[-1])
                        kap = 0.0 if ro0 <= rok else kappa
                        row = dict(K=K, alpha=alpha, variant=variant, steps=S, tokens_per_step=T / S, positions_per_step=width / S,
                                   union_per_step=usum / S / L, union_over_Kk=usum / S / L / (K * k), union_over_batch=usum / (width * k * L),
                                   belady=dict(reads_per_token=rb / T), online=dict(reads_per_token=min(ro0, rok) / T, kappa=kap))
                        if K == 1:
                            base = dict(belady=rb / T, online=min(ro0, rok) / T)
                        for pol in ("belady", "online"):
                            r = row[pol]
                            r["rel_to_K1"] = r["reads_per_token"] / base[pol]
                            r["bytes_per_token_bf16"] = r["reads_per_token"] * S_bf16
                            if S_gguf:
                                r["bytes_per_token_gguf"] = r["reads_per_token"] * S_gguf
                        rows.append(row)
            rec["budgets"][str(C)] = dict(C=C, frac=C / E, K1=base, rows=rows)
            k1 = base
            print(f"   C={C:3d} ({100 * C / E:.1f}%): K=1 reads/token belady {k1['belady']:.2f} online {k1['online']:.2f}")
            for alpha in ALPHAS:
                line = f"      alpha={alpha}:"
                for variant in VARIANTS:
                    parts = []
                    for K in K_LIST[1:]:
                        r = next(x for x in rows if x["K"] == K and x["alpha"] == alpha and x["variant"] == variant)
                        parts.append(f"K{K} {r['belady']['rel_to_K1']:.2f}/{r['online']['rel_to_K1']:.2f}")
                    line += f"  {variant}: " + " ".join(parts)
                print(line)
        rec["seconds"] = time.time() - t_model
        out["models"][key] = rec
        print(f"   [{rec['seconds']:.0f} s]")
        json.dump(out, open(a.out, "w"), indent=1)
    # cross-model summary: median and quartiles over models of reads per token relative to K = 1
    summ = {}
    for bi, bname in enumerate(("E/8", "E/4", "3E/8")):
        for K in K_LIST[1:]:
            for alpha in ALPHAS:
                for variant in VARIANTS:
                    for pol in ("belady", "online"):
                        vals = []
                        for key, rec in out["models"].items():
                            Cs = sorted(rec["budgets"], key=int)
                            rows = rec["budgets"][Cs[bi]]["rows"]
                            r = next(x for x in rows if x["K"] == K and x["alpha"] == alpha and x["variant"] == variant)
                            vals.append(r[pol]["rel_to_K1"])
                        v = np.array(vals)
                        summ[f"{bname}|K{K}|alpha{alpha}|{variant}|{pol}"] = dict(
                            budget=bname, K=K, alpha=alpha, variant=variant, policy=pol, n=len(v),
                            median=float(np.median(v)), q1=float(np.percentile(v, 25)), q3=float(np.percentile(v, 75)),
                            min=float(v.min()), max=float(v.max()))
    out["summary"] = summ
    out["seconds"] = time.time() - t_all
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out} in {out['seconds']:.0f} s")
    print("median across models of reads per committed token relative to K = 1 (belady / online):")
    for bname in ("E/8", "E/4", "3E/8"):
        for alpha in ALPHAS:
            line = f"  {bname:5s} alpha={alpha}:"
            for variant in VARIANTS:
                line += f"  {variant}:"
                for K in K_LIST[1:]:
                    b = summ[f"{bname}|K{K}|alpha{alpha}|{variant}|belady"]["median"]
                    o = summ[f"{bname}|K{K}|alpha{alpha}|{variant}|online"]["median"]
                    line += f" K{K} {b:.2f}/{o:.2f}"
            print(line)


if __name__ == "__main__":
    main()
