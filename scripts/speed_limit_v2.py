"""The speed limit of Table 1, tightened: the variants the reviews asked for, one change at a time from the 084 number.

The 084 limit (scripts/speed_limit.py, eq. 2 of the paper) is, per model and GPU expert budget C (slots per layer),
    T >= min over c of max((D + (Lk - c) S) / B_gpu, max(R*, c) S / B_host)
with R* the per-layer Belady-MIN-with-bypass reads per token on the routing trace of the measured text (cold start),
B_host the highest host read rate any probe method reached (one 1.5 s sample) and B_gpu the datasheet rate. This
script recomputes it and then changes one thing at a time:
  exact        the same per-layer optimum from mosl.cachesim (MIN with bypass, verified against exhaustive search),
               which may evict an expert already served this step; scripts/foresight._pol, used by 084, excludes the
               current step's residents from eviction and so reads slightly more than the optimum
  global pool  one pool of L*C slots shared by all layers (the equal-memory constraint the abstract states), Belady
               with bypass on the layer-interleaved stream (cachesim.simulate_global)
  fetch-only   MIN without bypass: every miss must be admitted (a system with no CPU execution and no zero-copy /
               staged run from host memory); the difference to `exact` is what bypass is worth
  warm         the same whole-trace optimum, reads counted from decode step 1024 on (the first four problems warm the
               cache); and free-start: the C (or LC) experts with the earliest first use resident before step 0,
               loaded by an uncounted synthetic prefix (mosl.bounds's construction)
  B_host       the probe's concurrent CPU + PCIe sums (concur.txt): limit under their maximum (084) and their median;
               statistics (max, median, mean, bootstrap 95% interval of the mean) of every sample group, and the
               second rental of the same CPU (job 087)
  GPU ceiling  datasheet B_gpu against the batch-1 rate stock llama.cpp reaches with every weight in VRAM (job 084b):
               B_gpu_eff = (D + L k S) x all-in-VRAM tok/s
  two forms    token-level max (above: every host read overlaps any GPU work, which needs foresight) against the
               per-layer sum  T >= D_head / B_gpu + sum over layers of min over c_l of
               max((d_l + (k - c_l) S) / B_gpu, max(R*_l, c_l) S / B_host), with R*_l the optimum's reads in layer l,
               which no system without cross-layer prefetch can beat; the difference is the value of foresight and
               overlap, defined without an order of attribution. Also with every layer's dense bytes in sequence
               (D / B_gpu + the sum of the expert-only maxima), and, as a reference that is not a bound, the sum over
               the optimum's own per-step miss counts.
  L3 note      the optimum's host bytes per token against the 9950X3D's 128 MB of L3, and the reuse distance (host
               bytes read in between) of its repeated reads of the same expert
Every system of Table 1 (ours, FreeToken, stock llama.cpp) is placed against every variant.

    python scripts/speed_limit_v2.py [--json prereg/speed_limit_v2.json] [--md prereg/speed_limit_v2_outcome.md]
"""
import argparse
import json
import os
import re
import sys

import numpy as np
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl import cachesim  # noqa: E402
from mosl.bounds import _prefix  # noqa: E402
from scripts.foresight import INF, _pol  # noqa: E402
from scripts.speed_limit import B_GPU, MODELS, host_rates, limit  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")
TRACES = {"gpt-oss-120b": f"{RES}/084c_gptoss_trace@vast/route_aime25_gptoss.npz",
          "qwen3-30b-a3b-bf16": f"{RES}/084b_vram_rerun@vast/route_aime25_qwen3.npz"}
BUDGETS = {"gpt-oss-120b": [14, 32, 51], "qwen3-30b-a3b-bf16": [16, 32, 56]}
# dense bytes read once per token outside the layers (the output head), the rest of D is spread over the L layers
D_HEAD = {"gpt-oss-120b": 615329280.0, "qwen3-30b-a3b-bf16": 2 * 311164928.0}
LABEL = {"gpt-oss-120b": "gpt-oss", "qwen3-30b-a3b-bf16": "Qwen3"}
BUDGET_LABEL = {("gpt-oss-120b", 14): "11%", ("gpt-oss-120b", 32): "25%", ("gpt-oss-120b", 51): "40%",
                ("qwen3-30b-a3b-bf16", 16): "12.5%", ("qwen3-30b-a3b-bf16", 32): "25%", ("qwen3-30b-a3b-bf16", 56): "43.75%"}
L3_BYTES = 128e6            # Ryzen 9 9950X3D: 96 MB (X3D CCD) + 32 MB
WARM_STEPS = 1024
CONCUR = f"{RES}/081_headline_law@vast/concur.txt"
CONCUR2 = f"{RES}/087_ablation_static@vast/concur.txt"
HELPERS = 14


# ----------------------------------------------------------------------------------------------- Belady variants
@njit(cache=True)
def _min_bypass_mask(R, E, cap):
    """cachesim._sim for MIN with bypass, returning the miss mask [T, k] (which requests were served from host memory)."""
    T, k = R.shape
    nxt = cachesim.next_use(R, E)
    incache = np.zeros(E, np.bool_)
    nu = np.full(E, cachesim.INF, np.int64)
    mask = np.zeros((T, k), np.bool_)
    size = 0
    for t in range(T):
        for j in range(k):
            nu[R[t, j]] = nxt[t, j]
            mask[t, j] = not incache[R[t, j]]      # hits are served before any eviction this step
        for j in range(k):
            e = R[t, j]
            if not mask[t, j] or cap == 0:
                continue
            if size >= cap:
                victim = -1
                best = -1
                for c in range(E):
                    if incache[c] and (victim == -1 or nu[c] > best):
                        best, victim = nu[c], c
                if victim == -1 or nu[e] >= nu[victim]:
                    continue
                incache[victim] = False
                size -= 1
            elif nu[e] >= cachesim.INF:
                continue
            incache[e] = True
            size += 1
    return mask


def per_layer(act, E, C, bypass=True):
    """MIN (with or without bypass) per layer: misses [T, L]."""
    return np.stack([cachesim.simulate(act[:, l, :], E, C, "min", bypass=bypass)[0] for l in range(act.shape[1])], 1)


def global_pool(act, E, C, bypass=True):
    """MIN with bypass over one pool of L*C slots, on the layer-interleaved stream: misses [T, L]."""
    return cachesim.simulate_global([act[:, l, :] for l in range(act.shape[1])], E, C, "min", bypass=bypass)[0]


def pol_reads(act, E, C):
    """scripts/foresight._pol at W = infinity, the optimum of job 084 (kappa is irrelevant at W = infinity)."""
    tot = 0
    for l in range(act.shape[1]):
        c, p = _pol(np.ascontiguousarray(act[:, l, :]), E, C, INF, 16.0, 0.0)
        tot += int(c) + int(p)
    return tot


def free_start(act, E, C, pooled):
    """Reads per token with the cache contents chosen before step 0: the C (per layer) or L*C (pooled) experts with
    the earliest first use, loaded by a synthetic prefix whose misses are not counted (mosl.bounds.mstar_segments)."""
    T, L, k = act.shape
    if pooled:
        stream = (act + (np.arange(L) * E)[None, :, None]).reshape(T * L, k)
        _, idx = np.unique(stream.ravel(), return_index=True)
        init = stream.ravel()[np.sort(idx)][:L * C]
        pre = _prefix(init.tolist(), k, E * L)
        miss, _ = cachesim.simulate(np.concatenate([pre, stream]), E * L, L * C, "min", bypass=True)
        return float(miss[len(pre):].sum()) / T
    tot = 0.0
    for l in range(L):
        r = act[:, l, :]
        _, idx = np.unique(r.ravel(), return_index=True)
        init = r.ravel()[np.sort(idx)][:C]
        pre = _prefix(init.tolist(), k, E)
        miss, _ = cachesim.simulate(np.concatenate([pre, r]), E, C, "min", bypass=True)
        tot += float(miss[len(pre):].sum())
    return tot / T


def reuse_distances(act, E, C, S, miss_pl):
    """Host bytes read by the per-layer optimum between two host reads of the same (layer, expert), for every repeated
    read, in token order (layer-minor). Returns (distances in bytes, number of reads)."""
    T, L, k = act.shape
    masks = np.stack([_min_bypass_mask(np.ascontiguousarray(act[:, l, :]), E, C) for l in range(L)], 1)  # [T, L, k]
    assert np.array_equal(masks.sum(2), miss_pl), "mask replica disagrees with mosl.cachesim"
    ids = (act + (np.arange(L) * E)[None, :, None]).reshape(T * L * k)
    e_all = ids[masks.reshape(T * L * k)]         # global ids of the host reads, in time order
    return _reuse(e_all, E * L) * S, int(len(e_all))


@njit(cache=True)
def _reuse(e_all, n_ids):
    """For every host read that repeats an earlier one: the number of host reads in between (the i-th read starts
    after i reads of S bytes)."""
    last = np.full(n_ids, -1, np.int64)
    d = np.empty(len(e_all), np.int64)
    n = 0
    for i in range(len(e_all)):
        e = e_all[i]
        if last[e] >= 0:
            d[n] = i - last[e]
            n += 1
        last[e] = i
    return d[:n]


# ----------------------------------------------------------------------------------------------- the two forms
def layer_sum(reads_l, k, S, d_layer, d_head, b_host, b_gpu, dense_in_sequence=False):
    """Per-layer sum form: D_head / B_gpu + sum_l min_c max((d_l + (k - c) S) / B_gpu, max(R_l, c) S / B_host), with
    reads_l the optimum's reads per token in every layer (a mean over steps: any system without cross-layer prefetch
    reads at least its misses of layer l during layer l, per-layer Belady minimises each layer's total, and max is
    convex, so the mean of the per-step maxima is at least the max of the means). With dense_in_sequence the dense
    bytes of every layer run before its router and cannot overlap that layer's host reads either:
    D / B_gpu + sum_l min_c max((k - c) S / B_gpu, max(R_l, c) S / B_host) (the reviews' layered refinement).
    Returns (seconds per token, CPU experts per token at the optimum)."""
    c = np.linspace(0, k, 4001)[None, :]
    r = np.asarray(reads_l, float)[:, None]
    L = len(r)
    if dense_in_sequence:
        t = np.maximum((k - c) * S / b_gpu, np.maximum(r, c) * S / b_host)
        head = (d_head + L * d_layer) / b_gpu
    else:
        t = np.maximum((d_layer + (k - c) * S) / b_gpu, np.maximum(r, c) * S / b_host)
        head = d_head / b_gpu
    i = t.argmin(1)
    return float(head + t[np.arange(L), i].sum()), float(c[0, i].sum())


def layer_sum_steps(miss, k, S, d_layer, d_head, b_host, b_gpu):
    """The same sum over the optimum's own per-step miss counts miss[T, L] (mean over steps of the sum over layers).
    Not a bound: another policy with the same total misses could spread them more evenly under the convex max."""
    T, L = miss.shape
    c = np.linspace(0, k, 4001)[None, :]
    m = np.arange(k + 1, dtype=float)[:, None]
    f = np.maximum((d_layer + (k - c) * S) / b_gpu, np.maximum(m, c) * S / b_host).min(1)   # seconds per layer-step
    h = np.bincount(miss.ravel().astype(np.int64), minlength=k + 1)[:k + 1]
    return float(d_head / b_gpu + (f * h).sum() / T)


# ----------------------------------------------------------------------------------------------- host rates
def parse_concur(txt):
    cpu = {}
    for t, v in re.findall(r"cpu_read_gbs t=(\d+) ([\d.]+)", txt):
        cpu.setdefault(int(t), []).append(float(v))
    pcie = {}
    for kind, rest, v in re.findall(r"pcie_(\w+)_gbs ([^\n]*?) ([\d.]+)\n", txt):
        pcie.setdefault(kind, {}).setdefault(rest, []).append(float(v))
    conc = [dict(t=int(t), pcie=m, cpu=float(c), pcie_gbs=float(p), sum=float(s)) for t, m, c, p, s in
            re.findall(r"concurrent t=(\d+) pcie=(\w+) cpu_gbs ([\d.]+) pcie_gbs ([\d.]+) sum ([\d.]+)", txt)]
    return cpu, pcie, conc


def cpu_at(cpu, helpers):
    """CPU-alone read rate at the helper thread count, interpolated between probed counts (as jobs/ec2/fetch_table.py)."""
    pts = sorted((t, float(np.mean(v))) for t, v in cpu.items())
    if helpers <= pts[0][0]:
        return pts[0][1]
    if helpers >= pts[-1][0]:
        return pts[-1][1]
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t0 <= helpers <= t1:
            return v0 + (v1 - v0) * (helpers - t0) / (t1 - t0)


def stats(v, rng, n_boot=10000):
    v = np.asarray(v, float)
    if len(v) == 0:
        return None
    boot = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(1)
    return dict(n=int(len(v)), samples=v.tolist(), max=float(v.max()), min=float(v.min()), median=float(np.median(v)),
                mean=float(v.mean()), mean_ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))])


def host_stats(path, rng):
    txt = open(path).read()
    cpu, pcie, conc = parse_concur(txt)
    zc = [x for kind, d in pcie.items() if kind == "zerocopy" for x in sum(d.values(), [])]
    ce = [x for kind, d in pcie.items() if kind == "copyengine" for x in sum(d.values(), [])]
    return dict(
        file=path, highest_any_method=host_rates(txt) / 1e9,
        concurrent_sum=stats([c["sum"] for c in conc], rng),
        concurrent_sum_zerocopy64=stats([c["sum"] for c in conc if c["pcie"] == "zerocopy64"], rng),
        concurrent_sum_copyengine=stats([c["sum"] for c in conc if c["pcie"] == "copyengine"], rng),
        cpu_alone_at_helpers=dict(helpers=HELPERS, gbs=cpu_at(cpu, HELPERS)),
        cpu_alone_all=stats(sum(cpu.values(), []), rng), cpu_alone_by_threads={str(t): v for t, v in sorted(cpu.items())},
        pcie_zerocopy=stats(zc, rng), pcie_zerocopy_16MB_64blocks=pcie.get("zerocopy", {}).get("chunk=16MB blocks=64"),
        pcie_copyengine=stats(ce, rng))


# ----------------------------------------------------------------------------------------------- Table 1 speeds
def measured():
    h = json.load(open(os.path.join(ROOT, "prereg", "headline_081.json")))
    s = json.load(open(os.path.join(ROOT, "prereg", "split_080.json")))["means"]
    r1 = json.load(open(os.path.join(ROOT, "prereg", "run1_082.json")))["llama"]
    key = {"gpt-oss-120b": "gpt-oss-120b", "Qwen3-30B-A3B": "qwen3-30b-a3b-bf16"}
    Cs = {("gpt-oss-120b", "11%"): 14, ("gpt-oss-120b", "25%"): 32, ("gpt-oss-120b", "40%"): 51, ("Qwen3-30B-A3B", "12.5%"): 16}
    m = {}
    for r in h["rows"]:
        m[(key[r["model"]], Cs[(r["model"], r["budget"])])] = dict(ours=r["ours_L2"], freetoken=r["ft_L2"], llama=r["llama"], source="headline_081")
    m[("qwen3-30b-a3b-bf16", 32)] = dict(ours=s["ours_C32_law L1"], freetoken=s["ft_hybrid_r0.25 L1"], llama=r1["q_stock_n36"]["measured"],
                                        source="split_080 (L1), run1_082 (llama)")
    m[("qwen3-30b-a3b-bf16", 56)] = dict(ours=s["ours_C56_law L2"], freetoken=s["ft_offload_r0.4375 L2"], llama=r1["q_stock_n27"]["measured"],
                                        source="split_080 (L2), run1_082 (llama)")
    return m


# ----------------------------------------------------------------------------------------------- main
def lim(R, Lk, S, D, bh, bg):
    t, c = limit(R, Lk, S, D, bh, bg)
    return dict(reads=float(R), t_ms=t * 1e3, tok_s=1 / t, cpu_at_limit=c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--md")
    ap.add_argument("--concur", default=CONCUR)
    ap.add_argument("--concur2", default=CONCUR2)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    ref = json.load(open(os.path.join(ROOT, "prereg", "speed_limit_084.json")))
    vram = json.load(open(os.path.join(ROOT, "prereg", "vram_084.json")))
    meas = measured()
    hs = dict(headline=host_stats(a.concur, rng), second_rental=host_stats(a.concur2, rng))
    b_max = hs["headline"]["highest_any_method"] * 1e9
    b_med = hs["headline"]["concurrent_sum"]["median"] * 1e9
    b_zc = hs["headline"]["concurrent_sum_zerocopy64"]["median"] * 1e9
    print(f"B_host: highest sample {b_max / 1e9:.1f} GB/s (084); concurrent sums median {b_med / 1e9:.1f}, mean "
          f"{hs['headline']['concurrent_sum']['mean']:.1f} [{hs['headline']['concurrent_sum']['mean_ci95'][0]:.1f}, "
          f"{hs['headline']['concurrent_sum']['mean_ci95'][1]:.1f}] (n={hs['headline']['concurrent_sum']['n']}); zero-copy "
          f"concurrent median {b_zc / 1e9:.1f}; second rental max {hs['second_rental']['highest_any_method']:.1f}, "
          f"concurrent median {hs['second_rental']['concurrent_sum']['median']:.1f}")
    notes = dict(
        reads_per_token="host reads per token of the optimum: pol_084 = scripts/foresight._pol (job 084); exact = mosl.cachesim MIN with bypass "
                        "per layer; global = one pool of L*C slots; *_fetch_only = MIN without bypass (every miss admitted); warm* = counted from "
                        f"step {WARM_STEPS} on; free_start* = the C (or L*C) earliest-first-use experts resident before step 0 (uncounted prefix)",
        limits="seconds per token and tok/s of min over c of max((D + (Lk - c) S) / B_gpu, max(R, c) S / B_host) unless named layer_sum*: "
               "current_084 = job 084; exact/global/fetch_only/warm/free_start = that read count, B_host max, B_gpu datasheet; median_bhost = exact "
               "reads at the median concurrent sum; median_bhost_zerocopy = at the median of the zero-copy pairs; measured_gpu = exact reads with "
               "B_gpu_effective; layer_sum = D_head/B_gpu + sum over layers of min_c max((d_l + (k - c) S)/B_gpu, max(R_l, c) S/B_host); "
               "layer_sum_dense_seq = D/B_gpu + sum over layers of min_c max((k - c) S/B_gpu, max(R_l, c) S/B_host); *_measured_gpu = with "
               "B_gpu_effective; layer_sum_per_step_not_a_bound = the layer sum over the optimum's own per-step miss counts (not a bound); "
               "all_changes = warm global-pool reads, median B_host, B_gpu_effective, token-level max",
        frac_of_limit="measured tok/s of Table 1 (ours L2, FreeToken L2, stock llama.cpp; Qwen3 25% from launch 1) over each limit",
        foresight_overlap_ms="per-layer sum minus token-level max, ms per token (the value of cross-layer prefetch and overlap)",
        l3="the optimum's host bytes per token and the reuse distance (host bytes read in between) of its repeated reads of one expert")
    out = dict(notes=notes, B_host=dict(max=b_max, median_concurrent=b_med, median_concurrent_zerocopy64=b_zc, stats=hs), B_gpu_datasheet=B_GPU,
               warm_steps=WARM_STEPS, models={})
    for key, path in TRACES.items():
        m = MODELS[key]
        z = np.load(path)
        act = z["act"].astype(np.int64)
        L, E, k = int(z["n_layer"]), int(z["n_expert"]), int(z["k"])
        T = act.shape[0]
        S, D = m["S"], m["D"]
        Lk = L * k
        d_head = D_HEAD[key]
        d_layer = (D - d_head) / L
        vram_tok_s = vram["vram"][key]
        b_gpu_eff = (D + Lk * S) * vram_tok_s
        M = out["models"][key] = dict(trace=path, T=T, L=L, E=E, k=k, S=S, D=D, D_head=d_head, d_layer=d_layer, all_in_vram_tok_s=vram_tok_s,
                                     B_gpu_effective=b_gpu_eff, B_gpu_effective_frac=b_gpu_eff / B_GPU, rows={})
        print(f"\n{key}: T {T}, L {L}, k {k}, E {E}; all in VRAM {vram_tok_s:.1f} tok/s -> effective B_gpu {b_gpu_eff / 1e9:.0f} GB/s "
              f"({b_gpu_eff / B_GPU:.1%} of datasheet)")
        for C in BUDGETS[key]:
            r084 = ref[key]["rows"][str(C)]["opt"]
            reads = {}
            reads["pol_084"] = pol_reads(act, E, C) / T
            assert abs(reads["pol_084"] - r084["reads"]) < 1e-9, (reads["pol_084"], r084["reads"])
            assert abs(lim(reads["pol_084"], Lk, S, D, b_max, B_GPU)["tok_s"] / r084["tok_s"] - 1) < 1e-9, "084 limit not reproduced"
            miss_pl = per_layer(act, E, C)
            miss_gl = global_pool(act, E, C)
            miss_pl_fetch = per_layer(act, E, C, bypass=False)
            miss_gl_fetch = global_pool(act, E, C, bypass=False)
            reads["exact"] = miss_pl.sum() / T
            reads["global"] = miss_gl.sum() / T
            reads["exact_fetch_only"] = miss_pl_fetch.sum() / T
            reads["global_fetch_only"] = miss_gl_fetch.sum() / T
            reads["warm"] = miss_pl[WARM_STEPS:].sum() / (T - WARM_STEPS)
            reads["warm_global"] = miss_gl[WARM_STEPS:].sum() / (T - WARM_STEPS)
            reads["free_start"] = free_start(act, E, C, pooled=False)
            reads["free_start_global"] = free_start(act, E, C, pooled=True)
            reads["free_start_after_warm"] = free_start(act[WARM_STEPS:], E, C, pooled=False)
            reads_l = miss_pl.sum(0) / T
            reads_l_gl = miss_gl.sum(0) / T
            # limits (token-level max unless named otherwise), one change at a time from `exact`
            lims = {}
            lims["current_084"] = dict(reads=r084["reads"], t_ms=r084["t_ms"], tok_s=r084["tok_s"], cpu_at_limit=r084["cpu_at_limit"])
            lims["exact"] = lim(reads["exact"], Lk, S, D, b_max, B_GPU)
            lims["global"] = lim(reads["global"], Lk, S, D, b_max, B_GPU)
            lims["fetch_only"] = lim(reads["exact_fetch_only"], Lk, S, D, b_max, B_GPU)
            lims["warm"] = lim(reads["warm"], Lk, S, D, b_max, B_GPU)
            lims["free_start"] = lim(reads["free_start"], Lk, S, D, b_max, B_GPU)
            lims["median_bhost"] = lim(reads["exact"], Lk, S, D, b_med, B_GPU)
            lims["median_bhost_zerocopy"] = lim(reads["exact"], Lk, S, D, b_zc, B_GPU)
            lims["measured_gpu"] = lim(reads["exact"], Lk, S, D, b_max, b_gpu_eff)
            for name, seq, bg in (("layer_sum", False, B_GPU), ("layer_sum_dense_seq", True, B_GPU),
                                  ("layer_sum_measured_gpu", False, b_gpu_eff), ("layer_sum_dense_seq_measured_gpu", True, b_gpu_eff)):
                t_s, c_s = layer_sum(reads_l, k, S, d_layer, d_head, b_max, bg, dense_in_sequence=seq)
                lims[name] = dict(reads=reads["exact"], t_ms=t_s * 1e3, tok_s=1 / t_s, cpu_at_limit=c_s)
            t_ss = layer_sum_steps(miss_pl, k, S, d_layer, d_head, b_max, B_GPU)
            lims["layer_sum_per_step_not_a_bound"] = dict(reads=reads["exact"], t_ms=t_ss * 1e3, tok_s=1 / t_ss)
            # everything at once in the token-level form: global pool, warm, median B_host, measured GPU
            lims["all_changes"] = lim(reads["warm_global"], Lk, S, D, b_med, b_gpu_eff)
            gpu_only = dict(datasheet_ms=(D + Lk * S) / B_GPU * 1e3, measured_ms=1e3 / vram_tok_s)
            # systems against every variant
            sp = meas[(key, C)]
            frac = {name: {sysn: sp[sysn] / v["tok_s"] for sysn in ("ours", "freetoken", "llama")} for name, v in lims.items()}
            chk = vram["check"][f"{key} C{C}"]["ours_frac_of_limit"]     # vram_stats.py uses Table 1's rounded speeds
            assert abs(frac["current_084"]["ours"] / chk - 1) < 2e-3, (frac["current_084"]["ours"], chk)
            # L3 note
            dist, n_reads = reuse_distances(act, E, C, S, miss_pl)
            l3 = dict(bytes_per_token=reads["exact"] * S, l3_bytes=L3_BYTES, reads=n_reads, repeated_reads=int(len(dist)),
                      frac_reads_reuse_under={f"{int(b / 1e6)}MB": float((dist < b).sum() / max(1, n_reads)) for b in (32e6, 96e6, 128e6)},
                      reuse_distance_bytes=dict(min=float(dist.min()), p5=float(np.percentile(dist, 5)), median=float(np.median(dist))))
            row = M["rows"][C] = dict(reads_per_token=reads, reads_per_layer=reads_l.tolist(), reads_per_layer_global=reads_l_gl.tolist(),
                                     limits=lims, gpu_only=gpu_only, measured_tok_s=sp, frac_of_limit=frac, l3=l3,
                                     foresight_overlap_ms=dict(
                                         layer_sum=lims["layer_sum"]["t_ms"] - lims["exact"]["t_ms"],
                                         layer_sum_dense_seq=lims["layer_sum_dense_seq"]["t_ms"] - lims["exact"]["t_ms"],
                                         layer_sum_measured_gpu=lims["layer_sum_measured_gpu"]["t_ms"] - lims["measured_gpu"]["t_ms"],
                                         layer_sum_dense_seq_measured_gpu=lims["layer_sum_dense_seq_measured_gpu"]["t_ms"] - lims["measured_gpu"]["t_ms"]))
            print(f"  C {C:3d}: reads/token 084 {reads['pol_084']:6.2f} exact {reads['exact']:6.2f} global {reads['global']:6.2f} "
                  f"({reads['global'] / reads['exact'] - 1:+.1%}) fetch-only {reads['exact_fetch_only']:6.2f} warm {reads['warm']:6.2f} "
                  f"free-start {reads['free_start']:6.2f} (global {reads['free_start_global']:6.2f}, after warm {reads['free_start_after_warm']:6.2f})")
            print("          limits tok/s: " + "  ".join(f"{n} {v['tok_s']:.1f}" for n, v in lims.items()))
            print(f"          two forms: max {lims['exact']['t_ms']:.3f} ms, per-layer sum {lims['layer_sum']['t_ms']:.3f} ms "
                  f"({row['foresight_overlap_ms']['layer_sum']:+.3f}), dense in sequence {lims['layer_sum_dense_seq']['t_ms']:.3f} ms "
                  f"({row['foresight_overlap_ms']['layer_sum_dense_seq']:+.3f}); L3: {l3['bytes_per_token'] / 1e6:.0f} MB/token, "
                  f"shortest reuse distance {l3['reuse_distance_bytes']['min'] / 1e6:.0f} MB")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    if a.md:
        write_md(out, a.md)


def _rng(vals, dec=0):
    """'a-b%' of the magnitudes over the cells (one value when they agree after rounding)."""
    v = sorted(abs(x) for x in vals)
    a, b = f"{100 * v[0]:.{dec}f}", f"{100 * v[-1]:.{dec}f}"
    return f"{a}%" if a == b else f"{a}-{b}%"


def write_md(out, path):
    hs = out["B_host"]["stats"]["headline"]
    h2 = out["B_host"]["stats"]["second_rental"]
    cs = hs["concurrent_sum"]
    cols = [("current_084", "084"), ("exact", "exact"), ("global", "global pool"), ("warm", "warm"), ("median_bhost", "median B_host"),
            ("measured_gpu", "measured GPU"), ("layer_sum", "per-layer sum"), ("layer_sum_dense_seq", "dense in seq.")]
    cells = [(LABEL[key], C, M, row) for key, M in out["models"].items() for C, row in M["rows"].items()]
    blab = {(LABEL[key], C): BUDGET_LABEL[(key, int(C))] for key, M in out["models"].items() for C in M["rows"]}
    lab = lambda m, C: f"{m} C={C}"  # noqa: E731
    L = ["# Speed limit v2: the Table 1 limit, one change at a time",
         "",
         "`scripts/speed_limit_v2.py`; every number in `prereg/speed_limit_v2.json`. Same traces (30 AIME-25 problems x 256 decode steps, "
         "cache carried across problems), model constants and formula as job 084 (`prereg/speed_limit_084.json`). The script reproduces the "
         "084 reads per token and limits exactly before it changes anything.",
         "",
         "Columns, in tok/s. **084**: per-layer Belady with bypass (`scripts/foresight._pol`), cold start, B_host the highest probe sample "
         f"({out['B_host']['max'] / 1e9:.1f} GB/s), B_gpu the datasheet 1,792 GB/s. **exact**: the same with the exact MIN-with-bypass optimum "
         "(`mosl.cachesim`, verified against exhaustive search; it may evict an expert already served this step, which `_pol` forbids). Each "
         "later column changes one thing from **exact**. **global pool**: one pool of L*C slots shared by all layers. **warm**: reads counted "
         f"from decode step {out['warm_steps']} on. **median B_host**: the median of the probe's {cs['n']} concurrent CPU + PCIe sums "
         f"({out['B_host']['median_concurrent'] / 1e9:.1f} GB/s). **measured GPU**: the GPU term at the rate stock llama.cpp reaches with every "
         "weight in VRAM (job 084b). **per-layer sum**: D_head/B_gpu + the sum over layers of min over c_l of max((d_l + (k - c_l) S)/B_gpu, "
         "max(R*_l, c_l) S/B_host), with R*_l the optimum's reads in layer l, instead of the token-level max; **dense in seq.**: the same with "
         "no layer's dense bytes overlapping its host reads either (D/B_gpu + the sum of the expert-only maxima). The last columns: ours as a "
         "percentage of the 084, global-pool and measured-GPU limits.",
         "",
         "| Cell | " + " | ".join(c[1] for c in cols) + " | ours (tok/s) | ours % of 084 | % of global | % of meas. GPU |",
         "|---|" + "---|" * (len(cols) + 4)]
    for m, C, M, row in cells:
        f = row["frac_of_limit"]
        L.append(f"| {m} {blab[(m, C)]} (C={C}) | " + " | ".join(f"{row['limits'][c]['tok_s']:.0f}" for c, _ in cols)
                 + f" | {row['measured_tok_s']['ours']:.1f} | {100 * f['current_084']['ours']:.0f}% | {100 * f['global']['ours']:.0f}% "
                 f"| {100 * f['measured_gpu']['ours']:.0f}% |")
    R = lambda name: [row["reads_per_token"][name] for _, _, _, row in cells]  # noqa: E731
    X = lambda name: [row["limits"][name]["tok_s"] for _, _, _, row in cells]  # noqa: E731
    ex = X("exact")
    top = [(m, C) for m, C, M, row in cells if row["limits"]["exact"]["cpu_at_limit"] > row["reads_per_token"]["exact"]]
    low = [i for i, (m, C, M, row) in enumerate(cells) if (m, C) not in top]
    rel = lambda a, b: [x / y - 1 for x, y in zip(a, b)]  # noqa: E731
    sel = lambda v, idx: [v[i] for i in idx]  # noqa: E731
    gpu_cells = [(m, C, row) for m, C, M, row in cells if row["limits"]["measured_gpu"]["tok_s"] < 0.99 * row["limits"]["exact"]["tok_s"]]
    L += ["", "## Reading", "",
          f"1. **The 084 optimum is not quite the optimum.** `_pol` never evicts an expert requested in the current step; the exact MIN with bypass "
          f"does (after serving it), and reads {_rng(rel(R('exact'), R('pol_084')), 1)} fewer experts per token. The limit rises "
          f"{_rng(rel(ex, X('current_084')), 1)} (most on Qwen3, where k = 8 of C = 16 slots are pinned each step). Every other row below starts from "
          "this exact optimum.",
          f"2. **Global pool.** A pool of L*C slots reads {_rng(rel(R('global'), R('exact')), 1)} fewer experts than C slots per layer "
          f"(more at larger budgets), which raises the limit by {_rng(sel(rel(X('global'), ex), low), 1)} at the four host-bound cells and not at all at "
          + ", ".join(lab(m, C) for m, C in top) + ": there the limit is the aggregate-bandwidth line (D + LkS)/(B_gpu + B_host), reached "
          "by running more resident experts on the CPU than the optimum reads (c* > R*), so fewer reads change nothing. The same holds for every "
          "read-count change at those two cells.",
          f"3. **Warm start.** Counting reads from step {out['warm_steps']} on lowers the reads by {_rng(rel(R('warm'), R('exact')), 1)} and "
          f"raises the host-bound limits by {_rng(sel(rel(X('warm'), ex), low), 1)}; the reviews' free-start construction (the C, or LC, experts with "
          f"the earliest first use resident before step 0) lowers them by {_rng(rel(R('free_start'), R('exact')), 1)} per layer and "
          f"{_rng(rel(R('free_start_global'), R('global')), 1)} pooled. Small, because the trace is 7,680 steps and the cold cache fills within "
          "the first problem.",
          f"4. **B_host.** The probe's concurrent CPU + PCIe sums are {', '.join(f'{x:.1f}' for x in cs['samples'])} GB/s (1.5 s each): max "
          f"{cs['max']:.1f}, median {cs['median']:.1f}, mean {cs['mean']:.1f}, bootstrap 95% interval of the mean [{cs['mean_ci95'][0]:.1f}, "
          f"{cs['mean_ci95'][1]:.1f}]; zero-copy pairs alone {', '.join(f'{x:.1f}' for x in hs['concurrent_sum_zerocopy64']['samples'])} (median "
          f"{hs['concurrent_sum_zerocopy64']['median']:.1f}, the law's B_cp), copy-engine pairs {', '.join(f'{x:.1f}' for x in hs['concurrent_sum_copyengine']['samples'])}. "
          f"CPU alone: {hs['cpu_alone_at_helpers']['gbs']:.1f} at {hs['cpu_alone_at_helpers']['helpers']} helpers, {hs['cpu_alone_all']['max']:.1f} at best; "
          f"zero-copy PCIe {hs['pcie_zerocopy']['max']:.1f}, copy engine {hs['pcie_copyengine']['max']:.1f}. The second rental of the same CPU (job 087) "
          f"summed to {', '.join(f'{x:.1f}' for x in h2['concurrent_sum']['samples'])} (max {h2['concurrent_sum']['max']:.1f}, median "
          f"{h2['concurrent_sum']['median']:.1f}) with CPU alone at most {h2['cpu_alone_all']['max']:.1f}. The 084 value is the largest of six samples, "
          f"{cs['max'] / cs['median'] - 1:.0%} above their median and {cs['max'] / hs['concurrent_sum_zerocopy64']['median'] - 1:.0%} above the "
          f"other two zero-copy pairs; the median lowers the host-bound limits by {_rng(sel(rel(X('median_bhost'), ex), low), 1)} "
          f"and the aggregate-bound ones by {_rng([rel(X('median_bhost'), ex)[i] for i in range(len(cells)) if i not in low], 1)} "
          "(B_host is a twentieth of the aggregate). A max over probe methods stays the right basis for a bound; what changes is which sample: the "
          "median of the repeated concurrent measurements, with the max reported as the spread.",
          "5. **GPU ceiling.** Stock llama.cpp with every weight in VRAM runs at " + "; ".join(
              f"{LABEL[k]} {M['all_in_vram_tok_s']:.1f} tok/s = {M['B_gpu_effective'] / 1e9:.0f} GB/s over the token's {(M['D'] + M['L'] * M['k'] * M['S']) / 1e9:.2f} GB "
              f"({M['B_gpu_effective_frac']:.0%} of datasheet)" for k, M in out["models"].items())
          + ". With that rate in the GPU term the limit is unchanged at the two smallest budgets (host-bound) and falls to the measured aggregate line at "
          + "; ".join(f"{m} C={C}: {row['limits']['measured_gpu']['tok_s']:.0f}" for m, C, row in gpu_cells)
          + " tok/s, where ours reaches " + ", ".join(f"{100 * row['frac_of_limit']['measured_gpu']['ours']:.0f}%" for m, C, row in gpu_cells)
          + " (FreeToken " + ", ".join(f"{100 * row['frac_of_limit']['measured_gpu']['freetoken']:.0f}%" for m, C, row in gpu_cells)
          + "; llama.cpp " + ", ".join(f"{100 * row['frac_of_limit']['measured_gpu']['llama']:.0f}%" for m, C, row in gpu_cells)
          + ") against " + ", ".join(f"{100 * row['frac_of_limit']['current_084']['ours']:.0f}%" for m, C, row in gpu_cells) + " of the 084 limit. "
          "So at 25% and above, '% of limit' on the datasheet basis mostly restates the batch-1 kernel efficiency; the paper should show both bases "
          "(the datasheet one is the physical bound, the measured one is what this software stack leaves on the table).",
          "6. **Two forms.** Token-level max (exact) against the per-layer sum, ms per token: "
          + "; ".join(f"{m} C={C}: {row['limits']['exact']['t_ms']:.2f} vs {row['limits']['layer_sum']['t_ms']:.2f} "
                      f"(+{row['foresight_overlap_ms']['layer_sum']:.2f} ms, {row['foresight_overlap_ms']['layer_sum'] / row['limits']['exact']['t_ms']:.0%}; "
                      f"dense in sequence {row['limits']['layer_sum_dense_seq']['t_ms']:.2f}, +{row['foresight_overlap_ms']['layer_sum_dense_seq']:.2f} ms)"
                      for m, C, M, row in cells)
          + ". The difference is what cross-layer prefetch and cross-token overlap are worth at the optimum's reads, with no order of attribution: "
          f"{_rng([row['foresight_overlap_ms']['layer_sum'] / row['limits']['exact']['t_ms'] for *_, row in cells])} of the token with the dense bytes "
          f"overlapped within their layer, {_rng([row['foresight_overlap_ms']['layer_sum_dense_seq'] / row['limits']['exact']['t_ms'] for *_, row in cells])} "
          "with them in sequence. Both sums are valid bounds for any system without cross-layer prefetch (it reads at least its misses of a layer "
          "during that layer, per-layer Belady minimises each layer's total, and the max is convex). The same sum over the optimum's own per-step "
          "miss counts (`layer_sum_per_step_not_a_bound` in the JSON) is larger still but is not a bound: another schedule with the same total could "
          "spread its misses more evenly.",
          "7. **Staging.** Nothing is staged in the engine: FETCH copies a missed expert into a slot (an eviction candidate) and serves it there, "
          "and background admission copies into a slot too. What the reviews call staging or zero-copy is FreeToken's gather kernel, which runs a "
          "missed expert on the GPU from host memory without a slot: in the bound that is one host read served off VRAM, exactly like a CPU-run "
          "miss, so the limit already covers it (c counts every expert not read from VRAM, whoever computes it) and no separate variant is needed. "
          f"The opposite restriction, a system that must admit every miss (no CPU execution, no zero-copy: MIN without bypass), reads "
          f"{_rng(rel(R('exact_fetch_only'), R('exact')))} more (`fetch_only` in the JSON); that is what bypass is worth.",
          "8. **L3.** The optimum reads " + ", ".join(f"{row['l3']['bytes_per_token'] / 1e6:.0f}" for *_, row in cells)
          + " MB per token at the six cells, so at the top budgets one token's host traffic is about the 9950X3D's 128 MB of L3. The DRAM assumption "
          "still holds for the optimum: it re-reads an expert it has read before only after "
          + ", ".join(f"{row['l3']['reuse_distance_bytes']['min'] / 1e9:.1f}" for *_, row in cells)
          + " GB of other host reads at the least (5th percentile " + ", ".join(f"{row['l3']['reuse_distance_bytes']['p5'] / 1e9:.1f}" for *_, row in cells)
          + " GB), because MIN with bypass admits anything reused before the farthest-reused resident; none of its repeated reads falls within 128 MB. "
          "A note in the paper suffices: the L3 could shorten an online policy's re-reads, not the optimum's.",
          "9. **All changes at once** (global pool, warm, median B_host, measured GPU; token-level max): "
          + "; ".join(f"{m} C={C}: {row['limits']['all_changes']['tok_s']:.0f} tok/s, ours {100 * row['frac_of_limit']['all_changes']['ours']:.0f}%"
                      for m, C, M, row in cells)
          + f". Ours spans {_rng([row['frac_of_limit']['current_084']['ours'] for *_, row in cells])} of the 084 limit and "
          f"{_rng([row['frac_of_limit']['all_changes']['ours'] for *_, row in cells])} of this one.",
          "",
          "**Systems, % of limit** (ours / FreeToken / llama.cpp) under 084, exact, global pool, median B_host, measured GPU, per-layer sum, all changes:",
          ]
    for m, C, M, row in cells:
        f = row["frac_of_limit"]
        s = lambda n: "/".join(f"{100 * f[n][x]:.0f}" for x in ("ours", "freetoken", "llama"))  # noqa: E731
        L.append(f"- {m} C={C}: {s('current_084')}, {s('exact')}, {s('global')}, {s('median_bhost')}, {s('measured_gpu')}, {s('layer_sum')}, {s('all_changes')}"
                 f" (measured {row['measured_tok_s']['ours']:.1f} / {row['measured_tok_s']['freetoken']:.1f} / {row['measured_tok_s']['llama']:.1f} tok/s)")
    L += ["", "**Reads per token of the optimum** (084 / exact / global pool / warm / free-start / free-start pooled / fetch-only):"]
    for m, C, M, row in cells:
        r = row["reads_per_token"]
        L.append(f"- {m} C={C}: {r['pol_084']:.2f} / {r['exact']:.2f} / {r['global']:.2f} / {r['warm']:.2f} / {r['free_start']:.2f} / "
                 f"{r['free_start_global']:.2f} / {r['exact_fetch_only']:.2f}")
    L += ["", "Runtime: about half a minute on the CPU (no subsampling)."]
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
