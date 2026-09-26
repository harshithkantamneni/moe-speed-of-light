"""Trace statistics, cache simulation, and speed-of-light analysis.

For each traced model: locality statistics; hit rates for static / LRU / LFU /
Belady policies across cache sizes; then decode tok/s on consumer platforms
for each placement strategy using the validated performance model.
Writes results/analysis_<model>.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mosl.archs import shape
from mosl.cachesim import simulate, top_frequency_set
from mosl.perfmodel import HW, Params, Workload, dynamic_time, speed_of_light_time, static_offload_time
from mosl.traces import Trace, locality_stats

MODELS = {
    "olmoe-1b-7b": ("allenai/OLMoE-1B-7B-0125-Instruct", "data/tok_olmoe.jsonl", 4.5, 8.5),
    "qwen3-30b-a3b": ("Qwen/Qwen3-30B-A3B-Instruct-2507", "data/tok_qwen3_30b.jsonl", 4.5, 8.5),
    "gpt-oss-20b": ("openai/gpt-oss-20b", "data/tok_gpt-oss-20b.jsonl", 4.25, 8.5),
    "gpt-oss-120b": ("openai/gpt-oss-120b", "data/tok_gpt-oss-120b.jsonl", 4.25, 8.5),
}
# Consumer platforms (datasheet peaks). DRAM = channels x MT/s x 8 B.
PLATFORMS = {
    "RTX 4060 8GB + DDR5-5600 (PCIe4 x8)": HW(bw_gpu=272, bw_cpu=89.6, bw_pcie=15.75),
    "RTX 4090 + DDR5-6000 (PCIe4 x16)": HW(bw_gpu=1008, bw_cpu=96.0, bw_pcie=31.5),
    "RTX 5090 + DDR5-6400 (PCIe5 x16)": HW(bw_gpu=1792, bw_cpu=102.4, bw_pcie=63.0),
}
FRACS = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
VRAM_GB = {"RTX 4060 8GB + DDR5-5600 (PCIe4 x8)": 8, "RTX 4090 + DDR5-6000 (PCIe4 x16)": 24,
           "RTX 5090 + DDR5-6400 (PCIe5 x16)": 32}
RESERVE_GB = 1.5   # CUDA context, activations, fragmentation
MATCH_CTX = 8192   # KV cache sized for an 8k-token conversation


TAUS = [None, 20.0, 50.0]  # hand-off latency sensitivity: fitted value, 20 us, 50 us


def model_params():
    """Fitted parameters (selected variant) with an *idealized* per-transfer
    latency for fetch strategies: the fitted tau_x reflects one Python
    prototype, not the DMA engine; 10 us approximates cudaMemcpyAsync."""
    v = json.load(open("results/validation.json"))
    p = v["variants"][v["selected"]]["params"]
    q = Params(eta_g=p["eta_g"], eta_c=p["eta_c"], eta_p=p["eta_p"], tau_us=p["tau_us"],
               tau_g_us=p.get("tau_g_us", 0.0), tau_c_us=p.get("tau_c_us", 0.0), tau_x_us=10.0)
    return q, v["selected"]


def per_layer(tr, cap, policy, bypass=False, static_from=None):
    miss = np.zeros((tr.T, tr.L), np.int32)
    adm = np.zeros((tr.T, tr.L), np.int32)
    for i, R in enumerate(tr.R):
        ss = None
        if policy == "static":
            ss = top_frequency_set(R if static_from is None else static_from[i], tr.E, cap)
        m, a = simulate(R, tr.E, cap, policy, bypass=bypass, static_set=ss)
        miss[:, i], adm[:, i] = m, a
    return miss, adm


def xdomain_static(tr, cap):
    """Static pinning calibrated on the *other* domains (held-out domain eval)."""
    miss = np.zeros((tr.T, tr.L), np.int32)
    for d in tr.domains():
        te = tr.dom == d
        for i, R in enumerate(tr.R):
            ss = top_frequency_set(R[~te], tr.E, cap)
            m, _ = simulate(R[te], tr.E, cap, "static", static_set=ss)
            miss[te, i] = m
    return miss, np.zeros_like(miss)


def fetch_extra_time(w, hw, p, miss):
    """Per-transfer latency for fetch strategies (added on the critical path)."""
    return float(miss.sum(1).mean()) * p.tau_x_us * 1e-6


def analyze(name):
    repo, tok, b_exp, b_dense = MODELS[name]
    tr = Trace(f"data/traces/{name}", tok)
    s = shape(repo)
    assert s.n_experts == tr.E and s.top_k == tr.k and s.n_moe_layers == tr.L, (s, tr.E, tr.k, tr.L)
    out = {"model": name, "repo": repo, "decode_tokens": tr.T, "E": tr.E, "k": tr.k, "L": tr.L,
           "teacher_forced_ppl": tr.meta.get("teacher_forced_ppl")}
    out["locality"] = locality_stats(tr)
    p0, variant = model_params()
    out["model_variant"] = variant
    out["params"] = {k: getattr(p0, k) for k in ("eta_g", "eta_c", "eta_p", "tau_us", "tau_g_us", "tau_c_us", "tau_x_us")}
    w = Workload(s, b_exp, b_dense, ctx=1024)
    hit, sims = {}, {}
    for f in FRACS:
        cap = int(round(f * tr.E))
        runs = {
            "static_layer": None,
            "static_hot_oracle": per_layer(tr, cap, "static"),
            "static_hot_xdomain": xdomain_static(tr, cap),
            "lru": per_layer(tr, cap, "lru"),
            "lfu": per_layer(tr, cap, "lfu"),
            "min_fetch": per_layer(tr, cap, "min"),
            "min_bypass": per_layer(tr, cap, "min", bypass=True),
        }
        L_gpu = int(round(f * tr.L))
        hit[f] = {"static_layer": L_gpu / tr.L}
        for k_, v in runs.items():
            if v is not None:
                hit[f][k_] = 1 - v[0].sum() / (tr.T * tr.L * tr.k)
                hit[f][k_ + "_adm_per_tok"] = float(v[1].sum(1).mean())
        sims[f] = (runs, L_gpu)
    out["hit_rate"] = {str(f): v for f, v in hit.items()}
    # VRAM-matched operating point per platform: experts that fit after dense weights + KV + reserve
    wk = Workload(s, b_exp, b_dense, ctx=MATCH_CTX)
    routed_gb = s.routed_params_total * b_exp / 8e9
    out["matched"] = {}
    for pname, vram in VRAM_GB.items():
        f = max(0.0, min(1.0, (vram - RESERVE_GB - wk.dense_bytes / 1e9) / routed_gb))
        cap = int(f * tr.E)
        L_gpu = int(f * tr.L)
        runs = {"static_hot_xdomain": xdomain_static(tr, cap), "static_hot_oracle": per_layer(tr, cap, "static"),
                "lru": per_layer(tr, cap, "lru"), "min_fetch": per_layer(tr, cap, "min"),
                "min_bypass": per_layer(tr, cap, "min", bypass=True)}
        hr = {k_: 1 - v[0].sum() / (tr.T * tr.L * tr.k) for k_, v in runs.items()}
        hr["static_layer"] = L_gpu / tr.L
        out["matched"][pname] = {"frac": f, "cap_per_layer": cap, "gpu_layers": L_gpu, "hit": hr, "_runs": (runs, L_gpu)}
    # tok/s per platform and strategy, for each hand-off latency assumption
    out["tok_s"] = {}
    for tau in TAUS:
        p = Params(**{k: getattr(p0, k) for k in Params.__dataclass_fields__ if k != "extra"})
        if tau is not None:
            p.tau_us = tau
        key = "tau=" + ("fitted" if tau is None else f"{tau:g}us")
        out["tok_s"][key] = throughput(tr, w, p, sims)
        for pname, hw in PLATFORMS.items():
            m = out["matched"][pname]
            m.setdefault("tok_s", {})[key] = strategies(tr, Workload(s, b_exp, b_dense, ctx=MATCH_CTX), hw, p, *m["_runs"])
    for m in out["matched"].values():
        m.pop("_runs")
    # reuse threshold r*: an expert is worth copying to the GPU only if reused at least r* times
    out["r_star"] = {}
    for pname, hw in PLATFORMS.items():
        g, c, pc = p0.eta_g * hw.bw_gpu, p0.eta_c * hw.bw_cpu, p0.eta_p * hw.bw_pcie
        out["r_star"][pname] = (1 / pc) / (1 / c - 1 / g) if c < g else float("inf")
    out["bytes"] = {"expert_MB": w.expert_bytes / 1e6, "dense_GB": w.dense_bytes / 1e9,
                    "routed_total_GB": s.routed_params_total * b_exp / 8e9}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open(f"results/analysis_{name}.json", "w"), indent=1, default=float)
    return out


def throughput(tr, w, p, sims):
    tps = {}
    for pname, hw in PLATFORMS.items():
        tps[pname] = {}
        for f in FRACS:
            tps[pname][str(f)] = strategies(tr, w, hw, p, *sims[f])
        # all experts resident (no offload) and all on CPU, for reference
        tps[pname]["all_gpu"] = 1 / static_offload_time(w, hw, p, 0)[0]
        tps[pname]["all_experts_cpu"] = 1 / static_offload_time(w, hw, p, tr.L)[0]
    return tps


def strategies(tr, w, hw, p, runs, L_gpu):
    r = {}
    r["static_layer_cpu"] = 1 / static_offload_time(w, hw, p, tr.L - L_gpu)[0]
    for pol in ("static_hot_xdomain", "static_hot_oracle"):
        m, a = runs[pol]
        r[pol + "_cpu"] = 1 / dynamic_time(w, hw, p, m, a, "cpu")[0]
    for pol in ("lru", "min_bypass"):
        m, a = runs[pol]
        r[pol + "_cpu"] = 1 / dynamic_time(w, hw, p, m, a, "cpu")[0]
    for pol in ("lru", "min_fetch"):
        m, a = runs[pol]
        t, _ = dynamic_time(w, hw, p, m, a, "fetch")
        r[pol + "_fetch"] = 1 / (t + fetch_extra_time(w, hw, p, m))
    M = runs["min_bypass"][0].sum() / (tr.T * tr.L)
    r["speed_of_light"] = 1 / speed_of_light_time(w, hw, p, M)
    r["all_experts_cpu"] = 1 / static_offload_time(w, hw, p, tr.L)[0]
    return r


if __name__ == "__main__":
    for n in sys.argv[1:] or [m for m in MODELS if os.path.exists(f"data/traces/{m}/meta.json")]:
        o = analyze(n)
        print(n, json.dumps(o["locality"], default=lambda v: round(v, 3)))
        for f in ("0.25", "0.5"):
            print("  hit@", f, {k: round(v, 3) for k, v in o["hit_rate"][f].items() if "adm" not in k})
        print("  r*:", {k[:9]: round(v, 2) for k, v in o["r_star"].items()})
        for pn, d in o["tok_s"]["tau=20us"].items():
            print("  ", pn, "all_gpu %.1f all_cpu %.1f" % (d["all_gpu"], d["all_experts_cpu"]))
            for f in ("0.25", "0.5"):
                print("     f=", f, {k: round(v, 1) for k, v in d[f].items()})
