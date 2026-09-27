"""Phase 4, section 4.6: bound-normalized audit of published MoE-offloading results (prereg/PROTOCOL.md).

For every adjudicable row of data/audit/normalized.jsonl:
  * the equal-VRAM strong baseline: llama.cpp-style static expert offload (--n-cpu-moe n, n = MoE layers whose
    experts do not fit the row's GPU expert budget), predicted by the validated model (variant M4) fitted with
    the row's own engine family left out, with a band = the host-DRAM imputation band x the model's
    leave-one-source-out 10th-90th percentile error on static-offload rows;
  * the speed-of-light: the time bound of mosl.perfmodel.speed_of_light_time at physical peaks (every efficiency
    1, no latencies, the top of the DRAM band), with MIN-bypass misses at the row's per-layer capacity from our
    on-policy (S) traces where we have the model, else from independent uniform routing (flagged);
  * fractions of that bound reached by the system, its reported baseline and the predicted baseline;
    S_n = system / predicted baseline, and the 4.6 labels.

    python scripts/audit.py --results /home/claude/gpu-branch/results --out prereg/audit
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl import cachesim  # noqa: E402
from mosl.archs import shape  # noqa: E402
from mosl.perfmodel import HW, Params, Workload, speed_of_light_time, static_offload_time  # noqa: E402
from mosl.traces import load_pack  # noqa: E402
from mosl.validation_set import ROWS  # noqa: E402
from scripts.validate import VARIANTS, fit, group_of  # noqa: E402

NAMES = VARIANTS["M4 M1 + per-CPU-expert latency"]
PHYS = Params(eta_g=1.0, eta_c=1.0, eta_p=1.0, tau_us=0.0, tau_x_us=0.0, tau_e_us=0.0)
TRACED = {  # repo -> S-arm pack (our on-policy traces)
    "Qwen/Qwen3-30B-A3B": "qwen3-30b-a3b_fp8_S.npz", "Qwen/Qwen3-30B-A3B-Instruct-2507": "qwen3-30b-a3b_fp8_S.npz",
    "openai/gpt-oss-120b": "gpt-oss-120b_S.npz", "openai/gpt-oss-20b": "gpt-oss-20b_S.npz",
}
UNMODELABLE = {"deepseek-ai/deepseek-moe-16b-base": "architecture needs remote code (not in transformers)",
               "Qwen/Qwen3.8-Flash-Next": "per-layer n-gram lookup table is not a routed expert or dense weight in the model"}
GPU_OVERHEAD_GB = 1.5      # CUDA context + compute buffers, subtracted when only the card size is known


def loso_band():
    """10th-90th percentile of measured/predicted on static-offload validation rows, leave-one-source-out"""
    rows = list(csv.DictReader(open("results/validation.csv")))
    r = [1 / (1 + float(x["rel_err_loso"])) for x in rows if x["kind"] == "static"]
    return float(np.percentile(r, 10)), float(np.percentile(r, 90))


_fits = {}
# Sensitivity (not pre-registered): first-party llama.cpp static-offload runs on the A10 (job 025, commit 2145525a,
# response windows), in the validation-set convention (datasheet peaks: A10 600 GB/s; Xeon 8358, 8 x DDR4-3200).
A10_ROWS = [dict(id=f"A10-{r}-{n}", repo=r, kind="static", n_cpu=n, b_exp=b, b_dense=bd, bw_gpu=600.0, bw_cpu=204.8,
                 engine="llama.cpp", tok_s=t, ctx=500)
            for r, b, bd, n, t in [("Qwen/Qwen3-30B-A3B", 4.85, 4.85, 42, 68.2), ("Qwen/Qwen3-30B-A3B", 4.85, 4.85, 36, 72.0),
                                   ("Qwen/Qwen3-30B-A3B", 4.85, 4.85, 24, 85.4), ("Qwen/Qwen3-30B-A3B", 8.5, 8.5, 42, 48.3),
                                   ("Qwen/Qwen3-30B-A3B", 8.5, 8.5, 36, 51.7), ("Qwen/Qwen3-30B-A3B", 8.5, 8.5, 24, 61.6),
                                   ("openai/gpt-oss-20b", 4.25, 8.5, 21, 63.8), ("openai/gpt-oss-20b", 4.25, 8.5, 18, 69.4),
                                   ("openai/gpt-oss-20b", 4.25, 8.5, 12, 80.9), ("openai/gpt-oss-120b", 4.25, 8.5, 32, 41.3),
                                   ("openai/gpt-oss-120b", 4.25, 8.5, 27, 44.8)]]
WITH_A10 = False


def params_without(system):
    """M4 refit without validation rows of the same engine family (leave the row's source out)"""
    fam = system.lower()
    # hold out validation rows produced by the same system (KTransformers, Mixtral-offloading, ...); llama.cpp
    # forks and PRs are separate sources from the llama.cpp rows of the validation set, so nothing is held out
    held = set() if "llama.cpp" in fam or "fork" in fam else {group_of(r) for r in ROWS if r["engine"].lower() in fam}
    key = tuple(sorted(held))
    if key not in _fits:
        _fits[key] = fit([r for r in ROWS if group_of(r) not in held] + (A10_ROWS if WITH_A10 else []), NAMES)
    return _fits[key], sorted(held)


def mstar(repo, E, k, C, results, rng):
    """MIN-bypass misses per layer-step at per-layer capacity C, and where the routing came from"""
    if C <= 0:
        return float(k), "no GPU expert capacity"
    if C >= E:
        return 0.0, "all experts fit"
    p = TRACED.get(repo)
    hits = sorted(glob.glob(os.path.join(results, "*", p))) if p else []
    if hits:
        pk = load_pack(hits[-1])
        idx = np.concatenate([np.arange(s + P, s + L) for s, L, P in zip(pk["starts"], pk["seq_lens"], pk["prompt_lens"]) if L - P > 1])
        R = pk["routes"][:, idx]
        m = np.mean([cachesim.simulate(R[l], E, C, "min", bypass=True)[0].mean() for l in range(R.shape[0])])
        return float(m), f"on-policy trace ({p})"
    T = 20000
    R = np.stack([rng.choice(E, k, replace=False) for _ in range(T)])
    return float(cachesim.simulate(R, E, C, "min", bypass=True)[0].mean()), "independent uniform routing (approximation)"


def budget_bytes(row, s, w, card_gb):
    """GPU bytes for routed experts, and a note"""
    b = row.get("budget") or {}
    kind, v = b.get("kind"), b.get("value")
    per_layer_all = s.n_experts * w.expert_bytes
    if kind == "bytes":
        return float(v) * 1e9, "stated expert bytes"
    if kind == "experts_per_layer":
        return float(v) * s.n_moe_layers * w.expert_bytes, "stated experts per layer"
    if kind == "fraction":
        return float(v) * s.n_moe_layers * per_layer_all, "stated fraction of experts"
    if kind == "layers_on_gpu":
        return float(v) * per_layer_all, "stated MoE layers on GPU"
    # card size only: everything but dense weights, KV cache and runtime overhead (an upper bound)
    gb = float(v) if kind == "gpu_mem_total" and v else card_gb
    return max(0.0, gb * 1e9 - w.dense_bytes - GPU_OVERHEAD_GB * 1e9), "card size minus dense, KV, 1.5 GB (upper bound)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="data/audit/normalized.jsonl")
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-a10", action="store_true", help="sensitivity: add first-party A10 static runs to the fit")
    a = ap.parse_args()
    global WITH_A10
    WITH_A10 = a.with_a10
    q10, q90 = loso_band()
    rng = np.random.default_rng(0)
    out, skipped = [], []
    for row in map(json.loads, open(a.rows)):
        repo = row.get("repo")
        if not repo or repo in UNMODELABLE:
            skipped.append(dict(id=row["id"], reason=UNMODELABLE.get(repo, "no public architecture")))
            continue
        s = shape(repo)
        w = Workload(s, row["b_exp"], row["b_dense"], ctx=int(row.get("ctx") or 512))
        E, k, L = s.n_experts, w.k, s.n_moe_layers
        bw_lo, bw_hi = row["bw_cpu"]
        bw_p = row.get("bw_pcie") or (row.get("bw_pcie_band") or [31.5, 31.5])[-1]
        B, bnote = budget_bytes(row, s, w, row.get("gpu_mem_gb") or 0)
        n_gpu = int(min(L, B // (E * w.expert_bytes)))
        C = int(min(E, B // (L * w.expert_bytes)))
        p, held = params_without(row["system"])
        imputed = bool((row.get("budget") or {}).get("imputed")) or not row.get("budget")
        # an imputed budget is only an upper bound: the band's lower edge then assumes no expert on the GPU
        t_lo_bw, _ = static_offload_time(w, HW(row["bw_gpu"], bw_lo, bw_p), p, L if imputed else L - n_gpu)
        t_hi_bw, _ = static_offload_time(w, HW(row["bw_gpu"], bw_hi, bw_p), p, L - n_gpu)
        base_mid = 2 / (t_lo_bw + t_hi_bw)
        base_lo, base_hi = (1 / t_lo_bw) * q10, (1 / t_hi_bw) * q90
        m, msrc = mstar(repo, E, k, C, a.results, rng)
        t_sol = speed_of_light_time(w, HW(row["bw_gpu"], bw_hi, bw_p), PHYS, m)
        sys_tok = row["system_tok_s"]
        lc = [b for b in row.get("baselines", []) if b.get("class", "").startswith("llama.cpp") and b.get("tok_s")]
        rep_base = lc[0] if lc else next((b for b in row.get("baselines", []) if b.get("tok_s")), None)
        claimed = sys_tok / rep_base["tok_s"] if rep_base else None
        sn = (sys_tok / base_hi, sys_tok / base_mid, sys_tok / base_lo)
        wide = base_hi / base_mid > 1.4 or base_lo / base_mid < 0.6
        labels = []
        if wide or (claimed is not None and claimed < 1.2):
            labels.append("not adjudicated: " + ("band wider than +-40 %" if wide else "claimed speed-up < 1.2x"))
        else:
            if rep_base and lc:
                v = rep_base["tok_s"]
                labels.append("weak baseline" if v < base_lo else ("at strength" if v <= base_hi else "baseline above prediction"))
            labels.append("gain survives" if sn[0] > 1 else "not established")
        out.append(dict(id=row["id"], system=row["system"], model=row.get("model"), repo=repo, gpu=row["gpu"],
                        bw_cpu=[bw_lo, bw_hi], budget_gb=B / 1e9, budget_note=bnote, budget_imputed=imputed, n_cpu_layers=L - n_gpu, L=L,
                        C_per_layer=C, E=E, k=k, system_tok_s=sys_tok,
                        reported_baseline=rep_base, claimed_speedup=claimed,
                        pred_baseline_tok_s=[base_lo, base_mid, base_hi], normalized_speedup=list(sn),
                        sol_tok_s=1 / t_sol, mstar=m, mstar_source=msrc,
                        system_of_sol=sys_tok * t_sol, reported_baseline_of_sol=(rep_base["tok_s"] * t_sol) if rep_base else None,
                        pred_baseline_of_sol=base_mid * t_sol, labels=labels, held_out_groups=held,
                        routing_exact=row.get("routing_exact"), lossy=row.get("lossy"), speculative=row.get("speculative"),
                        tier=row.get("tier")))
    adj = [r for r in out if not any(l.startswith("not adjudicated") for l in r["labels"])]
    with_lc = [r for r in adj if r["reported_baseline"] and r["reported_baseline"].get("class", "").startswith("llama.cpp")]
    summ = dict(n_rows=len(out), n_adjudicated=len(adj), n_skipped=len(skipped), band_q10_q90=[q10, q90],
                gain_survives=sum("gain survives" in r["labels"] for r in adj),
                llama_rows=len(with_lc), weak_baselines=sum("weak baseline" in r["labels"] for r in with_lc),
                median_system_of_sol=float(np.median([r["system_of_sol"] for r in adj])) if adj else None,
                P5=(sum("weak baseline" in r["labels"] for r in with_lc) >= len(with_lc) / 3) if with_lc else None,
                P6=(float(np.median([r["system_of_sol"] for r in adj])) <= 0.60) if adj else None)
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(summary=summ, rows=out, skipped=skipped), open(os.path.join(a.out, "audit.json"), "w"), indent=1, default=float)
    f = lambda v, fmt="{:.1f}": "–" if v is None else fmt.format(v)
    L_ = ["# Phase 4.6: bound-normalized audit\n",
          f"Model band (static rows, leave-one-source-out, measured/predicted 10th-90th pct): {q10:.2f}-{q90:.2f}.\n",
          "| system | model | GPU | budget GB (n CPU layers / L) | system tok/s | reported baseline | claimed | predicted equal-VRAM baseline [band] | S_n [band] | SoL tok/s (M* source) | system / SoL | labels |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out:
        rb = r["reported_baseline"]
        L_.append(f"| {r['system']} | {r['model']} | {r['gpu']} | {r['budget_gb']:.1f} ({r['n_cpu_layers']}/{r['L']}) | {r['system_tok_s']:.1f} | "
                  f"{(rb['name'] + ' ' + f(rb['tok_s'])) if rb else '–'} | {f(r['claimed_speedup'], '{:.2f}x')} | "
                  f"{r['pred_baseline_tok_s'][1]:.1f} [{r['pred_baseline_tok_s'][0]:.1f}, {r['pred_baseline_tok_s'][2]:.1f}] | "
                  f"{r['normalized_speedup'][1]:.2f} [{r['normalized_speedup'][0]:.2f}, {r['normalized_speedup'][2]:.2f}] | "
                  f"{r['sol_tok_s']:.1f} ({'trace' if 'trace' in r['mstar_source'] else 'iid' if 'independent' in r['mstar_source'] else r['mstar_source']}) | "
                  f"{r['system_of_sol']:.0%} | {'; '.join(r['labels'])} |")
    L_ += ["", f"Summary: {json.dumps(summ)}", "", "Skipped: " + "; ".join(f"{x['id']} ({x['reason']})" for x in skipped)]
    open(os.path.join(a.out, "audit.md"), "w").write("\n".join(L_) + "\n")
    print("\n".join(L_[-4:]))


if __name__ == "__main__":
    main()
