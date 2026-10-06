"""Job 098: FreeToken tuned per cell (five settings) against ours on one RTX 5090 + Ryzen 9 9950X host, with VRAM and
each engine's CPU-only decode. Paired by problem, 95% percentile bootstrap intervals (10,000 resamples). Writes
prereg/freetoken_tuned_098.json, paper/wsg_ft.tex (macros) and the job's clauses into prereg/scorecard_clauses.json.

    python scripts/freetoken_tuned.py
"""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
RES = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")
D = os.path.join(RES, "098_freetoken_tuned@vast")
CELLS = [("g", 14, "0.111", "gpt-oss 11%", "hybrid"), ("g", 32, "0.25", "gpt-oss 25%", "hybrid"), ("g", 51, "0.40", "gpt-oss 40%", "offload"),
         ("q", 16, "0.125", "Qwen3 12.5%", "hybrid"), ("q", 32, "0.25", "Qwen3 25%", "hybrid"), ("q", 56, "0.4375", "Qwen3 43.75%", "offload")]
HOST_S = {"gpt-oss 11%": 1.207, "gpt-oss 25%": 1.196, "gpt-oss 40%": 1.093, "Qwen3 12.5%": 1.027, "Qwen3 25%": 1.048, "Qwen3 43.75%": 0.974}


def boot(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    r = a / b
    idx = rng.integers(0, len(r), (n, len(r)))
    m = a[idx].mean(1) / b[idx].mean(1)
    return [float(a.mean() / b.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    rows = [json.loads(l) for l in open(os.path.join(D, "bs1.jsonl")) if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(r["label"], {})[r["problem"]] = r
    out = {"cells": []}
    M = {}
    clauses = []

    def add(cid, pred, short, clause, typ, qty, thr, meas, ci, st, dec=3, note=""):
        clauses.append(dict(id=f"098-{cid}", short=short, prediction=pred, clause=clause, type=typ, quantity=qty, threshold=thr,
                            source="prereg/freetoken_tuned_098.json", measured=round(meas, dec), ci=[round(x, dec) for x in ci] if ci else None,
                            status=st, paper_log="", note=note, decimals=dec))
    for p, C, r, lab, t1 in CELLS:
        o = by[f"{p}_ours_C{C}_law"]
        probs = sorted(o)
        fts = {k: v for k, v in by.items() if k.startswith(f"{p}_ft_") and f"_r{r}_" in k}
        means = {k: float(np.mean([v[q]["decode_tok_s"] for q in probs])) for k, v in fts.items()}
        best = max(means, key=means.get)
        default = f"{p}_ft_{t1}_r{r}_fm1_t0"
        ov = np.array([o[q]["decode_tok_s"] for q in probs])
        bv = np.array([fts[best][q]["decode_tok_s"] for q in probs])
        dv = np.array([fts[default][q]["decode_tok_s"] for q in probs])
        ob = boot(ov, bv); bd = boot(bv, dv)
        caps = [means[k] / means[f"{p}_ft_hybrid_r{r}_fm1_t0"] for k in (f"{p}_ft_hybrid_r{r}_f1_t0", f"{p}_ft_hybrid_r{r}_f2_t0")]
        thr8 = means[f"{p}_ft_hybrid_r{r}_fm1_t8"] / means[f"{p}_ft_hybrid_r{r}_fm1_t0"]
        vr_ft = sorted({round(x.get("vram_used_gib") or 0, 1) for v in fts.values() for x in v.values()})
        vr_o = sorted({round(x.get("vram_used_gib") or 0, 1) for x in o.values()})
        cell = dict(cell=lab, ours=float(ov.mean()), ft_means=means, ft_best=best, ft_default=default, ours_over_best=ob, best_over_default=bd,
                    fixed_cap_over_calibrated=caps, eight_threads_over_default=thr8, vram_ft=vr_ft, vram_ours=vr_o, host_s_ratio=HOST_S[lab])
        out["cells"].append(cell)
        tag = {"gpt-oss 11%": "GLow", "gpt-oss 25%": "GMid", "gpt-oss 40%": "GHigh", "Qwen3 12.5%": "QLow", "Qwen3 25%": "QMid", "Qwen3 43.75%": "QHigh"}[lab]
        M[f"ftt{tag}"] = f"{ob[0]:.2f}"
        t = lab.replace(" ", "").replace("%", "")
        st = "held" if bd[0] <= 1.10 and bd[2] <= 1.10 else ("held (point)" if bd[0] <= 1.10 else "failed")
        add(f"P1-{t}", 1, f"FreeToken's best at most 1.10x its Table 1 setting, {lab}", f"FreeToken's best setting is at most 1.10x its Table 1 setting at {lab}",
            "threshold", "best / Table 1 setting", "<= 1.10", bd[0], bd[1:], st)
        if lab in ("gpt-oss 11%", "gpt-oss 25%"):
            st = "held" if ob[1] > 1.05 else ("held (point)" if ob[0] > 1.05 else "failed")
            add(f"P2-{t}", 2, f"ours leads FreeToken's best by at least 5%, {lab}", f"ours / FreeToken-best > 1.05 at {lab}", "threshold", "ours / best", "> 1.05", ob[0], ob[1:], st)
        else:
            lo, hi = (0.95, 1.15) if lab == "gpt-oss 40%" else (0.90, 1.12)
            st = "held" if lo <= ob[1] and ob[2] <= hi else ("held (point)" if lo <= ob[0] <= hi else "failed")
            add(f"P2-{t}", 2, f"ours / FreeToken-best in {lo:.2f}-{hi:.2f}, {lab}", f"ours / FreeToken-best within {lo}-{hi} at {lab}", "band", "ours / best", f"{lo} to {hi}", ob[0], ob[1:], st)
        for k, cv in zip((1, 2), caps):
            m = 100 * (cv - 1)
            add(f"P3-{t}-cap{k}", 3, f"fetch cap {k} within 8% of the calibrated split, {lab}", f"FreeToken hybrid with the fetch cap fixed at {k} is within 8% of its calibrated split at {lab}",
                "band", "100 (cap / calibrated - 1)", "-8 to +8%", m, None, "held (point)" if abs(m) <= 8 else "failed", 1)
        m = 100 * (thr8 - 1)
        add(f"P4-{t}", 4, f"8 threads slower than default or within 3%, {lab}", f"8 CPU threads are slower than FreeToken's default or within 3% of it at {lab}", "threshold",
            "100 (8 threads / default - 1)", "<= +3%", m, None, "held (point)" if m <= 3 else "failed", 1)
        want = "offload" if lab in ("gpt-oss 40%", "Qwen3 43.75%") else "hybrid"
        add(f"P5-{t}", 5, f"FreeToken's best backend is {want}, {lab}", f"FreeToken's best backend is {want} at {lab}", "equality", "best backend", want,
            1.0 if want in best else 0.0, None, "held" if want in best else "failed", 0, note=f"best: {best}")
        st = "held" if all(27 <= v <= 30 for v in vr_ft) and max(vr_o) < min(vr_ft) else "failed"
        add(f"P6-{t}", 6, f"FreeToken 27-30 GiB, ours below it, {lab}", f"FreeToken holds 27-30 GiB at every run and ours less at {lab}", "band", "VRAM GiB (FreeToken; ours)",
            "27 to 30; ours < FreeToken", max(vr_ft), None, st, 1, note=f"FreeToken {vr_ft}, ours {vr_o}")
    cpu = {}
    for p, nm, n in (("g", "gpt-oss", 36), ("q", "Qwen3", 48)):
        l = by[f"{p}_llama_n{n}"]; f = by[f"{p}_ft_cpu"]
        probs = sorted(set(l) & set(f))
        r = boot(np.array([l[q]["decode_tok_s"] for q in probs]), np.array([f[q]["decode_tok_s"] for q in probs]))
        cpu[nm] = dict(llama=float(np.mean([l[q]["decode_tok_s"] for q in probs])), freetoken=float(np.mean([f[q]["decode_tok_s"] for q in probs])), ratio=r)
        thr = 2.0 if p == "g" else 1.2
        add(f"P7-{nm}", 7, f"CPU-only: llama.cpp at least {thr}x FreeToken's cpu backend, {nm}", f"stock llama.cpp with every MoE layer on the CPU decodes {nm} at least {thr}x FreeToken's cpu backend",
            "threshold", "llama.cpp / FreeToken cpu", f">= {thr}", r[0], r[1:], "held" if r[1] >= thr else ("held (point)" if r[0] >= thr else "failed"))
    out["cpu_only"] = cpu
    json.dump(out, open(P("prereg", "freetoken_tuned_098.json"), "w"), indent=1)
    obs = [c["ours_over_best"][0] for c in out["cells"]]
    M["fttMin"] = f"{min(obs):.2f}"; M["fttMax"] = f"{max(obs):.2f}"
    M["fttLeadCells"] = str(sum(c["ours_over_best"][1] > 1 for c in out["cells"]))
    M["fttBestOverDefaultMax"] = f"{max(c['best_over_default'][0] for c in out['cells']):.2f}"
    M["fttBestOverDefaultMin"] = f"{min(c['best_over_default'][0] for c in out['cells']):.2f}"
    M["fttVramFtMin"] = f"{min(min(c['vram_ft']) for c in out['cells']):.1f}"; M["fttVramFtMax"] = f"{max(max(c['vram_ft']) for c in out['cells']):.1f}"
    M["fttVramOursMin"] = f"{min(min(c['vram_ours']) for c in out['cells']):.1f}"; M["fttVramOursMax"] = f"{max(max(c['vram_ours']) for c in out['cells']):.1f}"
    import math
    M["fttHostSDiffMax"] = f"{math.ceil(100 * max(abs(c['ours_over_best'][0] - c['host_s_ratio']) for c in out['cells'])) / 100:.2f}"
    M["fttCapBestMax"] = f"{100 * (max(max(c['fixed_cap_over_calibrated']) for c in out['cells']) - 1):.1f}"
    M["fttCpuGpt"] = f"{cpu['gpt-oss']['ratio'][0]:.2f}"; M["fttCpuQwen"] = f"{cpu['Qwen3']['ratio'][0]:.2f}"
    with open(P("paper", "wsg_ft.tex"), "w") as f:
        f.write("% generated by scripts/freetoken_tuned.py from results/098_freetoken_tuned@vast\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    p = P("prereg", "scorecard_clauses.json")
    d = json.load(open(p))
    d["jobs"] = [j for j in d["jobs"] if j.get("job") != "098"] + [dict(job="098", era="088-098", script="jobs/098_freetoken_tuned@vast.sh", commit="73bf256",
                                                                         host="RTX 5090 (14,001 MHz) + Ryzen 9 9950X, Vast offer 52711021", outcome_note="prereg/freetoken_outcome_098.md", clauses=clauses)]
    if "098" not in d["meta"]["eras"]["088-098"]:
        d["meta"]["eras"]["088-098"].append("098")
    json.dump(d, open(p, "w"), indent=1, ensure_ascii=False)
    for c in out["cells"]:
        print(f"{c['cell']:14s} ours {c['ours']:6.1f}  FT best {c['ft_best']:30s} {c['ft_means'][c['ft_best']]:6.1f}  ours/best {c['ours_over_best'][0]:.3f} "
              f"[{c['ours_over_best'][1]:.3f}, {c['ours_over_best'][2]:.3f}]  best/default {c['best_over_default'][0]:.3f}  host S {c['host_s_ratio']}  vram FT {c['vram_ft']} ours {c['vram_ours']}")
    print("cpu-only", {k: (round(v["llama"], 1), round(v["freetoken"], 1), [round(x, 3) for x in v["ratio"]]) for k, v in cpu.items()})
    for c in clauses:
        if c["status"] == "failed":
            print("FAILED", c["id"], c["short"], c["measured"], c.get("note", ""))
    print(len(clauses), "clauses;", sum(c["status"] == "held" for c in clauses), "held,", sum(c["status"] == "held (point)" for c in clauses), "held (point)")


if __name__ == "__main__":
    main()
