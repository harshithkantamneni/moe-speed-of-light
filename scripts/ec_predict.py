"""Pre-registered predictions for the expert-cache runs (phase 2) on the A10.

Everything is fixed before any expert-cache run exists:
  * constants come from the A10 static-offload sweep only (prereg/a10/scored.json):
    per model the all-GPU step time T0 (intercept) and the cost s of offloading one
    layer (slope); a single CPU bandwidth Bc and hand-off cost h are fitted jointly to
    the four slopes via  s_m = h + k_m * S_m * (1/Bc - 1/Bg_m),  Bg_m = GPU bytes / T0_m;
  * hit/miss/admission streams come from mosl/ecsim.py (the cache's policy, re-implemented)
    replayed on the HF routing traces of the same corpus, decode windows as ec-bench runs them;
  * step time per layer l: serial  t = hits*a_l + h + miss*b_l,
                           overlap t = max(hits*a_l, miss*b_l) + h,
    with a_l = S_l/Bg, b_l = S_l/Bc, plus T_dense = T0 - sum_l k*a_l; admissions are copied
    asynchronously and only bind if their bytes exceed what the link moves in a step.

    python scripts/ec_predict.py
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mosl.ecsim import counts_init, load_steps, simulate
from mosl.gguf_bytes import GGUFS, account

OUT = "prereg/a10_ec"
N_PREFILL, N_DECODE = 128, 192
PCIE_GBS = 25.26                                           # measured pinned H2D, job 001
MODELS = {  # name -> (trace dir, corpus, slots per budget)
    "gpt-oss-20b-mxfp4":    ("data/traces/gpt-oss-20b",   "data/tok_gpt-oss-20b.jsonl", {0.125: 4, 0.25: 8, 0.5: 16}),
    "qwen3-30b-a3b-q4_k_m": ("data/traces/qwen3-30b-a3b", "data/tok_qwen3_30b.jsonl",   {0.125: 16, 0.25: 32, 0.5: 64}),
    "qwen3-30b-a3b-q8_0":   ("data/traces/qwen3-30b-a3b", "data/tok_qwen3_30b.jsonl",   {0.125: 16, 0.25: 32, 0.5: 64}),
}
POLICIES = ["static", "lru", "dfa"]


def split_sets(corpus, n_profile=2, n_test=4):
    """Per domain, in file order: first n_profile sequences profile the static hot set,
    the next n_test are the test set."""
    rows = [json.loads(l) for l in open(corpus)]
    prof, test, seen = [], [], {}
    for i, r in enumerate(rows):
        c = seen.get(r["domain"], 0)
        (prof if c < n_profile else test if c < n_profile + n_test else []).append(i)
        seen[r["domain"]] = c + 1
    return prof, test


def calibrate():
    sc = json.load(open("prereg/a10/scored.json"))
    pred = json.load(open("prereg/a10/predictions.json"))
    gbytes = {r["model"]: r["pred_gpu_dram_read_bytes_per_token"] for r in pred["sweep"] if r["n_cpu_moe"] == 0}
    rows = []
    for m, h4 in sc["H4"].items():
        g = account(*GGUFS[m])
        T0 = h4["meas_intercept_ms"] * 1e-3
        s = h4["meas_slope_us_per_layer"] * 1e-6
        by0 = gbytes.get(m) or (g.dense_bytes + g.head_bytes + g.top_k * sum(g.expert_bytes_per_layer))
        rows.append(dict(model=m, T0=T0, s=s, Bg=by0 / T0, k=g.top_k, S=float(np.mean(g.expert_bytes_per_layer))))
    def res(x):
        h, inv_bc = x
        return [(r["s"] - (h + r["k"] * r["S"] * (inv_bc - 1 / r["Bg"]))) / r["s"] for r in rows]
    fit = least_squares(res, [30e-6, 1 / 100e9], bounds=([0, 1 / 1000e9], [1e-3, 1 / 5e9]))
    h, inv_bc = fit.x
    for r in rows:
        r["s_fit"] = h + r["k"] * r["S"] * (inv_bc - 1 / r["Bg"])
    return dict(h=h, Bc=1 / inv_bc, per_model={r["model"]: r for r in rows})


def predict_model(m, cal, trace_dir, corpus):
    g = account(*GGUFS[m])
    c = cal["per_model"][m]
    S = np.array(g.expert_bytes_per_layer)
    a, b, h, k = S / c["Bg"], S / cal["Bc"], cal["h"], g.top_k
    T_dense = c["T0"] - k * a.sum()
    prof, test = split_sets(corpus)
    R_prof, E, _ = load_steps(trace_dir, corpus, prof, N_PREFILL, N_DECODE)
    R_test, _, win = load_steps(trace_dir, corpus, test, N_PREFILL, N_DECODE)
    out = dict(model=m, E=E, L=len(S), k=k, test_seqs=test, profile_seqs=prof, n_steps=int(len(R_test[0])),
               T_dense_ms=T_dense * 1e3, a_us=float(a.mean() * 1e6), b_us=float(b.mean() * 1e6), configs=[])
    L = len(S)
    for f, C in MODELS[m][2].items():
        n_static = int(round(L * (1 - f)))
        t_static = c["T0"] + n_static * c["s"]
        out["configs"].append(dict(budget=f, mode="llama.cpp static layers", n_cpu_moe=n_static, pred_tok_s=1 / t_static))
        init = counts_init(R_prof, E, C)
        for pol in POLICIES:
            hits, miss, adm = simulate(R_test, E, C, pol, init=init if pol == "static" else None)
            for overlap in ([True, False] if pol == "dfa" else [True]):
                gpu = hits * a[None, :]
                cpu = miss * b[None, :]
                lay = (np.maximum(gpu, cpu) if overlap else gpu + cpu) + h
                T = T_dense + lay.sum(1)
                T = np.maximum(T, (adm * S[None, :]).sum(1) / (PCIE_GBS * 1e9))
                out["configs"].append(dict(budget=f, mode=f"ec {pol}" + ("" if overlap else " serial"), slots=C,
                                           hit_rate=float(hits.sum() / (hits.sum() + miss.sum())),
                                           admits_per_step=float(adm.sum() / len(adm)), pred_tok_s=float(len(T) / T.sum())))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    cal = calibrate()
    res = dict(calibration=cal, n_prefill=N_PREFILL, n_decode=N_DECODE, models={})
    for m, (td, corpus, _) in MODELS.items():
        res["models"][m] = predict_model(m, cal, td, corpus)
    json.dump(res, open(os.path.join(OUT, "predictions.json"), "w"), indent=1, default=float)
    with open(os.path.join(OUT, "predictions.md"), "w") as f:
        f.write(f"# Pre-registered expert-cache predictions (A10)\n\ncalibration from the static sweep: "
                f"h = {cal['h']*1e6:.1f} us per layer hand-off, Bc = {cal['Bc']/1e9:.1f} GB/s\n\n")
        for m, r in res["models"].items():
            f.write(f"## {m}\n\nT_dense {r['T_dense_ms']:.2f} ms, a {r['a_us']:.1f} us, b {r['b_us']:.1f} us per expert; "
                    f"{r['n_steps']} decode steps over test sequences {r['test_seqs']}\n\n")
            f.write("| budget | configuration | slots / n_cpu_moe | predicted hit rate | admits/step | predicted tok/s |\n|---|---|---|---|---|---|\n")
            for c in r["configs"]:
                hr = f"{c['hit_rate']:.3f}" if "hit_rate" in c else ""
                ad = f"{c['admits_per_step']:.1f}" if "admits_per_step" in c else ""
                f.write(f"| {c['budget']:.3f} | {c['mode']} | {c.get('slots', c.get('n_cpu_moe'))} | {hr} | {ad} | {c['pred_tok_s']:.1f} |\n")
            f.write("\n")
    print(open(os.path.join(OUT, "predictions.md")).read())


if __name__ == "__main__":
    main()
