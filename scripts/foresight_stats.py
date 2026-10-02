"""Statistics for job 093: what foresight is worth in the engine, measured.

Per cell (gpt-oss-120b C 14 / 32 / 51, Qwen3-30B-A3B C 16 / 32 / 56) the job ran llama-ec-bench teacher-forced over the
models' own greedy AIME-25 text (30 sequences x 256 decode steps) with the Table 1 engine (base), the same with paced
copies (paced), the oracle policy at W = 2, 4, 16, 64 and 0 (the rest of the sequence) with paced copies, the oracle at
W = 0 with the unpaced copy path (oracle_np), base without CPU/GPU overlap within a layer (noovl) and base with no
FETCH table (allcpu). Speeds are per-sequence tok/s; ratios to base are of mean speeds, paired by sequence, 95%
percentile interval over 10,000 bootstrap resamples (seed 0). Each run's cache counters (hits, misses, admissions,
fetches) come from its stats file.

The model's foresight term for the same host: scripts/shapley_gap.value_function with this host's probe (concur.txt),
this host's base measurement and base counters, and the exact optimum's reads (prereg/speed_limit_v2.json and
prereg/shapley_gap.json): v(F) = the model with only the foresight fix applied, v(all) = the limit, v(none) = base.
"Foresight recovered" = (T_base - T_oracle) / (T_base - v(F)); "share of the gap closed" = (T_base - T_oracle) /
(T_base - T_limit). The law is also evaluated on the oracle run's own counters, to see how much of its reads overlapped.

    python scripts/foresight_stats.py --out prereg/foresight_093.json
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, "/home/claude/gpu-branch/jobs/ec2")
from fetch_table import bandwidths  # noqa: E402
from scripts.shapley_gap import value_function, shapley, FIX5  # noqa: E402
from scripts.speed_limit import MODELS, host_rates, limit  # noqa: E402

R = "/home/claude/gpu-branch/results"
D = f"{R}/093_foresight@vast"
CELLS = [("g", 14, "gpt-oss-120b", "11%"), ("g", 32, "gpt-oss-120b", "25%"), ("g", 51, "gpt-oss-120b", "40%"),
         ("q", 16, "qwen3-30b-a3b-bf16", "12.5%"), ("q", 32, "qwen3-30b-a3b-bf16", "25%"), ("q", 56, "qwen3-30b-a3b-bf16", "43.75%")]
LABEL = {"gpt-oss-120b": "gpt-oss", "qwen3-30b-a3b-bf16": "Qwen3"}
OPT_HIT = {("g", 14): 73.4, ("g", 32): 89.3, ("g", 51): 95.3, ("q", 16): 74.9, ("q", 32): 88.7, ("q", 56): 96.4}   # the header's
W50_TRACE = {("g", 14): 3.1, ("g", 32): 9.4, ("q", 16): 1.5, ("q", 32): 3.7}


def boot_ratio(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    r = a[idx].mean(1) / b[idx].mean(1)
    return [float(a.mean() / b.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def boot_mean(a, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    m = a[idx].mean(1)
    return [float(a.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def label_of(cfg):
    for part in cfg.split(":"):
        if part.startswith("stats="):
            return os.path.basename(part[6:]).replace(".json", "")
    return cfg[:40]


def read_cell(tag, C):
    p = f"{D}/ec_{tag}_C{C}.jsonl"
    if not os.path.exists(p):
        return None
    by = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            by.setdefault(label_of(r["config"]), {})[r["seq"]] = r["tok_s"]
    out = {}
    for lab, v in by.items():
        short = lab.split(f"_C{C}_", 1)[1] if f"_C{C}_" in lab else lab
        stf = f"{D}/{lab}.json"
        st = json.load(open(stf)) if os.path.exists(stf) else {}
        n = max(1, st.get("steps", 1))
        out[short] = dict(label=lab, tok_s=v, mean=float(np.mean(list(v.values()))), n_seq=len(v),
                          hit_rate=st.get("hit_rate"), misses_per_token=st.get("misses", 0) / n, admits_per_token=st.get("admits", 0) / n,
                          fetches_per_token=st.get("fetches", 0) / n, oracle_refused=st.get("oracle_refused"), steps=st.get("steps"),
                          host_us=st.get("host_us_per_step"), fetch_table=st.get("fetch_table"), paced=st.get("paced"), overlap=st.get("overlap"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--job", default="093_foresight@vast", help="results directory under the gpu branch (093_foresight@vast or 094_foresight_bytes@vast)")
    a = ap.parse_args()
    global D
    D = f"{R}/{a.job}"
    txt = open(f"{D}/concur.txt").read()
    cores = int(re.search(r"usable physical cores (\d+)", open(f"{D}/cores.txt").read()).group(1))
    H = cores - 2 if cores > 4 else 2
    bc, bp, bb = bandwidths(txt, H)
    b_host = host_rates(txt)
    v2 = json.load(open(os.path.join(ROOT, "prereg", "speed_limit_v2.json")))
    sg = json.load(open(os.path.join(ROOT, "prereg", "shapley_gap.json")))
    opt_split = {c["cell"]: c["reads_per_token"]["exact"] for c in sg["inputs"]["cells"]}   # optimum's cpu/copy split per cell
    online_split = {c["cell"]: c["reads_per_token"]["online"] for c in sg["inputs"]["cells"]}   # the best online policy's
    law_oracle = json.load(open(f"{D}/law_oracle.json")) if os.path.exists(f"{D}/law_oracle.json") else {}
    print(f"host: B_c {bc:.1f} B_p {bp:.1f} B_cp {bb:.1f} GB/s at {H} helpers; limit's host rate {b_host / 1e9:.1f}")
    cells = []
    for tag, C, key, budget in CELLS:
        runs = read_cell(tag, C)
        if not runs or "base" not in runs:
            print(f"{key} C{C}: no runs"); continue
        base = runs["base"]
        seqs = sorted(base["tok_s"])
        bv = np.array([base["tok_s"][s] for s in seqs])
        for short, r in runs.items():
            common = [s for s in seqs if s in r["tok_s"]]
            r["ratio_to_base"] = boot_ratio(np.array([r["tok_s"][s] for s in common]), np.array([base["tok_s"][s] for s in common]))
            r["mean_ci"] = boot_mean(np.array([r["tok_s"][s] for s in r["tok_s"]]))
        # the limit on this host and the model's foresight term
        m = v2["models"][key]; row = m["rows"][str(C)]; mm = MODELS[key]
        L, k, S, Dd = m["L"], m["k"], mm["S"], mm["D"]
        reads = row["reads_per_token"]["exact"]
        t_lim, c_at = limit(reads, L * k, S, Dd, b_host, v2["B_gpu_datasheet"])
        cell_label = f"{LABEL[key]} {budget}"
        osplit = opt_split.get(cell_label)
        st = json.load(open(f"{D}/{base['label']}.json"))
        n = st["steps"]; hu = st["host_us_per_step"]
        ci = dict(cell=cell_label, model=key, C=C, S=S, D=Dd, Lk=L * k, B_gpu=v2["B_gpu_datasheet"],
                  reads=dict(engine=((st["misses"] - st["fetches"]) / n, (st["fetches"] + st["admits"] + st.get("prefetches", 0)) / n),
                             exact=(osplit["cpu"], osplit["copy"]), online=(online_split[cell_label]["cpu"], online_split[cell_label]["copy"])),
                  hw=(hu["app"] + hu["pre"] + hu["inputs"] + hu["launch"] + hu["post"]) * 1e-6, T=1 / base["mean"],
                  limit_json=t_lim, c_at_limit=c_at, online_limit_json=None)
        v, info = value_function(ci, b_host, (bc * 1e9, bp * 1e9, bb * 1e9), opt="exact")
        vF = v((0, 1, 0, 0, 0)); vO = v((1, 0, 0, 0, 0)); vOF = v((1, 1, 0, 0, 0)); v_all = v((1, 1, 1, 1, 1)); v_none = v((0, 0, 0, 0, 0))
        sh = shapley(v, 5)
        shares = {f: float(x) for f, x in zip(FIX5, sh[0])} if isinstance(sh, tuple) else {}
        model = dict(v_none_ms=1e3 * v_none, v_F_ms=1e3 * vF, v_O_ms=1e3 * vO, v_OF_ms=1e3 * vOF, v_all_ms=1e3 * v_all, limit_ms=1e3 * t_lim,
                     limit_tok_s=1 / t_lim, eta=info["eta"], resid_ms=info["resid_ms"], c_at_limit=c_at,
                     shapley_share={f: shares[f] / max(1e-12, (v_none - v_all)) for f in shares} if shares else None,
                     shapley_ms={f: 1e3 * shares[f] for f in shares} if shares else None)
        # the law on each run's own counters (misses on the CPU, fetches + admissions over PCIe), through the engine's probe
        for short, r in runs.items():
            cpu, link = r["misses_per_token"] - r["fetches_per_token"], r["fetches_per_token"] + r["admits_per_token"]
            r["law_ms"] = 1e3 * (ci["hw"] + info["eta"] * (Dd + (L * k - cpu) * S) / v2["B_gpu_datasheet"]
                                 + max(cpu * S / (bc * 1e9), link * S / (bp * 1e9), (cpu + link) * S / (bb * 1e9)))
            r["ms"] = 1e3 / r["mean"]
            r["frac_of_limit"] = r["mean"] * t_lim
        T_base = 1e3 / base["mean"]
        for short, r in runs.items():
            r["gain"] = r["ratio_to_base"][0] - 1
            r["gap_closed"] = (T_base - r["ms"]) / (T_base - 1e3 * t_lim)
            r["foresight_recovered"] = (T_base - r["ms"]) / (T_base - 1e3 * vF) if T_base > 1e3 * vF else None
        # W sweep: the W at which half of the W = 0 gain is reached (log interpolation)
        ws = sorted([(int(s.split("oracle_w")[1]), runs[s]["gain"]) for s in runs if s.startswith("oracle_w") and s != "oracle_w0"])
        w50 = None
        if "oracle_w0" in runs and ws:
            g0 = runs["oracle_w0"]["gain"]
            if g0 > 0:
                half = 0.5 * g0
                prev = (0, 0.0)
                for w, g in ws:
                    if g >= half:
                        w0, g0_ = prev
                        if w0 == 0:
                            w50 = w * (half / g) if g > 0 else w
                        else:
                            w50 = float(np.exp(np.log(w0) + (np.log(w) - np.log(w0)) * (half - g0_) / (g - g0_)))
                        break
                    prev = (w, g)
                if w50 is None:
                    w50 = float("inf")
        cell = dict(model=key, C=C, budget=budget, label=cell_label, runs=runs, model_terms=model, w50_measured=w50,
                    w50_trace=W50_TRACE.get((tag, C)), opt_hit_rate=OPT_HIT[(tag, C)], opt_reads_per_token=reads,
                    law_prediction=law_oracle.get("predicted", {}).get(f"{tag}_C{C}"))
        cells.append(cell)
        print(f"\n{cell_label} (C {C}): base {base['mean']:.1f} tok/s ({T_base:.2f} ms); limit {1 / t_lim:.0f} tok/s; model v(F) {1e3 * vF:.2f} ms "
              f"= {1e3 / (1e3 * vF):.1f} tok/s; v(O) {1e3 * vO:.2f}; v(O,F) {1e3 * vOF:.2f}; shares {({f: round(100 * s) for f, s in model['shapley_share'].items()} if model['shapley_share'] else None)}")
        for short in ("base", "paced", "oracle_w2", "oracle_w4", "oracle_w16", "oracle_w64", "oracle_w0", "oracle_np", "noovl", "allcpu",
                      "bypass_w0", "bypass_w16", "prefetch_w0"):
            if short in runs:
                r = runs[short]
                print(f"  {short:10s} {r['mean']:7.1f} tok/s  x base {r['ratio_to_base'][0]:.3f} [{r['ratio_to_base'][1]:.3f}, {r['ratio_to_base'][2]:.3f}]  "
                      f"hit {100 * (r['hit_rate'] or 0):.1f}%  misses/tok {r['misses_per_token']:.1f} admits/tok {r['admits_per_token']:.1f} fetches/tok {r['fetches_per_token']:.1f}  "
                      f"law {r['law_ms']:.2f} ms vs {r['ms']:.2f}  frac {100 * r['frac_of_limit']:.0f}%  gap closed {100 * r['gap_closed']:.0f}%  "
                      f"foresight recovered {('%.0f%%' % (100 * r['foresight_recovered'])) if r['foresight_recovered'] is not None else '--'}")
        print(f"  W50 measured {w50} (trace {W50_TRACE.get((tag, C))}); optimum hit rate {OPT_HIT[(tag, C)]}%")
    out = dict(host=dict(B_c=bc, B_p=bp, B_cp=bb, helpers=H, b_host_max=b_host / 1e9), cells=cells)
    for f in ("gate.txt", "tables.txt", "oneline.txt", "skipped.txt", "runs.txt", "configs.txt", "patch_sha.txt"):
        if os.path.exists(f"{D}/{f}"):
            out[f] = open(f"{D}/{f}").read()
    for f in glob.glob(f"{D}/la_*.bin.json"):
        out[os.path.basename(f)] = open(f).read()
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
