"""Emit paper/numbers.tex: every number quoted in the paper, computed from results/.

The paper never hard-codes a result; it uses these macros, so the text cannot
drift from the data.
"""
import json
import os

out = []


PLACEHOLDER = False


def put(name, value, fmt="{:.0f}"):
    if PLACEHOLDER:
        value, fmt = "\\textbf{??}", "{}"
    out.append("\\newcommand{\\%s}{%s}" % (name, fmt.format(value) if not isinstance(value, str) else value))


def pct(x):
    return 100 * x


v = json.load(open("results/validation.json"))
sel = v["variants"][v["selected"]]
put("nRows", v["n_rows"]); put("nGroups", v["n_groups"])
put("cvMedian", pct(sel["loso"]["median_ape"])); put("cvMape", pct(sel["loso"]["mape"]))
put("cvWithinTwentyFive", pct(sel["loso"]["within_25"])); put("cvMax", pct(sel["loso"]["max_ape"]))
for k, n in (("cpu_only", "Cpu"), ("static", "Static"), ("fetch", "Fetch")):
    put(f"cvMedian{n}", pct(sel["loso_by_kind"][k]["median_ape"])); put(f"n{n}", sel["loso_by_kind"][k]["n"])
put("numaMedian", pct(sel["numa_holdout"]["median_ape"])); put("nNuma", sel["numa_holdout"]["n"])
P = sel["params"]
put("etaG", P["eta_g"], "{:.2f}"); put("etaC", P["eta_c"], "{:.2f}"); put("etaP", P["eta_p"], "{:.2f}")
put("tauH", P["tau_us"]); put("tauX", P["tau_x_us"] / 1000, "{:.1f}")
for name, rep in v["variants"].items():
    tag = {"M0": "MZero", "M1": "MOne", "M2": "MTwo", "M3": "MThree"}[name.split()[0]]
    put(f"cv{tag}", pct(rep["loso"]["median_ape"])); put(f"cvMape{tag}", pct(rep["loso"]["mape"]))
    put(f"in{tag}", pct(rep["in_sample"]["median_ape"]))
eng = v["per_engine_eta_c"]
for e, tag in (("llama.cpp", "Llama"), ("ik_llama.cpp", "Ik"), ("KTransformers", "Kt"), ("OSDI26-engine", "Osdi")):
    if e in eng:
        put(f"eng{tag}", eng[e]["eta_c"], "{:.2f}")

import csv
meas = [float(r["measured"]) for r in csv.DictReader(open("results/validation.csv")) if r["kind"] != "numa_holdout"]
put("tokMin", min(meas), "{:.2f}"); put("tokMax", max(meas), "{:.0f}")

TAGS = {"olmoe-1b-7b": "Olmoe", "qwen3-30b-a3b": "Qwen", "gpt-oss-20b": "Gptsmall", "gpt-oss-120b": "Gptbig"}
PL = {"RTX 4060 8GB + DDR5-5600 (PCIe4 x8)": "Low", "RTX 4090 + DDR5-6000 (PCIe4 x16)": "Mid",
      "RTX 5090 + DDR5-6400 (PCIe5 x16)": "High"}
for m, t in TAGS.items():
    f = f"results/analysis_{m}.json"
    PLACEHOLDER = not os.path.exists(f)
    a = json.load(open(f if not PLACEHOLDER else "results/analysis_qwen3-30b-a3b.json"))
    L = a["locality"]
    put(f"{t}Tokens", a["decode_tokens"], "{:,}")
    put(f"{t}Ppl", a["teacher_forced_ppl"], "{:.2f}")
    put(f"{t}E", a["E"]); put(f"{t}K", a["k"]); put(f"{t}L", a["L"])
    put(f"{t}ReuseOne", pct(L["reuse_within_1"])); put(f"{t}ReuseSixteen", pct(L["reuse_within_16"]))
    put(f"{t}Uniform", pct(L["uniform_reuse_within_1"]), "{:.1f}")
    put(f"{t}ReuseX", L["reuse_within_1"] / L["uniform_reuse_within_1"], "{:.1f}")
    put(f"{t}TopQuarter", pct(L["top25pct_share"]))
    put(f"{t}Jaccard", L["domain_top25_jaccard"], "{:.2f}"); put(f"{t}JaccardRand", L["random_top25_jaccard"], "{:.2f}")
    h = a["hit_rate"]["0.25"]
    for k, n in (("static_layer", "Layer"), ("static_hot_xdomain", "Xdom"), ("static_hot_oracle", "Oracle"),
                 ("lru", "Lru"), ("lfu", "Lfu"), ("min_fetch", "Min"), ("min_bypass", "Bypass")):
        put(f"{t}Hit{n}", pct(h[k]))
    for pname, pt in PL.items():
        put(f"{t}Rstar{pt}", a["r_star"][pname], "{:.1f}")
        mm = a["matched"][pname]
        put(f"{t}Frac{pt}", pct(mm["frac"]))
        for tau, tt in (("tau=fitted", ""), ("tau=20us", "Fast")):
            d = mm["tok_s"][tau]
            for k, n in (("static_layer_cpu", "Static"), ("static_hot_xdomain_cpu", "Hot"), ("lru_cpu", "LruCpu"),
                         ("lru_fetch", "LruFetch"), ("min_bypass_cpu", "BypassCpu"), ("speed_of_light", "Sol"),
                         ("all_experts_cpu", "AllCpu")):
                put(f"{t}{pt}{n}{tt}", d[k], "{:.1f}")
            put(f"{t}{pt}StaticOfSol{tt}", pct(d["static_layer_cpu"] / d["speed_of_light"]))
            put(f"{t}{pt}LruOfSol{tt}", pct(d["lru_cpu"] / d["speed_of_light"]))
            put(f"{t}{pt}LruGain{tt}", d["lru_cpu"] / d["static_layer_cpu"], "{:.2f}")
    a25 = a["tok_s"]["tau=fitted"]["RTX 4090 + DDR5-6000 (PCIe4 x16)"]["0.25"]
    put(f"{t}MidQuarterStatic", a25["static_layer_cpu"], "{:.0f}"); put(f"{t}MidQuarterLru", a25["lru_cpu"], "{:.0f}")
    put(f"{t}MidQuarterSol", a25["speed_of_light"], "{:.0f}")
PLACEHOLDER = False

os.makedirs("paper", exist_ok=True)
open("paper/numbers.tex", "w").write("\n".join(out) + "\n")
print(len(out), "macros")
