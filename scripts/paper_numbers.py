"""Emit paper/numbers.tex: every number quoted in the paper, computed from results/.

The paper never hard-codes a result; it uses these macros, so the text cannot
drift from the data.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
put("tauH", P["tau_us"]); put("tauX", P["tau_x_us"] / 1000, "{:.1f}"); put("tauE", P.get("tau_e_us", 0.0))
put("selectedVariant", v["selected"].split()[0])
MEAS = {x["id"]: x for x in json.load(open("data/published_measurements.json"))}
from mosl.validation_set import ROWS
srcs = {MEAS[r.get("src", r["id"])]["source_url"].split("#")[0] for r in ROWS}
put("nSources", len(srcs)); put("nModelsVal", len({r["repo"] for r in ROWS}))
put("nUnstatedRam", sum(1 for r in ROWS if "RAM speed not stated" in r.get("assume", "")))
pm = v["loso_per_model"]
put("qwenThirtyResid", pct(pm["Qwen/Qwen3-30B-A3B"]["median_abs"]))
# hybrid (dynamic cache + CPU misses) consistency check: DeepSeek-V4-Flash rows P079-P082
T0, T53 = 1 / 13.61, 1 / 15.84
d = (T0 - T53) / 0.53
for h, meas, nm in ((0.76, 17.58, "SeventySix"), (0.806, 17.66, "Eighty")):
    pred = 1 / (T0 - h * d)
    put(f"hyb{nm}Pred", pred, "{:.2f}"); put(f"hyb{nm}Err", pct(abs(pred - meas) / meas), "{:.0f}")
put("hybCpuShare", pct(d / T0))
for name, rep in v["variants"].items():
    tag = {"M0": "MZero", "M1": "MOne", "M2": "MTwo", "M3": "MThree", "M4": "MFour"}[name.split()[0]]
    put(f"cv{tag}", pct(rep["loso"]["median_ape"])); put(f"cvMape{tag}", pct(rep["loso"]["mape"]))
    put(f"in{tag}", pct(rep["in_sample"]["median_ape"]))
eng = v["per_engine_eta_c"]
for e, tag in (("llama.cpp", "Llama"), ("ik_llama.cpp", "Ik"), ("KTransformers", "Kt"), ("OSDI26-engine", "Osdi")):
    if e in eng:
        put(f"eng{tag}", eng[e]["eta_c"], "{:.2f}")

import csv
meas = [float(r["measured"]) for r in csv.DictReader(open("results/validation.csv")) if r["kind"] != "numa_holdout"]
put("tokMin", min(meas), "{:.2f}"); put("tokMax", max(meas), "{:.0f}")

TAGS = {"olmoe-1b-7b": "Olmoe", "qwen3-30b-a3b": "Qwen", "gpt-oss-20b": "Gptsmall"}
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
    ci = a.get("hit_rate_ci_0.25", {})
    for k, n in (("lru", "Lru"), ("static_hot_xdomain", "Xdom"), ("min_bypass", "Bypass")):
        lo, hi = ci.get(k, [0, 0])
        put(f"{t}Hit{n}Ci", f"{pct(lo):.0f}--{pct(hi):.0f}")
    lo, hi = a.get("reuse1_ci", [0, 0]); put(f"{t}ReuseOneCi", f"{pct(lo):.0f}--{pct(hi):.0f}")
    put(f"{t}MissRatio", (1 - h["lru"]) / max(1e-9, 1 - h["min_bypass"]), "{:.1f}")
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
            put(f"{t}{pt}LruSeq{tt}", d["lru_cpuseq"], "{:.1f}"); put(f"{t}{pt}SolSerial{tt}", d["speed_of_light_serial"], "{:.1f}")
            put(f"{t}{pt}Oracle{tt}", d["oracle_prefetch_cpu"], "{:.1f}")
            put(f"{t}{pt}BypassOfSol{tt}", pct(d["min_bypass_cpu"] / d["speed_of_light"]))
            put(f"{t}{pt}LruSeqGain{tt}", d["lru_cpuseq"] / d["static_layer_cpu"], "{:.2f}")
            put(f"{t}{pt}CpuGain{tt}", d["lru_cpu"] / d["all_experts_cpu"], "{:.1f}")
        sv = mm.get("sensitivity", {}).get("eta_g=0.6")
        if sv:
            put(f"{t}{pt}LruGainSens", sv["lru_cpu"] / sv["static_layer_cpu"], "{:.2f}")
            put(f"{t}{pt}LruOfSolSens", pct(sv["lru_cpu"] / sv["speed_of_light"]))
    for pname, pt in PL.items():
        a25 = a["tok_s"]["tau=fitted"][pname]["0.25"]
        put(f"{t}{pt}QStatic", a25["static_layer_cpu"], "{:.0f}"); put(f"{t}{pt}QLru", a25["lru_cpu"], "{:.0f}")
        put(f"{t}{pt}QLruFetch", a25["lru_fetch"], "{:.0f}"); put(f"{t}{pt}QSol", a25["speed_of_light"], "{:.0f}")
        put(f"{t}{pt}QStaticOfSol", pct(a25["static_layer_cpu"] / a25["speed_of_light"]))
        put(f"{t}{pt}QLruOfSol", pct(a25["lru_cpu"] / a25["speed_of_light"]))
        put(f"{t}{pt}QLruGain", a25["lru_cpu"] / a25["static_layer_cpu"], "{:.1f}")
        put(f"{t}{pt}QBypassOfSol", pct(a25["min_bypass_cpu"] / a25["speed_of_light"]))
        put(f"{t}{pt}QBypass", a25["min_bypass_cpu"], "{:.0f}"); put(f"{t}{pt}QOracle", a25["oracle_prefetch_cpu"], "{:.0f}")
        put(f"{t}{pt}QOracleOfSol", pct(a25["oracle_prefetch_cpu"] / a25["speed_of_light"]))
        put(f"{t}{pt}QLruSeq", a25["lru_cpuseq"], "{:.0f}"); put(f"{t}{pt}QSolSerial", a25["speed_of_light_serial"], "{:.0f}")
        put(f"{t}{pt}QAllCpu", a["tok_s"]["tau=fitted"][pname]["all_experts_cpu"], "{:.0f}")
PLACEHOLDER = False

if os.path.exists("results/bf16_check.json"):
    bf = json.load(open("results/bf16_check.json"))
    put("olmoeBfDiff", pct(bf["frac_token_layer_sets_differ"]), "{:.1f}")
    put("olmoeBfHitDelta", pct(abs(bf["lru25_bf16"] - bf["lru25_fp32"])), "{:.1f}")
else:
    put("olmoeBfDiff", "\\textbf{??}"); put("olmoeBfHitDelta", "\\textbf{??}")
os.makedirs("paper", exist_ok=True)
open("paper/numbers.tex", "w").write("\n".join(out) + "\n")
print(len(out), "macros")

# ---- ranges quoted in the text, computed across models/platforms ----
out2 = []
def put2(n, v):
    out2.append("\\newcommand{\\%s}{%s}" % (n, v))
AN = {m: json.load(open(f"results/analysis_{m}.json")) for m in ("olmoe-1b-7b", "qwen3-30b-a3b", "gpt-oss-20b")}
MIDHI = ["RTX 4090 + DDR5-6000 (PCIe4 x16)", "RTX 5090 + DDR5-6400 (PCIe5 x16)"]
LOW = "RTX 4060 8GB + DDR5-5600 (PCIe4 x8)"
def rng(vals, f="{:.0f}"):
    return f"{f.format(min(vals))}--{f.format(max(vals))}"
q = [AN[m]["tok_s"]["tau=fitted"][p]["0.25"] for m in AN for p in MIDHI]
put2("rngQStatic", rng([pct(d["static_layer_cpu"] / d["speed_of_light"]) for d in q]))
put2("rngQLru", rng([pct(d["lru_cpu"] / d["speed_of_light"]) for d in q]))
put2("rngQOracle", rng([pct(d["oracle_prefetch_cpu"] / d["speed_of_light"]) for d in q]))
put2("rngQGain", rng([d["lru_cpu"] / d["static_layer_cpu"] for d in q], "{:.1f}"))
off = [m for m in AN if AN[m]["matched"][LOW]["frac"] < 1.0]
lo = [AN[m]["matched"][LOW]["tok_s"]["tau=fitted"] for m in off]
put2("rngLowLruOfSol", rng([pct(d["lru_cpu"] / d["speed_of_light"]) for d in lo]))
put2("rngLowGain", rng([d["lru_cpu"] / d["static_layer_cpu"] for d in lo], "{:.2f}"))
put2("rngLowGainSens", rng([AN[m]["matched"][LOW]["sensitivity"]["eta_g=0.6"]["lru_cpu"] / AN[m]["matched"][LOW]["sensitivity"]["eta_g=0.6"]["static_layer_cpu"] for m in off], "{:.2f}"))
put2("rngLowLruOfSolSens", rng([pct(AN[m]["matched"][LOW]["sensitivity"]["eta_g=0.6"]["lru_cpu"] / AN[m]["matched"][LOW]["sensitivity"]["eta_g=0.6"]["speed_of_light"]) for m in off]))
strat = ("static_layer_cpu", "static_hot_xdomain_cpu", "lru_cpu", "lru_cpuseq", "min_bypass_cpu", "lru_fetch", "oracle_prefetch_cpu")
put2("maxLowOverCpu", "{:.2f}".format(max(d[k] / d["all_experts_cpu"] for d in lo for k in strat)))
# claim (i): CPU-miss LRU vs fetch-on-miss LRU across all fractions and hand-off latencies
def ratio_min(plats):
    r = []
    for m in AN:
        for tau, T in AN[m]["tok_s"].items():
            for p in plats:
                for f, d in T[p].items():
                    if isinstance(d, dict):
                        r.append(d["lru_cpu"] / d["lru_fetch"])
    return min(r)
put2("pcieFourMinAdv", "{:.2f}".format(ratio_min([LOW, MIDHI[0]])))
put2("pcieFiveMinAdv", "{:.2f}".format(ratio_min([MIDHI[1]])))
# extra memory the per-layer cache gets vs the layer split at matched points
ex = []
for m in off:
    mm = AN[m]["matched"][LOW]
    ex.append(pct(mm["cap_per_layer"] / AN[m]["E"] - mm["gpu_layers"] / AN[m]["L"]))
put2("matchedExtraMem", rng(ex))
# r* per model
for m, t in (("qwen3-30b-a3b", "Qwen"), ("gpt-oss-20b", "Gptsmall"), ("olmoe-1b-7b", "Olmoe")):
    pass
open("paper/numbers.tex", "a").write("\n".join(out2) + "\n")
print("\n".join(out2))
open("paper/numbers.tex", "a").write("\\newcommand{\\pcieFiveFetchGain}{%.0f}\n" % (100 * (1 / float([x for x in out2 if "pcieFiveMinAdv" in x][0].split("{")[-1].rstrip("}")) - 1)))
