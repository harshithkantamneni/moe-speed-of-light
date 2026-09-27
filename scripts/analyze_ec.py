"""Score the phase-2 (expert cache) pre-registration against the A10 runs.

    python scripts/analyze_ec.py <results/009_ec_correct> <results/010_ec_eval> [out dir]

H6 correctness, H7 speed-up at 25 % budget, H8 prediction error, H9 simulated vs measured
hit rate, H10 ordering (DFA >= LRU, DFA >= DFA-serial). Writes scored.json / scored.md.
"""
import glob
import json
import os
import re
import statistics as st
import sys

TAG = {"gpt-oss-20b-MXFP4": "gpt-oss-20b-mxfp4", "Qwen3-30B-A3B-Instruct-2507-Q4_K_M": "qwen3-30b-a3b-q4_k_m",
       "Qwen3-30B-A3B-Instruct-2507-Q8_0": "qwen3-30b-a3b-q8_0", "gpt-oss-120b-MXFP4": "gpt-oss-120b-mxfp4"}
LE = {"gpt-oss-20b-mxfp4": (24, 32), "qwen3-30b-a3b-q4_k_m": (48, 128), "qwen3-30b-a3b-q8_0": (48, 128), "gpt-oss-120b-mxfp4": (36, 128)}


def tok_s(rows):
    return sum(r["n_decode"] for r in rows) / (sum(r["decode_ms"] for r in rows) / 1000)


def measured(evaldir):
    out = {}
    for tag, m in TAG.items():
        res = {}
        p = os.path.join(evaldir, f"{tag}_ec.jsonl")
        if os.path.exists(p):
            by = {}
            for l in open(p):
                if l.strip():
                    r = json.loads(l)
                    by.setdefault(r["config"], []).append(r)
            for cfg, rows in by.items():
                kv = dict(x.split("=", 1) for x in cfg.split(":"))
                if kv.get("slots") == "1":
                    continue  # profile run
                pol = kv["policy"] + ("" if kv.get("overlap", "1") == "1" else " serial")
                stats = json.load(open(os.path.join(evaldir, os.path.basename(kv["stats"])))) if "stats" in kv else {}
                res[("ec " + pol, int(kv["slots"]))] = dict(tok_s=tok_s(rows), hit_rate=stats.get("hit_rate"),
                                                          admits_per_step=stats.get("admits", 0) / max(1, stats.get("steps", 1)))
        for p in glob.glob(os.path.join(evaldir, f"{tag}_static_n*.jsonl")):
            n = int(re.search(r"_n(\d+)\.jsonl$", p).group(1))
            rows = [json.loads(l) for l in open(p) if l.strip()]
            if rows:
                res[("llama.cpp static layers", n)] = dict(tok_s=tok_s(rows))
        out[m] = res
    return out


def main(corrdir, evaldir, outdir="prereg/a10_ec"):
    pred = json.load(open(os.path.join(outdir, "predictions.json")))
    meas = measured(evaldir)
    rows = []
    for m, pm in pred["models"].items():
        for c in pm["configs"]:
            key = (c["mode"], c.get("slots", c.get("n_cpu_moe")))
            mm = meas.get(m, {}).get(key)
            r = dict(model=m, budget=c["budget"], mode=c["mode"], size=key[1], pred_tok_s=c["pred_tok_s"],
                     pred_hit=c.get("hit_rate"), meas_tok_s=mm and mm["tok_s"], meas_hit=mm and mm.get("hit_rate"))
            if r["meas_tok_s"]:
                r["ape"] = abs(r["pred_tok_s"] - r["meas_tok_s"]) / r["meas_tok_s"]
            rows.append(r)
    # unpredicted model(s): measured only
    extra = []
    for m, res in meas.items():
        if m in pred["models"]:
            continue
        L, E = LE[m]
        for (mode, size), v in sorted(res.items()):
            extra.append(dict(model=m, mode=mode, size=size, meas_tok_s=v["tok_s"], meas_hit=v.get("hit_rate")))

    out = dict(rows=rows, unpredicted=extra)
    # H6
    cj = os.path.join(corrdir, "correctness.json")
    if os.path.exists(cj):
        c = json.load(open(cj))
        h6 = {}
        for tag in ("gpt-oss-20b-MXFP4", "Qwen3-30B-A3B-Instruct-2507-Q4_K_M"):
            gpu = c.get(f"{tag}:gpu", {}).get("top1_agreement")
            ecs = [c[k]["top1_agreement"] for k in c if k.startswith(tag + ":ec")]
            if ecs:
                h6[tag] = dict(gpu_vs_cpu=gpu, ec_min=min(ecs), pass_=min(ecs) >= 0.98 and (gpu is None or min(ecs) >= gpu - 0.01))
        out["H6"] = h6
    # H7: 25 % budget, dfa (overlap) vs static layers
    h7 = {}
    for m in pred["models"]:
        d = [r for r in rows if r["model"] == m and r["budget"] == 0.25]
        dfa = next((r for r in d if r["mode"] == "ec dfa"), None)
        sl = next((r for r in d if r["mode"] == "llama.cpp static layers"), None)
        if dfa and sl and dfa["meas_tok_s"] and sl["meas_tok_s"]:
            h7[m] = dict(speedup=dfa["meas_tok_s"] / sl["meas_tok_s"], pred_speedup=dfa["pred_tok_s"] / sl["pred_tok_s"],
                         pass_=dfa["meas_tok_s"] / sl["meas_tok_s"] >= 1.2)
    out["H7"] = h7
    ec_rows = [r for r in rows if r["mode"].startswith("ec") and r.get("ape") is not None]
    if ec_rows:
        apes = [r["ape"] for r in ec_rows]
        out["H8"] = dict(n=len(apes), median_ape=st.median(apes), mape=sum(apes) / len(apes), max_ape=max(apes),
                         pass_=st.median(apes) <= 0.15)
    h9 = [dict(model=r["model"], mode=r["mode"], size=r["size"], pred=r["pred_hit"], meas=r["meas_hit"],
               diff=r["meas_hit"] - r["pred_hit"]) for r in rows
          if r["mode"] in ("ec dfa", "ec lru") and r["meas_hit"] is not None and r["pred_hit"] is not None]
    if h9:
        out["H9"] = dict(rows=h9, max_abs_diff=max(abs(x["diff"]) for x in h9), pass_=all(abs(x["diff"]) <= 0.05 for x in h9))
    h10 = []
    for m in pred["models"]:
        for b in sorted({r["budget"] for r in rows if r["model"] == m}):
            g = {r["mode"]: r["meas_tok_s"] for r in rows if r["model"] == m and r["budget"] == b}
            if all(g.get(k) for k in ("ec dfa", "ec lru", "ec dfa serial")):
                h10.append(dict(model=m, budget=b, dfa_ge_lru=g["ec dfa"] >= g["ec lru"], dfa_ge_serial=g["ec dfa"] >= g["ec dfa serial"]))
    if h10:
        out["H10"] = dict(rows=h10, pass_=all(x["dfa_ge_lru"] and x["dfa_ge_serial"] for x in h10))

    json.dump(out, open(os.path.join(outdir, "scored.json"), "w"), indent=1, default=float)
    P = lambda x: "PASS" if x else "FAIL"
    with open(os.path.join(outdir, "scored.md"), "w") as f:
        f.write("# Expert cache on the A10: pre-registered hypotheses, scored\n\n")
        for tag, v in out.get("H6", {}).items():
            f.write(f"- H6 {tag}: cache top-1 agreement (min over configs) {100*v['ec_min']:.1f}%, all-GPU {100*(v['gpu_vs_cpu'] or 0):.1f}% -> {P(v['pass_'])}\n")
        for m, v in h7.items():
            f.write(f"- H7 {m}: DFA / static layers at 25% = {v['speedup']:.2f}x (predicted {v['pred_speedup']:.2f}x) -> {P(v['pass_'])}\n")
        if "H8" in out:
            v = out["H8"]
            f.write(f"- H8 cache tok/s prediction: median APE {100*v['median_ape']:.1f}%, MAPE {100*v['mape']:.1f}%, max {100*v['max_ape']:.1f}% (n={v['n']}) -> {P(v['pass_'])}\n")
        if "H9" in out:
            f.write(f"- H9 hit rate, simulated vs measured: max |diff| {100*out['H9']['max_abs_diff']:.1f} pp -> {P(out['H9']['pass_'])}\n")
        if "H10" in out:
            f.write(f"- H10 DFA >= LRU and DFA >= serial at every budget -> {P(out['H10']['pass_'])}\n")
        f.write("\n| model | budget | configuration | slots / n | measured tok/s | predicted | error | hit (meas / sim) |\n|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            mt = f"{r['meas_tok_s']:.1f}" if r["meas_tok_s"] else "—"
            er = f"{100*(r['pred_tok_s']/r['meas_tok_s']-1):+.0f}%" if r["meas_tok_s"] else ""
            hh = f"{r['meas_hit']:.3f} / {r['pred_hit']:.3f}" if r["meas_hit"] is not None and r["pred_hit"] is not None else ""
            f.write(f"| {r['model']} | {r['budget']} | {r['mode']} | {r['size']} | {mt} | {r['pred_tok_s']:.1f} | {er} | {hh} |\n")
        if extra:
            f.write("\n**Not predicted (measured only):**\n\n| model | configuration | slots / n | tok/s | hit |\n|---|---|---|---|---|\n")
            for r in extra:
                hh = f"{r['meas_hit']:.3f}" if r.get("meas_hit") is not None else ""
                f.write(f"| {r['model']} | {r['mode']} | {r['size']} | {r['meas_tok_s']:.1f} | {hh} |\n")
    print(open(os.path.join(outdir, "scored.md")).read())


if __name__ == "__main__":
    main(*sys.argv[1:])
