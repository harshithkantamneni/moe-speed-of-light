"""Statistics for job 078: Qwen3-30B-A3B BF16 on one RTX 5090 + Ryzen 9 9950X, four systems at equal GPU expert memory
(12.5 / 25 / 43.75% of experts), 30 AIME-25 problems, greedy, 256 tokens, held-out warm-up, one launch per config.

Per budget: ours = the better of FETCH off/on, FreeToken = the better of offload/hybrid (both picked by their mean over
the 30 problems; the interval does not include that selection). Ratios of mean speeds, paired by problem, with a 95%
percentile interval from 10,000 bootstrap resamples of the problems (seed 0). Scores the three predictions in the job
header.

    python scripts/qwen3_stats.py --dir /home/claude/gpu-branch/results/078_qwen3_4way@vast --out prereg/qwen3_078.json
"""
import argparse
import json
import os

import numpy as np

from samehost_stats import boot_ratio, load, per_problem

BUDGETS = [  # (label, llama.cpp, ours candidates, FreeToken candidates, KTransformers)
    ("12.5%", "llama_n42", ["oursv2_C16", "oursv2f_C16"], ["ft_offload_r0.125", "ft_hybrid_r0.125"], "kt_E16"),
    ("25%", "llama_n36", ["oursv2_C32", "oursv2f_C32"], ["ft_offload_r0.25", "ft_hybrid_r0.25"], "kt_E32"),
    ("43.75%", "llama_n27", ["oursv2_C56", "oursv2f_C56"], ["ft_offload_r0.4375", "ft_hybrid_r0.4375"], "kt_E56"),
]


def mean_all(by, lab):
    return float(np.mean([v for d in by[lab].values() for v in d.values()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/claude/gpu-branch/results/078_qwen3_4way@vast")
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(os.path.join(a.dir, "bs1.jsonl")) if line.strip()]
    by, meta = load(os.path.join(a.dir, "bs1.jsonl"))
    out = {"configs": {}, "comparisons": [], "predictions": {}}
    pt = {}
    for r in rows:
        pt.setdefault(r["label"], {})[r["problem"]] = r.get("prompt_tokens")
    print("configuration          mean tok/s  problems  VRAM GiB")
    for lab in sorted(by):
        out["configs"][lab] = dict(mean=mean_all(by, lab), problems=len(by[lab]), vram=sorted(meta[lab]["vram"]))
        print(f"{lab:22s} {mean_all(by, lab):9.2f}  {len(by[lab]):8d}  {sorted(meta[lab]['vram'])}")
    # same prompt length everywhere (same chat template / tokenizer)
    ref_pt = pt.get("llama_n36") or next(iter(pt.values()))
    out["prompt_tokens_equal"] = {lab: all(d.get(p) == ref_pt.get(p) for p in d) for lab, d in pt.items()}
    # same greedy text as llama.cpp (sha of the full output)
    ref = meta.get("llama_n36", {}).get("sha", {}).get(1, {})
    agree = {}
    for lab, m in meta.items():
        s = m["sha"].get(1) or next(iter(m["sha"].values()))
        common = [p for p in s if p in ref]
        agree[lab] = sum(1 for p in common if s[p] == ref[p]) / max(1, len(common))
    out["same_text_as_llama_n36"] = agree
    p1, p2, p3 = [], [], []
    for name, lc, ours, fts, kt in BUDGETS:
        ours = [o for o in ours if o in by]
        fts = [f for f in fts if f in by]
        if not ours or lc not in by:
            print(f"{name}: missing configurations")
            continue
        ob = max(ours, key=lambda o: mean_all(by, o))
        row = dict(budget=name, ours_best=ob, ours=mean_all(by, ob), llama=mean_all(by, lc),
                   ours_variants={o: mean_all(by, o) for o in ours})
        others = [(lc, "llama")]
        if fts:
            fb = max(fts, key=lambda f: mean_all(by, f))
            row.update(freetoken_best=fb, freetoken=mean_all(by, fb), ft_variants={f: mean_all(by, f) for f in fts})
            others.append((fb, "ft"))
        if kt in by:
            row["ktransformers"] = mean_all(by, kt)
            others.append((kt, "kt"))
        for lab, key in others:
            problems = sorted(set(by[ob]) & set(by[lab]))
            r = boot_ratio(per_problem(by, ob, problems), per_problem(by, lab, problems))
            row[f"ours_over_{key}"] = r
            row[f"n_{key}"] = len(problems)
        out["comparisons"].append(row)
        line = f"{name}: ours ({ob}) {row['ours']:.1f}, llama.cpp {row['llama']:.1f}"
        if "freetoken" in row:
            line += f", {row['freetoken_best']} {row['freetoken']:.1f}"
        if "ktransformers" in row:
            line += f", KTransformers {row['ktransformers']:.1f}"
        print(line + " tok/s")
        for key, lab in (("llama", "llama.cpp"), ("ft", "FreeToken"), ("kt", "KTransformers")):
            if f"ours_over_{key}" in row:
                m, lo, hi = row[f"ours_over_{key}"]
                print(f"   ours / {lab:13s} {m:.3f} [{lo:.3f}, {hi:.3f}]  (n={row[f'n_{key}']})")
        if "ours_over_ft" in row and name != "43.75%":
            p1.append(row["ours_over_ft"])
        if "ours_over_kt" in row:
            p2.append(row["ours_over_kt"])
        p3.append(row["ours_over_llama"])
    out["predictions"]["1 ours ahead of FreeToken's better backend at 12.5% and 25%"] = (
        all(m > 1 and lo > 1 for m, lo, _ in p1) if len(p1) == 2 else None)
    out["predictions"]["2 ours ahead of KTransformers at every budget"] = (
        all(m > 1 and lo > 1 for m, lo, _ in p2) if len(p2) == 3 else None)
    out["predictions"]["3 ours >= 1.5x stock llama.cpp at every budget"] = (
        all(m >= 1.5 for m, _, _ in p3) if len(p3) == 3 else None)
    print("same greedy text as llama_n36: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in sorted(agree.items())))
    print("prompt tokens equal to llama_n36: " + ", ".join(f"{k} {v}" for k, v in sorted(out["prompt_tokens_equal"].items())))
    for k, v in out["predictions"].items():
        print(f"prediction {k}: {'HELD' if v else 'FAILED' if v is not None else 'n/a (missing data)'}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
