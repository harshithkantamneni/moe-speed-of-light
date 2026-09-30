"""Statistics for job 082 (the review's run 1) on the headline machine (RTX 5090 + 9950X3D, listing B).

A. Steady state on held-out prompts: 10 MATH-500 problems x 2,048 tokens; per-request window rates from the client
   (tokens 1-256 and 257-2048). Ours / FreeToken paired by problem, 95% percentile interval over 10,000 bootstrap
   resamples of the 10 problems (seed 0).
B. Ablation ladder at 25% (C32), AIME-25 problems 0-14: each step / the previous one, paired by problem.
C. Teacher-forced parity (llama-ec-bench dumps): top-1 agreement with stock llama.cpp and NLL.
D. llama.cpp on Qwen3 at 25% / 43.75% vs the law's prediction written on the machine before the runs.

    python scripts/run1_stats.py --dir /home/claude/gpu-branch/results/082_review_run1@vast --out prereg/run1_082.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from samehost_stats import boot_ratio  # noqa: E402

LONG = [  # model, budget, ours label, FreeToken label, FreeToken no-overlap label
    ("gpt-oss-120b", "11%", "g_long_ours_C14", "g_long_ft_hybrid_r0.111", None),
    ("gpt-oss-120b", "25%", "g_long_ours_C32", "g_long_ft_hybrid_r0.25", None),
    ("gpt-oss-120b", "40%", "g_long_ours_C51", "g_long_ft_offload_r0.40", "g_long_ftnoov_offload_r0.40"),
    ("Qwen3-30B-A3B", "12.5%", "q_long_ours_C16", "q_long_ft_hybrid_r0.125", None),
    ("Qwen3-30B-A3B", "25%", "q_long_ours_C32", "q_long_ft_hybrid_r0.25", None),
    ("Qwen3-30B-A3B", "43.75%", "q_long_ours_C56", "q_long_ft_offload_r0.4375", "q_long_ftnoov_offload_r0.4375"),
]
LADDER = {
    "g": ["g_abl_0_stock_n27", "g_abl_1_static", "g_abl_2_lru", "g_abl_3_dfa", "g_abl_4_mailbox", "g_abl_5_mapsgpu",
          "g_abl_6_gpusample", "g_abl_7_fetchfixed", "g_abl_8_fetchlaw"],
    "q": ["q_stock_n36", "q_abl_1_static", "q_abl_2_lru", "q_abl_3_dfa", "q_abl_4_mailbox", "q_abl_5_mapsgpu",
          "q_abl_6_gpusample", "q_abl_7_fetchfixed", "q_abl_8_fetchlaw"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/claude/gpu-branch/results/082_review_run1@vast")
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(os.path.join(a.dir, "bs1.jsonl")) if line.strip()]
    by = {}
    for r in rows:
        by.setdefault(r["label"], {})[r["problem"]] = r
    out = dict(long=[], ladder={}, parity={}, llama={}, predictions={})

    def per(lab, key, probs):
        v = []
        for p in probs:
            r = by[lab][p]
            v.append(r["decode_tok_s"] if key == "all" else (r.get("windows") or {}).get(key))
        return np.array(v, dtype=float)

    print("A. steady state, held-out MATH-500 prompts, 2,048 tokens (tok/s: all / tokens 1-256 / tokens 257-2048)")
    p1, p2, p3 = [], [], None
    for model, pct, o, f, fno in LONG:
        if o not in by or f not in by:
            print(f"  {model} {pct}: missing")
            continue
        P = sorted(set(by[o]) & set(by[f]))
        row = dict(model=model, budget=pct, n=len(P))
        for name, lab in (("ours", o), ("freetoken", f)) + ((("freetoken_no_overlap", fno),) if fno and fno in by else ()):
            PP = [p for p in P if p in by[lab]]
            row[name] = {k: float(np.nanmean(per(lab, k, PP))) for k in ("all", "1-256", "257-end")}
        for k in ("1-256", "257-end", "all"):
            x, y = per(o, k, P), per(f, k, P)
            ok = ~(np.isnan(x) | np.isnan(y))
            row[f"ours_over_ft_{k}"] = boot_ratio(x[ok], y[ok])
        row["gain_ours"] = row["ours"]["257-end"] / row["ours"]["1-256"]
        row["gain_ft"] = row["freetoken"]["257-end"] / row["freetoken"]["1-256"]
        out["long"].append(row)
        m, lo, hi = row["ours_over_ft_257-end"]
        m1 = row["ours_over_ft_1-256"][0]
        print(f"  {model:14s} {pct:6s} ours {row['ours']['all']:6.1f} / {row['ours']['1-256']:6.1f} / {row['ours']['257-end']:6.1f}"
              f"   FreeToken {row['freetoken']['all']:6.1f} / {row['freetoken']['1-256']:6.1f} / {row['freetoken']['257-end']:6.1f}"
              f"   ours/FT first-256 {m1:.3f}, steady {m:.3f} [{lo:.3f}, {hi:.3f}]; gain 257+/1-256 ours {row['gain_ours']:.3f} FT {row['gain_ft']:.3f}")
        if "freetoken_no_overlap" in row:
            r = row["freetoken_no_overlap"]
            print(f"      FreeToken with prefill overlap off: {r['all']:6.1f} / {r['1-256']:6.1f} / {r['257-end']:6.1f}")
        if model == "gpt-oss-120b":
            p1.append(lo > 1.0)
        p2.append(row["gain_ft"] > row["gain_ours"])
        if pct == "43.75%":
            p3 = m < 1.0
    print("B. ablation ladder at 25% (C32), AIME-25 problems 0-14, 256 tokens")
    p4 = []
    for tag, steps in LADDER.items():
        have = [s for s in steps if s in by]
        P = sorted(set.intersection(*[set(by[s]) & set(range(15)) for s in have])) if have else []
        prev = None
        out["ladder"][tag] = []
        for s in have:
            x = per(s, "all", P)
            d = dict(step=s, tok_s=float(x.mean()))
            if prev is not None:
                d["over_prev"] = boot_ratio(x, per(prev, "all", P))
                p4.append(d["over_prev"][0] >= 0.99)
            d["over_stock"] = boot_ratio(x, per(have[0], "all", P))
            out["ladder"][tag].append(d)
            op = d.get("over_prev", (float("nan"),) * 3)
            print(f"  {s:22s} {x.mean():7.2f} tok/s   vs previous {op[0]:.3f} [{op[1]:.3f}, {op[2]:.3f}]   vs stock {d['over_stock'][0]:.2f}x")
            prev = s
        if have and have[-1].endswith("fetchlaw"):
            p4.append(out["ladder"][tag][-1]["over_stock"][0] >= 2.0)
    print("C. parity (teacher-forced, 12 x 128 steps)")
    p5 = []
    for tag, (ref, alt) in (("g", ("27", "22")), ("q", ("36", "27"))):
        base = os.path.join(a.dir, f"par_{tag}_n{ref}.bin")
        if not os.path.exists(base):
            print(f"  {tag}: missing")
            continue
        t0 = np.fromfile(base, dtype=np.int32).reshape(-1, 10)[:, 0]

        def nll(p, measured=False):
            v = [json.loads(line) for line in open(p) if line.strip().startswith("{")] if os.path.exists(p) else []
            v = [r for r in v if "nll_sum" in r and (not measured or "stats=" in r.get("config", ""))]
            return sum(r["nll_sum"] for r in v) / max(1, sum(r["nll_n"] for r in v)) if v else None
        n0 = nll(os.path.join(a.dir, f"par_{tag}_n{ref}.jsonl"))
        res = {}
        for name, bp, jp, meas in (("ours", f"par_{tag}_ours.bin.1", f"par_{tag}_ours.jsonl", True),
                                   (f"stock_n{alt}", f"par_{tag}_n{alt}.bin", f"par_{tag}_n{alt}.jsonl", False)):
            p = os.path.join(a.dir, bp)
            if not os.path.exists(p):
                continue
            t1 = np.fromfile(p, dtype=np.int32).reshape(-1, 10)[:, 0]
            n1 = nll(os.path.join(a.dir, jp), meas)
            agree = float((t1 == t0).mean()) if len(t1) == len(t0) else None
            dn = (n1 - n0) / n0 if n0 and n1 else None
            res[name] = dict(top1_agree=agree, nll=n1, nll_ref=n0, dnll=dn, steps=int(len(t1)))
            print(f"  {tag}: {name:10s} vs stock n{ref}: top-1 agreement {agree}, NLL {n1} vs {n0} ({100 * dn:+.2f}%)" if dn is not None
                  else f"  {tag}: {name}: top-1 {agree}")
        out["parity"][tag] = res
        if "ours" in res and res["ours"]["top1_agree"] is not None and res["ours"]["dnll"] is not None:
            p5.append(res["ours"]["top1_agree"] >= 0.98 and abs(res["ours"]["dnll"]) <= 0.01)
    print("D. llama.cpp on Qwen3 vs the law's prediction")
    p6 = []
    lp = os.path.join(a.dir, "law_llama_qwen3.json")
    if os.path.exists(lp):
        pred = json.load(open(lp))["predicted_tok_s"]
        for lab, key in (("q_stock_n36", "llama_n36"), ("q_stock_n27", "llama_n27")):
            if lab in by:
                m = float(np.mean([r["decode_tok_s"] for r in by[lab].values()]))
                e = pred[key] / m - 1
                out["llama"][lab] = dict(measured=m, predicted=pred[key], err=e)
                p6.append(abs(e) <= 0.05)
                print(f"  {lab}: measured {m:.2f}, predicted {pred[key]:.2f} ({100 * e:+.1f}%)")
    P = out["predictions"]
    P["1 steady state: ours leads FreeToken on gpt-oss at 11/25/40% (CI > 1)"] = all(p1) if len(p1) == 3 else None
    P["2 FreeToken's 257+/1-256 gain exceeds ours at every budget"] = all(p2) if len(p2) == 6 else None
    P["3 Qwen3 43.75% steady state: FreeToken above ours"] = p3
    P["4 every ablation step >= -1%, ours >= 2x stock at 25% on both models"] = all(p4) if len(p4) == 18 else None
    P["5 parity: >= 98% top-1 agreement, |dNLL| <= 1%, both models"] = all(p5) if len(p5) == 2 else None
    P["6 llama.cpp Qwen3 25%/43.75% within 5% of the law"] = all(p6) if len(p6) == 2 else None
    for k, v in P.items():
        print(f"prediction {k}: {'HELD' if v else 'FAILED' if v is not None else 'n/a (missing data)'}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
