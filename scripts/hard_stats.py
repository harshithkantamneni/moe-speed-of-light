"""Statistics for job 083: gpt-oss-120b long outputs (2,048 tokens) on 10 hard held-out problems (AIME 2022-2024,
problems 11-15 of each exam), ours (law split) vs FreeToken's Table 1 backend at 11 / 25 / 40%.

Per request the client records every streamed event as reasoning or answer text and the output's last 200 characters.
A request "reached an answer" if it streamed answer (content) events, or if its tail contains \\boxed. Ratios of mean
speeds, paired by problem, 95% percentile interval over 10,000 bootstrap resamples (seed 0), over all tokens and over
tokens 257-2,048.

    python scripts/hard_stats.py --out prereg/hard_083.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from samehost_stats import boot_ratio  # noqa: E402

CELLS = [("11%", "g_hard_ours_C14", "g_hard_ft_hybrid_r0.111", 69.9), ("25%", "g_hard_ours_C32", "g_hard_ft_hybrid_r0.25", 109.2),
         ("40%", "g_hard_ours_C51", "g_hard_ft_offload_r0.40", 152.5)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results"), "083_gptoss_hard_long@vast"))
    ap.add_argument("--out")
    a = ap.parse_args()
    by = {}
    for line in open(os.path.join(a.dir, "bs1.jsonl")):
        if line.strip():
            r = json.loads(line)
            by.setdefault(r["label"], {})[r["problem"]] = r

    def answered(r):
        return bool(r.get("n_content_events")) or "boxed" in (r.get("text_tail") or "")

    def v(lab, key, P):
        return np.array([by[lab][p]["decode_tok_s"] if key == "all" else (by[lab][p].get("windows") or {}).get(key, np.nan)
                         for p in P], dtype=float)
    out = dict(rows=[], predictions={})
    p1, p2, p3, p4 = [], [], [], []
    for budget, o, f, t1 in CELLS:
        if o not in by or f not in by:
            print(f"{budget}: missing")
            continue
        P = sorted(set(by[o]) & set(by[f]))
        row = dict(budget=budget, n=len(P))
        for name, lab in (("ours", o), ("freetoken", f)):
            rr = [by[lab][p] for p in P]
            row[name] = dict(all=float(v(lab, "all", P).mean()), w1=float(np.nanmean(v(lab, "1-256", P))),
                             w2=float(np.nanmean(v(lab, "257-end", P))),
                             events_per_token=float(np.mean([r["events"] / r["completion_tokens"] for r in rr])),
                             answered=sum(answered(r) for r in rr),
                             reasoning_events=float(np.mean([r.get("n_reasoning_events", 0) for r in rr])),
                             content_events=float(np.mean([r.get("n_content_events", 0) for r in rr])))
        for key in ("all", "257-end"):
            x, y = v(o, key, P), v(f, key, P)
            ok = ~(np.isnan(x) | np.isnan(y))
            row[f"ours_over_ft_{key}"] = boot_ratio(x[ok], y[ok])
        row["ours_vs_table1"] = row["ours"]["all"] / t1 - 1
        out["rows"].append(row)
        m, lo, hi = row["ours_over_ft_all"]
        m2, lo2, hi2 = row["ours_over_ft_257-end"]
        print(f"{budget}: ours {row['ours']['all']:.1f} (1-256 {row['ours']['w1']:.1f}, 257+ {row['ours']['w2']:.1f}; "
              f"{row['ours_vs_table1']:+.1%} vs Table 1), FreeToken {row['freetoken']['all']:.1f} (1-256 {row['freetoken']['w1']:.1f}, "
              f"257+ {row['freetoken']['w2']:.1f}); ours/FT all {m:.3f} [{lo:.3f}, {hi:.3f}], 257+ {m2:.3f} [{lo2:.3f}, {hi2:.3f}]; "
              f"events/token ours {row['ours']['events_per_token']:.2f} FT {row['freetoken']['events_per_token']:.2f}; "
              f"answered ours {row['ours']['answered']}/{len(P)} FT {row['freetoken']['answered']}/{len(P)}")
        p1.append(len(P) - row["ours"]["answered"] >= 8 and len(P) - row["freetoken"]["answered"] >= 8)
        p2.append(abs(row["ours_vs_table1"]) <= 0.15)
        both_one = row["ours"]["events_per_token"] > 0.9 and row["freetoken"]["events_per_token"] > 0.9
        p3.append(lo > 1 and (hi2 > 0 and lo2 > 1 if both_one else True))
        if budget in ("11%", "25%"):
            p4.append(m >= 1.15)
    P = out["predictions"]
    P["1 both still reasoning at 2,048 tokens on >= 8 of 10 problems"] = all(p1) if len(p1) == 3 else None
    P["2 ours within 15% of its Table 1 rate"] = all(p2) if len(p2) == 3 else None
    P["3 ours leads FreeToken at 11/25/40% (all tokens; and 257+ where both stream ~1 event per token)"] = all(p3) if len(p3) == 3 else None
    P["4 lead >= 15% at 11% and 25%"] = all(p4) if len(p4) == 2 else None
    for k, x in P.items():
        print(f"prediction {k}: {'HELD' if x else 'FAILED' if x is not None else 'n/a'}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
