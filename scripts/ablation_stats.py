"""Statistics for job 087: the ablation ladder at 25% (C32) with a real static cache, on one machine (listing A).
AIME-25 problems 0-14, 256 tokens, greedy, session. Each step against the one before it, paired by problem; 95%
percentile interval over 10,000 bootstrap resamples (seed 0). Steps 1a (no expert resident) and 1h (static in hindsight)
are reported against stock and against the profiled static cache, outside the ladder.

    python scripts/ablation_stats.py --out prereg/ablation_087.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from samehost_stats import boot_ratio  # noqa: E402

STEPS = ["abl_0_stock", "abl_1_static", "abl_2_lru", "abl_3_dfa", "abl_4_mailbox", "abl_5_mapsgpu", "abl_6_gpusample",
         "abl_7_fetchfixed", "abl_8_fetchlaw"]
EXTRA = ["abl_1a_nocache", "abl_1h_static_hindsight"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results"), "087_ablation_static@vast"))
    ap.add_argument("--out")
    a = ap.parse_args()
    by, hit = {}, {}
    for line in open(os.path.join(a.dir, "bs1.jsonl")):
        if line.strip():
            r = json.loads(line)
            by.setdefault(r["label"], {})[r["problem"]] = r["decode_tok_s"]
    for lab in by:
        p = os.path.join(a.dir, f"srv_{lab}.json")
        if os.path.exists(p):
            hit[lab] = json.load(open(p))["hit_rate"]
    out = dict(ladder={}, extra={}, predictions={})
    P1, P2, P3, P4 = [], [], [], []
    for m in ("g", "q"):
        labs = {s: next(l for l in by if l.startswith(f"{m}_{s}")) for s in STEPS + EXTRA}
        probs = sorted(set.intersection(*[set(by[l]) for l in labs.values()]))
        v = {s: np.array([by[labs[s]][p] for p in probs]) for s in labs}
        lad = []
        for i, s in enumerate(STEPS):
            d = dict(step=s, label=labs[s], tok_s=float(v[s].mean()), hit=hit.get(labs[s]))
            if i:
                d["over_prev"] = boot_ratio(v[s], v[STEPS[i - 1]])
            d["over_stock"] = boot_ratio(v[s], v[STEPS[0]])
            lad.append(d)
        out["ladder"][m] = lad
        ex = {}
        for s in EXTRA:
            ex[s] = dict(label=labs[s], tok_s=float(v[s].mean()), hit=hit.get(labs[s]), over_stock=boot_ratio(v[s], v[STEPS[0]]),
                         over_static=boot_ratio(v[s], v[STEPS[1]]), over_dfa=boot_ratio(v[s], v[STEPS[3]]))
        out["extra"][m] = ex
        print(f"== {m} ({len(probs)} problems)")
        for d in lad:
            op = d.get("over_prev", (float('nan'),) * 3)
            print(f"  {d['step']:18s} {d['tok_s']:7.2f} tok/s  hit {d['hit'] if d['hit'] is not None else '-':>6}  vs previous "
                  f"{op[0]:.3f} [{op[1]:.3f}, {op[2]:.3f}]  vs stock {d['over_stock'][0]:.2f}x")
        for s, d in ex.items():
            print(f"  {s:26s} {d['tok_s']:7.2f} tok/s  hit {d['hit']}  vs stock {d['over_stock'][0]:.3f}  vs profiled static "
                  f"{d['over_static'][0]:.3f}  vs decayed frequency {d['over_dfa'][0]:.3f} [{d['over_dfa'][1]:.3f}, {d['over_dfa'][2]:.3f}]")
        st, dfa = lad[1], lad[3]
        P1.append(st["over_stock"][1] >= 1.3 or st["over_stock"][0] >= 1.3)
        P2.append(dfa["tok_s"] / st["tok_s"] < st["over_stock"][0])
        P3.append(boot_ratio(v[STEPS[3]], v[STEPS[1]])[1] > 1)
        P4.append(all(d["over_prev"][0] >= 0.99 for d in lad[2:]) and lad[-1]["over_stock"][0] >= 2)
    P = out["predictions"]
    P["1 profiled static >= 1.3x stock, both models"] = all(P1)
    P["2 dynamic replacement adds less than placement (step3/step1 < step1/step0), both models"] = all(P2)
    P["3 decayed frequency beats the profiled static cache (CI > 1), both models"] = all(P3)
    P["4 every step after the static cache >= -1% and ours >= 2x stock, both models"] = all(P4)
    for k, x in P.items():
        print(f"prediction {k}: {'HELD' if x else 'FAILED'}  (per model {[P1, P2, P3, P4][int(k[0]) - 1]})")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
