"""Statistics for job 080: the per-host FETCH table (Qwen3-30B-A3B BF16, RTX 5090 + 9950X3D).

Launch 1 at 43.75% (C56) sweeps the tables; launch 2 reruns the launch-1 best table (and the current and law tables)
and FreeToken's launch-1 better backend, so the confirmation is not biased by the selection. 25% (C32) is launch 1
only. Ratios of mean speeds, paired by problem, 95% percentile interval over 10,000 bootstrap resamples (seed 0).

Also checks the per-layer model behind the law table against the sweep: for each run, the modelled MoE time per token
is 48 layers x sum_n h_n T(n, f(n)) with that run's own per-layer miss histogram h (so cache effects of the table are
in h), and the measured token time is regressed on it (slope ~1 expected if the model's costs are right).

    python scripts/split_stats.py --dir /home/claude/gpu-branch/results/080_fetch_split@vast --out prereg/split_080.json
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from samehost_stats import boot_ratio  # noqa: E402

S = 9437184.0


def load(path):
    by = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            by.setdefault((r["label"], r["launch"]), {})[r["problem"]] = r["decode_tok_s"]
    return by


def ratio(by, a, b):
    P = sorted(set(by[a]) & set(by[b]))
    return boot_ratio(np.array([by[a][p] for p in P]), np.array([by[b][p] for p in P])), len(P)


def model_T(n, f, bc, bp, bb, g=48.0, lc=20.0):
    sp, sc, sb = S / bp / 1e3, S / bc / 1e3, S / bb / 1e3
    t = f * sp + g
    if n - f > 0:
        t = max(t, lc + (n - f) * sc)
    if 0 < f < n:
        t = max(t, n * sb)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/claude/gpu-branch/results/080_fetch_split@vast")
    ap.add_argument("--out")
    a = ap.parse_args()
    by = load(os.path.join(a.dir, "bs1.jsonl"))
    mean = {k: float(np.mean(list(v.values()))) for k, v in by.items()}
    law = json.load(open(os.path.join(a.dir, "fetch_table_law.json")))
    tables = dict(re.findall(r"(current|law|law_f1|lite|all) ([\d,]+)", open(os.path.join(a.dir, "tables.txt")).read()))
    print(f"host: B_c {law['B_c']} B_p {law['B_p']} B_both {law['B_both']} GB/s; law table {law['table']}")
    out = dict(host=law, tables=tables, means={f"{k[0]} L{k[1]}": v for k, v in mean.items()}, ratios={}, predictions={})
    for (lab, L), v in sorted(mean.items(), key=lambda x: (x[0][1], x[0][0])):
        print(f"  {lab:24s} L{L} {v:7.2f} tok/s (n={len(by[(lab, L)])})")

    def R(x, y, key):
        if x in by and y in by:
            (m, lo, hi), n = ratio(by, x, y)
            out["ratios"][key] = (m, lo, hi)
            print(f"  {key:44s} {m:.3f} [{lo:.3f}, {hi:.3f}] (n={n})")
            return m, lo, hi
        return None
    print("43.75%, launch 1:")
    sweep = [k for k in by if k[1] == 1 and k[0].startswith("ours_C56_")]
    best = max(sweep, key=lambda k: mean[k])
    for k in sorted(sweep, key=lambda k: -mean[k]):
        R(k, ("ours_C56_cur", 1), f"{k[0]} / cur (L1)")
    r1 = R(("ours_C56_law", 1), ("ours_C56_cur", 1), "law / cur (L1)")
    r2 = R(("ours_C56_law", 1), best, f"law / best {best[0]} (L1)")
    fts1 = [k for k in by if k[1] == 1 and k[0].endswith("r0.4375")]
    ftb = max(fts1, key=lambda k: mean[k]) if fts1 else None
    if ftb:
        R(best, ftb, f"best / {ftb[0]} (L1)")
    print("43.75%, launch 2 (confirmation):")
    ft2 = [k for k in by if k[1] == 2 and k[0].startswith("ft_")]
    r3 = None
    if ft2:
        b2 = (best[0], 2)
        r3 = R(b2, ft2[0], f"{best[0]} / {ft2[0][0]} (L2)")
        for t in ("cur", "law"):
            if (f"ours_C56_{t}", 2) in by and f"ours_C56_{t}" != best[0]:
                R((f"ours_C56_{t}", 2), ft2[0], f"ours_C56_{t} / {ft2[0][0]} (L2)")
        if ("ours_C56_law", 2) in by and ("ours_C56_cur", 2) in by:
            R(("ours_C56_law", 2), ("ours_C56_cur", 2), "law / cur (L2)")
    print("25%, launch 1:")
    r4 = R(("ours_C32_law", 1), ("ours_C32_cur", 1), "C32 law / cur (L1)")
    o32 = [k for k in by if k[0].startswith("ours_C32_")]
    f32 = [k for k in by if k[0].endswith("r0.25")]
    if o32 and f32:
        ob, fb = max(o32, key=lambda k: mean[k]), max(f32, key=lambda k: mean[k])
        R(ob, fb, f"C32 best {ob[0]} / {fb[0]} (L1)")
    out["predictions"]["1 law beats current by >= 3% at 43.75% (L1, CI lower bound > 1)"] = (
        bool(r1[0] >= 1.03 and r1[1] > 1.0) if r1 else None)
    out["predictions"]["2 law within 2% of the best table (L1)"] = bool(r2[0] >= 0.98) if r2 else None
    out["predictions"]["3 best table leads FreeToken's better backend at 43.75% (L2, CI > 1)"] = (
        bool(r3[1] > 1.0) if r3 else None)
    out["predictions"]["4 law beats current at 25% (L1, CI > 1)"] = bool(r4[1] > 1.0) if r4 else None
    # per-layer model vs the sweep
    xs, ys, names = [], [], []
    for f in sorted(os.listdir(a.dir)):
        m = re.fullmatch(r"srv_(ours_C56_\w+?)_L(\d)\.json", f)
        if not m:
            continue
        d = json.load(open(os.path.join(a.dir, f)))
        tab = [int(x) for x in str(d.get("fetch_table") or "").split(",") if x.strip().isdigit()] or [0] * 9
        tab = (tab + [tab[-1]] * 9)[:9]
        h = np.zeros(9)
        for L in d["layers"]:
            h[:len(L["miss_hist"])] += L["miss_hist"]
        h /= h.sum()
        moe = 48 * sum(h[n] * model_T(n, min(n, tab[n]), law["B_c"], law["B_p"], law["B_both"]) for n in range(9)) / 1e3
        key = (m.group(1), int(m.group(2)))
        if key in mean:
            xs.append(moe); ys.append(1e3 / mean[key]); names.append(f"{key[0]} L{key[1]}")
            print(f"  model {key[0]:18s} L{key[1]}: modelled MoE {moe:6.2f} ms, measured token {1e3 / mean[key]:6.2f} ms, "
                  f"misses/token {d['misses'] / d['steps']:.1f}, fetches {d.get('fetches', 0) / d['steps']:.1f}, admits {d['admits'] / d['steps']:.1f}")
    if len(xs) >= 3:
        b, c = np.polyfit(xs, ys, 1)
        r = float(np.corrcoef(xs, ys)[0, 1])
        out["model_fit"] = dict(slope=float(b), intercept_ms=float(c), r=r, points=dict(zip(names, zip(xs, ys))))
        print(f"  measured token = {c:.2f} ms + {b:.2f} x modelled MoE time (r = {r:.3f}, n = {len(xs)})")
    for k, v in out["predictions"].items():
        print(f"prediction {k}: {'HELD' if v else 'FAILED' if v is not None else 'n/a'}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
