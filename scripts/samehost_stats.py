"""Statistics for the same-machine comparison v2 (job 073): gpt-oss-120b, 30 AIME-25 problems, greedy, 256 tokens,
held-out warm-up then a session, 3 server launches per main configuration.

Per configuration: tok/s per problem (FreeToken's formula, their client code) averaged over launches; the reported
speed is the mean over problems (their summary statistic). Launch-to-launch spread: coefficient of variation of the
per-launch means. Comparisons are paired by problem: the ratio of mean speeds with a 95% percentile interval from
10,000 bootstrap resamples of the 30 problems (seed 0). FreeToken's better backend is picked per budget by its mean
over all launches (stated; the interval does not include that selection).

Scores the four predictions in the job header and writes a JSON with everything.

    python scripts/samehost_stats.py --dir /home/claude/gpu-branch/results/073_samehost_v2@vast --out prereg/samehost_v2.json
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

BUDGETS = [  # (label, llama.cpp label, ours+fetch, ours, FreeToken rate)
    ("11% (C14)", "llama_n32", "oursf_C14", "ours_C14", "0.111"),
    ("25% (C32)", "llama_n27", "oursf_C32", "ours_C32", "0.25"),
    ("40% (C51)", "llama_n22", "oursf_C51", "ours_C51", "0.40"),
]


def load(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    by = defaultdict(lambda: defaultdict(dict))      # label -> problem -> launch -> tok/s
    meta = {}
    for r in rows:
        by[r["label"]][r["problem"]][r["launch"]] = r["decode_tok_s"]
        m = meta.setdefault(r["label"], {"vram": set(), "sha": {}, "rows": 0})
        m["vram"].add(round(r.get("vram_used_gib") or 0.0, 2))
        m["sha"].setdefault(r["launch"], {})[r["problem"]] = r.get("output_sha1_full") or r.get("output_sha1")
        m["rows"] += 1
    return by, meta


def per_problem(by, label, problems):
    return np.array([np.mean(list(by[label][p].values())) for p in problems])


def launch_means(by, label):
    per = defaultdict(list)
    for p, d in by[label].items():
        for L, v in d.items():
            per[L].append(v)
    return {L: float(np.mean(v)) for L, v in sorted(per.items())}


def boot_ratio(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    r = a[idx].mean(1) / b[idx].mean(1)
    return float(a.mean() / b.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results"), "073_samehost_v2@vast"))
    ap.add_argument("--out")
    a = ap.parse_args()
    by, meta = load(os.path.join(a.dir, "bs1.jsonl"))
    out = {"configs": {}, "comparisons": [], "predictions": {}}
    print("configuration               mean tok/s   per-launch means            CV     VRAM GiB   problems")
    for lab in sorted(by):
        lm = launch_means(by, lab)
        allv = [v for d in by[lab].values() for v in d.values()]
        cv = float(np.std(list(lm.values())) / np.mean(list(lm.values()))) if len(lm) > 1 else 0.0
        out["configs"][lab] = dict(mean=float(np.mean(allv)), launch_means=lm, cv=cv, vram=sorted(meta[lab]["vram"]),
                                   problems=len(by[lab]))
        print(f"{lab:26s} {np.mean(allv):9.2f}   " + " ".join(f"{v:7.2f}" for v in lm.values()).ljust(26)
              + f"  {100 * cv:4.1f}%  {sorted(meta[lab]['vram'])}  {len(by[lab])}")
    ref = meta.get("llama_n27", {}).get("sha", {}).get(1, {})
    agree = {}
    for lab, m in meta.items():
        s = m["sha"].get(1) or next(iter(m["sha"].values()))
        common = [p for p in s if p in ref]
        agree[lab] = sum(1 for p in common if s[p] == ref[p]) / max(1, len(common))
    out["same_text_as_llama_n27"] = agree
    p1 = []
    p2 = []
    for name, lc, of, ob, rate in BUDGETS:
        fts = [f"ft_offload_r{rate}", f"ft_hybrid_r{rate}"]
        fts = [f for f in fts if f in by]
        if not fts or of not in by or lc not in by:
            print(f"{name}: missing configurations")
            continue
        best = max(fts, key=lambda f: np.mean([v for d in by[f].values() for v in d.values()]))
        problems = sorted(set(by[of]) & set(by[best]) & set(by[lc]))
        A, Fb, Lc = per_problem(by, of, problems), per_problem(by, best, problems), per_problem(by, lc, problems)
        r_ft = boot_ratio(A, Fb)
        r_lc = boot_ratio(A, Lc)
        r_ftlc = boot_ratio(Fb, Lc)
        row = dict(budget=name, ours_fetch=float(A.mean()), freetoken_best=best, freetoken=float(Fb.mean()),
                   llama=float(Lc.mean()), ours_over_ft=r_ft, ours_over_llama=r_lc, ft_over_llama=r_ftlc,
                   n_problems=len(problems))
        if ob in by:
            B = per_problem(by, ob, [p for p in problems if p in by[ob]])
            row["ours_base"] = float(B.mean())
        out["comparisons"].append(row)
        print(f"{name}: ours+FETCH {A.mean():.1f}, {best} {Fb.mean():.1f}, llama.cpp {Lc.mean():.1f} tok/s (n={len(problems)})")
        print(f"   ours/FreeToken {r_ft[0]:.3f} [{r_ft[1]:.3f}, {r_ft[2]:.3f}]   ours/llama.cpp {r_lc[0]:.2f} [{r_lc[1]:.2f}, {r_lc[2]:.2f}]"
              f"   FreeToken/llama.cpp {r_ftlc[0]:.2f} [{r_ftlc[1]:.2f}, {r_ftlc[2]:.2f}]")
        p1.append((name, r_ft))
        p2.append((name, r_lc))
    # prediction 1
    ok1 = True
    for name, (m, lo, hi) in p1:
        if "C51" in name:
            ok1 &= 0.95 <= m <= 1.05
        else:
            ok1 &= m >= 1.05 and lo > 1.0
    out["predictions"]["1 ours+FETCH ahead at C14/C32 (>=5%, CI>1), within 5% at C51"] = ok1 if p1 else None
    out["predictions"]["2 ours+FETCH >= 1.8x llama.cpp at every budget"] = all(m >= 1.8 for _, (m, _, _) in p2) if p2 else None
    # prediction 3: old protocol vs session, same 5 problems, launch 1
    old = {}
    for olab, lab in (("old_llama_n27", "llama_n27"), ("old_oursf_C32", "oursf_C32"), ("old_ft_offload_r0.25", "ft_offload_r0.25")):
        if olab in by and lab in by:
            ps = sorted(by[olab])
            o = np.mean([by[olab][p][1] for p in ps])
            s = np.mean([by[lab][p][1] for p in ps if 1 in by[lab][p]])
            old[lab] = float(o / s - 1)
            print(f"old protocol / session, problems {ps}: {lab} {100 * (o / s - 1):+.1f}%")
    out["old_vs_session"] = old
    if old:
        out["predictions"]["3 old protocol +5-15% for caches, +-2% for llama.cpp"] = (
            all(0.05 <= old[k] <= 0.15 for k in old if not k.startswith("llama")) and abs(old.get("llama_n27", 0)) <= 0.02)
    ag = {k: v for k, v in agree.items() if not k.startswith("old_")}
    out["predictions"]["4 >= 80% same greedy text as llama.cpp, every system"] = all(v >= 0.8 for v in ag.values()) if ag else None
    print("same greedy text as llama_n27 (launch 1): " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in sorted(ag.items())))
    for k, v in out["predictions"].items():
        print(f"prediction {k}: {'HELD' if v else 'FAILED' if v is not None else 'n/a'}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
