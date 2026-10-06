"""Job 097 (the learned admission order in the engine): its registered predictions (jobs/097_learned@vast.sh on the gpu
branch, commit 26713ad) scored per host from prereg/foresight_097<host>.json (scripts/foresight_stats.py), the
paper's macros (paper/wsg_learned_engine.tex) and the clauses written into prereg/scorecard_clauses.json.

    python scripts/scorecard_097.py
"""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
RES = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")
HOSTS = [("097a", "O4", "the O4 machine (RTX 5090 + Ryzen 9 9950X, Vast offer 52267630)", "097a_learned_9950x@vast"),
         ("097b", "O6", "RTX 5090 + Ryzen 9 9950X3D (Vast offer 48592284)", "097b_learned_9950x3d@vast")]
CELLS = ["gpt-oss 11%", "gpt-oss 25%", "gpt-oss 40%", "Qwen3 12.5%", "Qwen3 25%", "Qwen3 43.75%"]
HB = ["gpt-oss 11%", "gpt-oss 25%", "Qwen3 12.5%", "Qwen3 25%"]
GB = ["gpt-oss 40%", "Qwen3 43.75%"]
REPLAY = dict(zip(CELLS, [5.8, 7.6, 12.1, 12.0, 11.1, 9.0]))   # % fewer reads, learned vs single-read, prereg/learned_offline.json
O4_096 = {"foa": dict(zip(CELLS, [1.020, 1.030, 1.024, 1.015, 1.016, 1.017])), "fetch": dict(zip(CELLS, [1.361, 1.426, 1.335, 1.438, 1.513, 1.375])),
          "both3p": dict(zip(CELLS, [1.495, 1.809, 1.609, 1.479, 1.804, 1.638]))}


def boot_ratio(ra, rb, n=10000, seed=0):
    """paired ratio of mean speeds of two runs over their common problems, 95% percentile interval"""
    common = sorted(set(ra["tok_s"]) & set(rb["tok_s"]))
    a = np.array([ra["tok_s"][q] for q in common]); b = np.array([rb["tok_s"][q] for q in common])
    idx = np.random.default_rng(seed).integers(0, len(a), (n, len(a)))
    m = a[idx].mean(1) / b[idx].mean(1)
    return [float(a.mean() / b.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def reads(r):
    return r["misses_per_token"] + r["admits_per_token"] + r.get("prefetches_per_token", 0.0)


def band(m, lo, hi, ci=None):
    if not lo <= m <= hi:
        return "failed"
    return "held" if ci and lo <= ci[0] and ci[1] <= hi else "held (point)"


def side(m, thr, above=True, ci=None):
    ok = m > thr if above else m < thr
    if not ok:
        return "failed"
    if ci is None:
        return "held (point)"
    return "held" if ((ci[0] > thr) if above else (ci[1] < thr)) else "held (point)"


def main():
    jobs, M, per = [], {}, {}
    for job, hname, hdesc, rdir in HOSTS:
        f = P("prereg", f"foresight_{job}.json")
        if not os.path.exists(f):
            continue
        cells = {c["label"]: c for c in json.load(open(f))["cells"]}
        per[hname] = cells
        cl = []

        def add(cid, pred, short, clause, typ, qty, thr, meas, ci, st, dec=3, note=""):
            cl.append(dict(id=f"097{hname.replace(chr(39), 'p')}-{cid}", short=f"{short} ({hname})", prediction=pred, clause=clause, type=typ,
                           quantity=qty, threshold=thr, source=f"prereg/foresight_{job}.json", measured=None if meas is None else round(meas, dec),
                           ci=[round(x, dec) for x in ci] if ci else None, status=st, paper_log="", note=note, decimals=dec))
        exp = os.path.join(RES, rdir, "learned_export.txt")
        if os.path.exists(exp):
            txt = open(exp).read()
            n = txt.count("MATCH") - txt.count("MISMATCH")
            add("P1", 1, "six model files hash to the replay's", "the six model files written on the host hash to the replay's", "equality", "MATCH lines", "6",
                n, None, "held" if n == 6 else "failed", 0)
        for lab in CELLS:
            c = cells.get(lab)
            if not c or "learned" not in c["runs"]:
                continue
            R = c["runs"]
            t = lab.replace(" ", "").replace("%", "")
            red = 100 * (1 - reads(R["learned"]) / reads(R["foa"]))
            add(f"P2a-{t}", 2, f"learned reads fewer than foa, {lab}", f"learned reads fewer experts per token than foa at {lab}", "sign", "100 (1 - learned / foa reads)", "> 0",
                red, None, side(red, 0.0), 1)
            add(f"P2b-{t}", 2, f"learned's read reduction within 5 points of the replay's, {lab}", f"within 5 points of the replay's {REPLAY[lab]}% at {lab}", "band",
                "reduction - replay (points)", "-5 to +5", red - REPLAY[lab], None, band(red - REPLAY[lab], -5, 5), 1)
            lfr = boot_ratio(R["learned"], R["foa"])
            R["learned"]["ratio_to_foa"] = lfr
            lf, ci = lfr[0], lfr[1:]
            if lab in HB:
                add(f"P3-{t}", 3, f"learned 1.01-1.18x foa, {lab}", f"learned runs 1.01-1.18x foa at {lab}", "band", "learned / foa speed", "1.01 to 1.18", lf, ci,
                    band(lf, 1.01, 1.18, ci))
                g = R["learned"]["ratio_to_base"]
                fg = R["fetch"]["ratio_to_base"][0]
                share = (g[0] - 1) / (fg - 1)
                add(f"P5a-{t}", 5, f"learned faster than base, {lab}", f"learned runs faster than the online policy at {lab}", "sign", "learned / base speed", "> 1", g[0], g[1:],
                    side(g[0], 1.0, ci=g[1:]))
                add(f"P5b-{t}", 5, f"learned recovers 10-45% of fetch's gain, {lab}", f"(learned/base - 1) / (fetch/base - 1) in 10-45% at {lab}", "band", "share of the fetch oracle's gain",
                    "0.10 to 0.45", share, None, band(share, 0.10, 0.45))
            else:
                add(f"P4-{t}", 4, f"learned at least 0.99x foa, {lab}", f"learned / foa >= 0.99 at {lab}", "threshold", "learned / foa speed", ">= 0.99", lf, ci,
                    ("held" if ci[0] >= 0.99 else "held (point)") if lf >= 0.99 else "failed")
            us = None
            stf = os.path.join(RES, rdir, f"{R['learned']['label']}.json")
            if os.path.exists(stf):
                us = json.load(open(stf)).get("learned_us_per_step")
                R["learned"]["learned_us_per_step"] = us
            if us is not None:
                lim = 300 if lab.startswith("gpt") else 600
                add(f"P6-{t}", 6, f"learned host time at most {lim} us per token, {lab}", f"learned_us_per_step <= {lim} at {lab}", "threshold", "us per token", f"<= {lim}", us, None,
                    "held (point)" if us <= lim else "failed", 0)
            if hname == "O4":
                for k, tol in (("foa", 0.02), ("fetch", 0.05), ("both3p", 0.08)):
                    d = R[k]["ratio_to_base"][0] - O4_096[k][lab]
                    add(f"P7-{k}-{t}", 7, f"{k}/base within {tol} of job 096 on the same machine, {lab}", f"{k} / base within {tol} of 096's {O4_096[k][lab]} at {lab}", "band",
                        f"{k}/base (097) - (096)", f"-{tol} to +{tol}", d, None, band(d, -tol, tol))
        jobs.append(dict(job=f"097{hname.replace(chr(39), 'p')}", era="088-098", script="jobs/097_learned@vast.sh", commit="26713ad", host=hdesc,
                         outcome_note="prereg/learned_outcome_097.md", clauses=cl))
        json.dump(json.load(open(f)) | {"cells": list(cells.values())}, open(f, "w"), indent=1)
    if not jobs:
        print("no 097 statistics yet"); return
    p = P("prereg", "scorecard_clauses.json")
    d = json.load(open(p))
    d["jobs"] = [j for j in d["jobs"] if not str(j.get("job", "")).startswith("097")] + jobs
    for j in jobs:
        if j["job"] not in d["meta"]["eras"]["088-098"]:
            d["meta"]["eras"]["088-098"].append(j["job"])
    json.dump(d, open(p, "w"), indent=1, ensure_ascii=False)
    # macros
    def rng(name, vals, fmt="{:.2f}"):
        M[name + "Min"] = fmt.format(min(vals)); M[name + "Max"] = fmt.format(max(vals))
    allc = [(h, c) for h, cells in per.items() for c in cells.values() if "learned" in c["runs"]]
    hb = [(h, c) for h, c in allc if c["label"] in HB]
    if hb:
        rng("lreOverFoa", [c["runs"]["learned"]["ratio_to_foa"][0] for h, c in hb])
        M["lreOverFoaLoMin"] = f"{min(c['runs']['learned']['ratio_to_foa'][1] for h, c in hb):.3f}"
        rng("lreOverBase", [c["runs"]["learned"]["ratio_to_base"][0] for h, c in hb])
        rng("lreShare", [100 * (c["runs"]["learned"]["ratio_to_base"][0] - 1) / (c["runs"]["fetch"]["ratio_to_base"][0] - 1) for h, c in hb], "{:.0f}")
        rng("lreFewer", [100 * (1 - reads(c["runs"]["learned"]) / reads(c["runs"]["foa"])) for h, c in hb], "{:.0f}")
        rng("lreFewerBase", [100 * (1 - reads(c["runs"]["learned"]) / reads(c["runs"]["base"])) for h, c in hb], "{:.0f}")
        rng("lreShareBest", [100 * (c["runs"]["learned"]["ratio_to_base"][0] - 1) / (c["runs"]["both3p"]["ratio_to_base"][0] - 1) for h, c in hb], "{:.0f}")
        rng("lreFetchOverBase", [c["runs"]["fetch"]["ratio_to_base"][0] for h, c in hb])
    for hn, tag in (("O4", "Four"), ("O6", "Six")):
        hh = [(h, c) for h, c in hb if h == hn]
        if hh:
            rng(f"lreOverFoa{tag}", [c["runs"]["learned"]["ratio_to_foa"][0] for h, c in hh])
            rng(f"lreOverBase{tag}", [c["runs"]["learned"]["ratio_to_base"][0] for h, c in hh])
            rng(f"lreFoaOverBase{tag}", [c["runs"]["foa"]["ratio_to_base"][0] for h, c in hh])
            rng(f"lreFetchMore{tag}", [c["runs"]["learned"]["fetches_per_token"] / c["runs"]["foa"]["fetches_per_token"] for h, c in hh])
    hall = [(h, c) for h, c in allc]
    rng("lreFewerAll", [100 * (1 - reads(c["runs"]["learned"]) / reads(c["runs"]["foa"])) for h, c in hall], "{:.0f}")
    gb = [(h, c) for h, c in allc if c["label"] in GB]
    if gb:
        rng("lreOverFoaGpu", [c["runs"]["learned"]["ratio_to_foa"][0] for h, c in gb])
    us = [c["runs"]["learned"].get("learned_us_per_step") for h, c in allc if c["runs"]["learned"].get("learned_us_per_step") is not None]
    if us:
        M["lreUsMax"] = f"{max(us):.0f}"; M["lreUsMin"] = f"{min(us):.0f}"
    M["lreHostsN"] = str(len(per))
    for job, tag in (("097a", "Four"), ("097b", "Six")):
        f = P("prereg", f"foresight_{job}.json")
        if os.path.exists(f):
            hst = json.load(open(f))["host"]
            M[f"lreBest{tag}"] = f"{hst['b_host_max']:.0f}"; M[f"lreLink{tag}"] = f"{hst['B_p']:.0f}"; M[f"lreCpu{tag}"] = f"{hst['B_c']:.0f}"
    with open(P("paper", "wsg_learned_engine.tex"), "w") as f:
        f.write("% generated by scripts/scorecard_097.py from prereg/foresight_097*.json\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    for k in sorted(M):
        print(f"{k:22s} {M[k]}")
    for j in jobs:
        n = len(j["clauses"]); h = sum(c["status"] == "held" for c in j["clauses"]); pt = sum(c["status"] == "held (point)" for c in j["clauses"])
        print(f"{j['job']}: {n} clauses, {h} held, {pt} held (point), {n - h - pt} failed")
        for c in j["clauses"]:
            if c["status"] == "failed":
                print("   FAILED", c["id"], c["short"], c["measured"], c["threshold"])


if __name__ == "__main__":
    main()
