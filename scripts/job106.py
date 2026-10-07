"""Job 106: the law with its overlap term on both models at both budgets; online admission rules on the deployed path
(dk: the decayed count with a larger admission margin; lrn: the learned reuse predictor with a margin); launch-level
intervals on the slow-link machines. Scores the predictions in the header of jobs/106_onlineadmit@vast.sh by machine and
writes prereg/job106.json, prereg/scorecard_106.json, paper/wsg_job106.tex and paper/tab_job106.tex.

    python scripts/job106.py
"""
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from scripts.panel_099 import cell_data, host_info, _status  # noqa: E402

RES = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
S = {"g": 13253760, "q": 9437184}
CELLS = (("g", 14, 3), ("g", 32, 2), ("q", 16, 1), ("q", 32, 1))
LAB = {("g", 14): "gpt-oss 11%", ("g", 32): "gpt-oss 25%", ("q", 16): "Qwen3 12.5%", ("q", 32): "Qwen3 25%"}
SHORT = {("g", 14): "g11", ("g", 32): "g25", ("q", 16): "q12", ("q", 32): "q25"}
PROF = {("g", 14): ("g_prof.json", "G14"), ("g", 32): ("g_prof.json", "G32"), ("q", 16): ("q_prof.json", "Q16"), ("q", 32): ("q_prof.json", "Q32")}
NAMES = {"106a": "Threadripper 9960X again", "106b": "Pf again (285K)", "106c": "EPYC 7302", "106d": "Xeon 8347C", "106e": "Pe again (9950X)"}
RNG = np.random.default_rng(106)


def law(G, M, A, B, S_, overlap=True):
    """T = G + (M + A (1 - G/T)) S / B_host, solved by fixed-point iteration (overlap=False: the plain law)"""
    tm = lambda r: r * S_ / (B * 1e9) * 1e3  # noqa: E731
    T = G + tm(M + A)
    if overlap:
        for _ in range(100):
            T = G + tm(M + A * (1 - G / T))
    return T


def boot_rounds(rounds, cfg, nb=4000):
    """launch-level interval of base/cfg: resample rounds, then problems within each resampled round (paired)"""
    rs = [r for r in rounds if cfg in r["arr"] and "base" in r["arr"]]
    if not rs:
        return None
    pt = np.mean([r["arr"]["base"].mean() for r in rs]) / np.mean([r["arr"][cfg].mean() for r in rs])
    bs = []
    for _ in range(nb):
        pick = RNG.integers(0, len(rs), len(rs))
        b = x = 0.0
        for i in pick:
            n = len(rs[i]["arr"]["base"])
            j = RNG.integers(0, n, n)
            b += rs[i]["arr"]["base"][j].mean(); x += rs[i]["arr"][cfg][j].mean()
        bs.append(b / x)
    return [float(pt), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def load(d):
    h = dict(dir=os.path.basename(d), job=os.path.basename(d)[:4], **host_info(d), cells={})
    h["ratio"] = h["B_p"] / h["B_c"]
    for (tag, C, nr) in CELLS:
        rounds = []
        for rd in range(1, nr + 1):
            cd = cell_data(d, tag, f"{C}_r{rd}")
            if not cd or "base" not in cd["arr"]:
                continue
            ct = {}
            for k in cd["arr"]:
                f = f"{d}/st_{tag}_C{C}_r{rd}_{k}.json"
                if os.path.exists(f):
                    s = json.load(open(f)); n = max(1, s["steps"])
                    ct[k] = dict(misses=s["misses"] / n, admits=s["admits"] / n, fetches=s["fetches"] / n,
                                 learned=s.get("learned", 0), learned_us=s.get("learned_us_per_step", 0.0))
            rounds.append(dict(round=rd, arr=cd["arr"], counters=ct))
        if not rounds:
            continue
        pf, key = PROF[(tag, C)]
        G = None
        if os.path.exists(f"{d}/{pf}"):
            G = json.load(open(f"{d}/{pf}")).get(key, {}).get("G_prof_ms")
        cfgs = sorted(set.intersection(*(set(r["arr"]) for r in rounds)))
        ms = {k: float(np.mean([r["arr"][k].mean() for r in rounds])) for k in cfgs}
        cnt = {k: {f: float(np.mean([r["counters"][k][f] for r in rounds if k in r["counters"]])) for f in ("misses", "admits", "fetches", "learned_us")}
               for k in cfgs if all(k in r["counters"] for r in rounds)}
        speed = {k: boot_rounds(rounds, k) for k in cfgs if k != "base"}
        per_round = {k: [float(r["arr"]["base"].mean() / r["arr"][k].mean()) for r in rounds] for k in cfgs if k != "base"}
        base_rounds = [float(r["arr"]["base"].mean()) for r in rounds]
        c = dict(tag=tag, C=C, rounds=len(rounds), G=G, ms=ms, counters=cnt, speed=speed, per_round=per_round, base_rounds=base_rounds)
        if G is not None:
            pred = {}
            for k, ct in cnt.items():
                extra = ct["learned_us"] / 1e3 if k == "lrn" else 0.0
                pred[k] = dict(olap=law(G, ct["misses"], ct["admits"], h["b_host"], S[tag]) + extra,
                               plain=law(G, ct["misses"], ct["admits"], h["b_host"], S[tag], False) + extra)
            c["pred"] = pred
            c["err"] = {k: {f: pred[k][f] / ms[k] - 1 for f in ("olap", "plain")} for k in pred}
        h["cells"][f"{tag}{C}"] = c
    return h


def main():
    hosts = {}
    for d in sorted(glob.glob(f"{RES}/106?_onlineadmit@vast")):
        if not os.path.exists(f"{d}/concur.txt") or not glob.glob(f"{d}/ec_g_C14_r*.jsonl"):
            continue
        hosts[os.path.basename(d)[:4]] = load(d)
    clauses = []

    def add(cid, short, typ, meas, ci, ok, thr, host):
        stt, why = _status(meas, ci, ok)
        clauses.append(dict(id=cid, short=short, type=typ, measured=None if meas is None else round(float(meas), 4),
                            ci=None if ci is None else [round(float(x), 4) for x in ci], threshold=thr, status=stt, why=why, host=host))
    errs_olap, errs_plain_g25, errs_olap_g25 = [], [], []
    dk_pool = []
    for job, h in hosts.items():
        nm = NAMES.get(job, job)
        for key, c in h["cells"].items():
            tag, C = c["tag"], c["C"]; g = SHORT[(tag, C)]
            # 1. G_prof
            lo, hi = (4.0, 5.0) if tag == "g" else (4.5, 6.5)
            add(f"{job}-P1-{g}", f"profiled GPU compute {lo}-{hi} ms ({nm}, {g})", "band", c["G"], None, lambda v, lo=lo, hi=hi: lo <= v <= hi, f"{lo} to {hi}", nm)
            # 2. the law with its overlap term, on base
            if "err" in c and "base" in c["err"]:
                e = c["err"]["base"]["olap"]; errs_olap.append(e)
                add(f"{job}-P2-{g}", f"overlap law within 6% of base ({nm}, {g})", "band", abs(e), None, lambda v: v <= 0.06, "<= 0.06", nm)
                if (tag, C) == ("g", 32):
                    errs_olap_g25.append(abs(e)); errs_plain_g25.append(abs(c["err"]["base"]["plain"]))
            ct, sp = c["counters"], c["speed"]
            rd = lambda k: ct[k]["misses"] + ct[k]["admits"]  # noqa: E731
            if tag == "g":
                # 3. dk
                if "dk" in ct and "base" in ct:
                    add(f"{job}-P3-reads-{g}", f"dk reads fewer host experts than base ({nm}, {g})", "sign", rd("dk") / rd("base"), None, lambda v: v < 1, "< 1", nm)
                if "dk" in sp and sp["dk"]:
                    add(f"{job}-P3-speed-{g}", f"dk/base >= 1.01 ({nm}, {g})", "band", sp["dk"][0], sp["dk"][1:], lambda v: v >= 1.01, ">= 1.01", nm)
                    dk_pool.append(sp["dk"][0])
                # 4. lrn
                if "lrn" in ct and "dk" in ct:
                    add(f"{job}-P4-reads-{g}", f"lrn reads at most dk's ({nm}, {g})", "sign", rd("lrn") / rd("dk"), None, lambda v: v <= 1, "<= 1", nm)
                if "lrn" in sp and sp["lrn"]:
                    add(f"{job}-P4-speed-{g}", f"lrn/base >= 1.00 ({nm}, {g})", "sign", sp["lrn"][0], sp["lrn"][1:], lambda v: v >= 1.0, ">= 1.00", nm)
                # 5. the law's ratio prediction
                for k in ("dk", "lrn"):
                    if "pred" in c and k in c["pred"] and "base" in c["pred"] and k in sp and sp[k]:
                        pr = c["pred"]["base"]["olap"] / c["pred"][k]["olap"]
                        c.setdefault("ratio_pred", {})[k] = pr
                        add(f"{job}-P5-{k}-{g}", f"law-predicted {k}/base within 0.03 ({nm}, {g})", "band", abs(pr - sp[k][0]), None, lambda v: v <= 0.03, "<= 0.03", nm)
            else:
                if "dk" in sp and sp["dk"]:
                    add(f"{job}-P3-speed-{g}", f"dk/base >= 1.00 ({nm}, {g})", "sign", sp["dk"][0], sp["dk"][1:], lambda v: v >= 1.0, ">= 1.00", nm)
            if "lrn" in ct:
                lim = 400 if tag == "g" else 700
                add(f"{job}-P4-host-{g}", f"learned host time <= {lim} us per step ({nm}, {g})", "band", ct["lrn"]["learned_us"], None, lambda v, lim=lim: v <= lim, f"<= {lim}", nm)
            # 6, 7: MIN's sets at gpt-oss 11%
            if (tag, C) == ("g", 14):
                if h["ratio"] < 0.4:
                    for i, (fr, fpr) in enumerate(zip(c["per_round"].get("fetch", []), c["per_round"].get("fetchplan", [])), 1):
                        add(f"{job}-P6-fetch-r{i}", f"greedy MIN in the step loses in round {i} ({nm})", "sign", fr, None, lambda v: v < 1, "< 1", nm)
                        add(f"{job}-P6-plan-r{i}", f"fewest-admission set gains in round {i} ({nm})", "sign", fpr, None, lambda v: v > 1, "> 1", nm)
                    if sp.get("fetchplan"):
                        add(f"{job}-P6-launch", f"launch-level interval of fetchplan/base above 1 ({nm})", "sign", sp["fetchplan"][0], sp["fetchplan"][1:],
                            lambda v: v > 1, "> 1", nm)
                if sp.get("fetchplan"):
                    add(f"{job}-P7-speed", f"fetchplan/base > 1 ({nm})", "sign", sp["fetchplan"][0], sp["fetchplan"][1:], lambda v: v > 1, "> 1", nm)
                if "fetchplan" in ct and "fetch" in ct:
                    add(f"{job}-P7-copies", f"fetchplan copies <= 0.65x fetch's ({nm})", "band", ct["fetchplan"]["fetches"] / ct["fetch"]["fetches"], None,
                        lambda v: v <= 0.65, "<= 0.65", nm)
            # 8. base between rounds
            if len(c["base_rounds"]) >= 2:
                b = c["base_rounds"]
                add(f"{job}-P8-{g}", f"base varies <= 2% between rounds ({nm}, {g})", "band", (max(b) - min(b)) / np.mean(b), None, lambda v: v <= 0.02, "<= 0.02", nm)
    if errs_olap:
        add("106-P2-median", "median |error| of the overlap law at most 4%", "band", float(np.median(np.abs(errs_olap))), None, lambda v: v <= 0.04, "<= 0.04", "pooled")
    if errs_olap_g25:
        add("106-P2-g25", "at gpt-oss 25% the overlap law's median |error| at most half the plain law's", "band",
            float(np.median(errs_olap_g25) / np.median(errs_plain_g25)), None, lambda v: v <= 0.5, "<= 0.5", "pooled")
    if dk_pool:
        add("106-P3-pooled", "pooled median dk/base at gpt-oss >= 1.02", "band", float(np.median(dk_pool)), None, lambda v: v >= 1.02, ">= 1.02", "pooled")
    out = [dict(**{k: v for k, v in h.items() if k != "cells"}, cells=h["cells"]) for h in hosts.values()]
    json.dump(dict(hosts=out), open(P("prereg", "job106.json"), "w"), indent=1, default=float)
    json.dump(dict(job="106", script="jobs/106_onlineadmit@vast.sh", commit="525a9c9", scored_by="scripts/job106.py (machine)", clauses=clauses),
              open(P("prereg", "scorecard_106.json"), "w"), indent=1)
    st = Counter(c["status"] for c in clauses)
    M = dict(jiHosts=str(len(hosts)), jiClauses=str(len(clauses)), jiHeld=str(st.get("held", 0)), jiPoint=str(st.get("held (point)", 0)),
             jiFailed=str(st.get("failed", 0)), jiUntested=str(st.get("untested", 0)))

    def rng(k, vals, fmt="{:.2f}"):
        vals = [v for v in vals if v is not None]
        if vals:
            M[k + "Min"] = fmt.format(min(vals)).replace("-", "$-$"); M[k + "Max"] = fmt.format(max(vals)).replace("-", "$-$")
    # the summaries leave out a host whose deployed cache varied between rounds by more than the registered 2% at
    # gpt-oss 11% (clause P8); its clauses stay scored above
    def var14(h):
        b = h["cells"].get("g14", {}).get("base_rounds", [])
        return (max(b) - min(b)) / np.mean(b) if len(b) >= 2 else 0.0
    unstable = {j: var14(h) for j, h in hosts.items() if var14(h) > 0.02}
    hs = {j: h for j, h in hosts.items() if j not in unstable}
    M["jiStable"] = str(len(hs)); M["jiUnstableN"] = str(len(unstable))
    if unstable:
        M["jiUnstableVar"] = f"{100 * max(unstable.values()):.0f}"
        M["jiUnstableName"] = ", ".join(NAMES.get(j, j) for j in unstable)
    if unstable:
        M["jiUnstableVarMin"] = f"{100 * min(unstable.values()):.0f}"; M["jiUnstableVarMax"] = f"{100 * max(unstable.values()):.0f}"
        sp = [max(hosts[j]["cells"]["g14"]["base_rounds"]) / min(hosts[j]["cells"]["g14"]["base_rounds"]) - 1 for j in unstable]
        M["jiUnstableSpreadMax"] = f"{100 * max(sp):.0f}"; M["jiUnstableSpreadMin"] = f"{100 * min(sp):.0f}"   # max/min - 1, as job 107's check
        ue = [100 * abs(c["err"]["base"]["olap"]) for j in unstable for c in hosts[j]["cells"].values() if "err" in c and "base" in c["err"]]
        if ue:
            M["jiUnstErrMin"] = f"{min(ue):.0f}"; M["jiUnstErrMax"] = f"{max(ue):.0f}"
    # teacher-forced loss per token of the deployed cache, round by round (outputs must match the other hosts')
    def nll(h, key):
        out = []
        tag, C = key[0], key[1:]
        for f in sorted(glob.glob(f"{RES}/{h['dir']}/ec_{tag}_C{C}_r*.jsonl")):
            rows = [json.loads(l) for l in open(f) if l.strip()]
            v = [r["nll_sum"] / r["nll_n"] for r in rows if r["config"].rsplit("stats=", 1)[-1].endswith("_base.json")]
            if v:
                out.append(float(np.mean(v)))
        return out
    ng = {j: nll(h, "g14") for j, h in hosts.items()}
    ref = float(np.median([x for v in ng.values() for x in v]))
    bad = {j: v for j, v in ng.items() if v and max(abs(x / ref - 1) for x in v) > 0.05}
    M["jiNllGood"] = f"{ref:.2f}"
    if bad:
        bv = [x for v in bad.values() for x in v] + [x for j in bad for x in nll(hosts[j], "q16")]
        M["jiNllBadMin"] = f"{min(bv):.2f}"; M["jiNllBadMax"] = f"{max(bv):.2f}"
        M["jiNllBadName"] = ", ".join(NAMES.get(j, j) for j in bad)
    M["jiNllGoodQ"] = f"{np.median([x for j in hosts if j not in bad for x in nll(hosts[j], 'q16')]):.3f}"
    # the pooled law error over every host, as registered
    pm = next((c for c in clauses if c["id"] == "106-P2-median"), None)
    if pm:
        M["jiErrMedAll"] = f"{100 * pm['measured']:.1f}"
    # predicting no change of speed, against the law's ratio prediction (stable hosts, gpt-oss)
    nc = [abs(1 - c["speed"][k][0]) for h in hs.values() for c in h["cells"].values() if c["tag"] == "g" for k in ("dk", "lrn") if c["speed"].get(k)]
    if nc:
        M["jiNoChangeMed"] = f"{np.median(nc):.3f}"; M["jiNoChangeMax"] = f"{max(nc):.3f}"
    # the stable hosts against their previous launch (same GPU), deployed cache at gpt-oss 11%
    prev = {"106a": ("105b_sumlaw@vast", "14"), "106b": ("105e_sumlaw@vast", "14"), "106e": ("099e_panel@vast", "g11")}
    rd = []
    for j, (pd_, ck) in prev.items():
        if j not in hs or not os.path.exists(f"{RES}/{pd_}"):
            continue
        cd = cell_data(f"{RES}/{pd_}", "g", 14)
        if cd and "base" in cd["arr"]:
            rd.append(100 * (hs[j]["cells"]["g14"]["ms"]["base"] / cd["arr"]["base"].mean() - 1))
    if rd:
        M["jiRelaunchMax"] = f"{max(abs(x) for x in rd):.1f}"
    # alternative forms on the stable hosts' cells: admissions discounted by half, and misses only
    alt = {"half": 0, "miss": 0, "n": 0}
    for h in hs.values():
        for c in h["cells"].values():
            if c.get("G") is None or "base" not in c["counters"]:
                continue
            ct = c["counters"]["base"]; tm = lambda r: r * S[c["tag"]] / (h["b_host"] * 1e9) * 1e3  # noqa: E731
            alt["n"] += 1
            alt["half"] += abs((c["G"] + tm(ct["misses"] + 0.5 * ct["admits"])) / c["ms"]["base"] - 1) <= 0.06
            alt["miss"] += abs((c["G"] + tm(ct["misses"])) / c["ms"]["base"] - 1) <= 0.06
    M["jiAltHalf"] = str(alt["half"]); M["jiAltMiss"] = str(alt["miss"])
    # the fewest-admission set loaded by the CPU (two reads) on the slow-link machines of jobs 104-105, gpt-oss 11% and 25%
    bp = []
    for pj in (P("prereg", "job104.json"), P("prereg", "job105.json")):
        for hh in json.load(open(pj))["hosts"]:
            if hh["link_over_cpu"] < 1 / 3:
                for C in ("14", "32"):
                    c = hh["cells"].get(C)
                    if c and "bypassplan" in c["speed"]:
                        bp.append(c["speed"]["bypassplan"][0])
    rng("jkSlowBypassPlan", bp)
    fail = Counter("unst" if c["id"][:4] in unstable else ("pooled" if c["id"].startswith("106-") else "stable")
                   for c in clauses if c["status"] == "failed")
    M["jiFailedUnst"] = str(fail.get("unst", 0)); M["jiFailedStable"] = str(fail.get("stable", 0)); M["jiFailedPooled"] = str(fail.get("pooled", 0))
    M["jiClausesUnst"] = str(sum(1 for c in clauses if c["id"][:4] in unstable))

    def used_gb(h):   # host memory in use when the job started (free -g at the gate, before any download)
        try:
            for ln in open(f"{RES}/{h['dir']}/free.txt"):
                if ln.startswith("Mem:"):
                    return int(ln.split()[2])
        except OSError:
            return None
    ug = {j: used_gb(h) for j, h in hosts.items()}
    rng("jiUsedUnst", [ug[j] for j in unstable], "{:.0f}"); rng("jiUsedStable", [ug[j] for j in hs], "{:.0f}")
    # the slowest link (EPYC 7302): MIN's sets round by round, whatever its stability
    vs = [h for h in hosts.values() if h["ratio"] < 0.2 and "g14" in h["cells"]]
    if vs:
        v = vs[0]; c = v["cells"]["g14"]
        M["jiVsRatio"] = f"{v['ratio']:.2f}"; M["jiVsLink"] = f"{v['B_p']:.0f}"; M["jiVsName"] = NAMES.get(v["job"], v["job"])
        for k, nm in (("fetchplan", "Plan"), ("fetch", "Fetch")):
            r = c["per_round"].get(k, [])
            if r:
                M[f"jiVs{nm}Min"] = f"{min(r):.2f}"; M[f"jiVs{nm}Max"] = f"{max(r):.2f}"
        if "fetchplan" in c["counters"]:
            M["jiVsPlanCopies"] = f"{c['counters']['fetchplan']['fetches']:.1f}"
            M["jiVsPlanCopyMs"] = f"{c['counters']['fetchplan']['fetches'] * S['g'] / (v['B_p'] * 1e9) * 1e3:.0f}"
            M["jiVsBaseMs"] = f"{c['ms']['base']:.0f}"
    cells = [(j, c) for j, h in hs.items() for c in h["cells"].values()]
    errs_olap = [c["err"]["base"]["olap"] for _, c in cells if "err" in c and "base" in c["err"]]
    dk_pool = [c["speed"]["dk"][0] for _, c in cells if c["tag"] == "g" and c["speed"].get("dk")]
    for (tag, C), nm in (((t, C), SHORT[(t, C)].replace("g11", "GLow").replace("g25", "GMid").replace("q12", "QLow").replace("q25", "QMid")) for t, C, _ in CELLS):
        sel = [c for _, c in cells if (c["tag"], c["C"]) == (tag, C)]
        rng(f"jiG{nm}", [c["G"] for c in sel], "{:.1f}")
        rng(f"jiErr{nm}", [100 * c["err"]["base"]["olap"] for c in sel if "err" in c and "base" in c["err"]], "{:.1f}")
        rng(f"jiErrPlain{nm}", [100 * c["err"]["base"]["plain"] for c in sel if "err" in c and "base" in c["err"]], "{:.1f}")
        for k in ("dk", "lrn"):
            rng(f"ji{k.capitalize()}{nm}", [c["speed"][k][0] for c in sel if c["speed"].get(k)])
            rng(f"ji{k.capitalize()}Reads{nm}", [100 * (1 - (c["counters"][k]["misses"] + c["counters"][k]["admits"]) /
                                                  (c["counters"]["base"]["misses"] + c["counters"]["base"]["admits"])) for c in sel
                                                 if k in c["counters"] and "base" in c["counters"]], "{:.0f}")
    if errs_olap:
        M["jiErrMed"] = f"{100 * np.median(np.abs(errs_olap)):.1f}"; M["jiErrMax"] = f"{100 * np.max(np.abs(errs_olap)):.1f}"
        M["jiCells"] = str(len(errs_olap))
        M["jiWithin"] = str(int(np.sum(np.abs(errs_olap) <= 0.06)))
    if dk_pool:
        M["jiDkPooled"] = f"{np.median(dk_pool):.2f}"
        M["jiDkPctMin"] = f"{max(0, round(100 * (min(dk_pool) - 1))):.0f}"; M["jiDkPctMax"] = f"{round(100 * (max(dk_pool) - 1)):.0f}"
    rng("jiRatio", [h["ratio"] for h in hs.values()])
    slow = [(j, h) for j, h in hs.items() if h["ratio"] < 0.4 and "g14" in h["cells"]]
    M["jiSlowN"] = str(len(slow))
    rng("jiSlowPlan", [h["cells"]["g14"]["speed"]["fetchplan"][0] for _, h in slow if h["cells"]["g14"]["speed"].get("fetchplan")])
    rng("jiSlowPlanLo", [h["cells"]["g14"]["speed"]["fetchplan"][1] for _, h in slow if h["cells"]["g14"]["speed"].get("fetchplan")])
    rng("jiSlowFetch", [h["cells"]["g14"]["speed"]["fetch"][0] for _, h in slow if h["cells"]["g14"]["speed"].get("fetch")])
    rng("jiPlanAll", [h["cells"]["g14"]["speed"]["fetchplan"][0] for h in hs.values() if "g14" in h["cells"] and h["cells"]["g14"]["speed"].get("fetchplan")])
    rng("jiBaseVar", [100 * (max(c["base_rounds"]) - min(c["base_rounds"])) / np.mean(c["base_rounds"]) for _, c in cells if len(c["base_rounds"]) >= 2], "{:.1f}")
    rng("jiLrnUsG", [c["counters"]["lrn"]["learned_us"] for _, c in cells if c["tag"] == "g" and "lrn" in c["counters"]], "{:.0f}")
    rng("jiLrnUsQ", [c["counters"]["lrn"]["learned_us"] for _, c in cells if c["tag"] == "q" and "lrn" in c["counters"]], "{:.0f}")
    # MIN's fewest-admission set over every launch that ran it (jobs 104, 105, 106), at gpt-oss 11%; machines by GPU UUID
    pl = []
    for pj in (P("prereg", "job104.json"), P("prereg", "job105.json")):
        for hh in json.load(open(pj))["hosts"]:
            c = hh["cells"].get("14")
            if c and hh["planned"].get("14") and "fetchplan" in c["speed"] and "fetch" in c["speed"]:
                pl.append((hh["dir"], hh["link_over_cpu"], c["speed"]["fetch"][0], c["speed"]["fetchplan"][0]))
    for j, hh in hs.items():
        c = hh["cells"].get("g14")
        if c and c["speed"].get("fetchplan") and c["speed"].get("fetch"):
            pl.append((hh["dir"], hh["ratio"], c["speed"]["fetch"][0], c["speed"]["fetchplan"][0]))

    def uuid(dn):
        for ln in open(f"{RES}/{dn}/nvidia-smi-q.txt"):
            if "GPU UUID" in ln:
                return ln.split(":", 1)[1].strip()
        return dn
    U = {x[0]: uuid(x[0]) for x in pl}
    M["jkLaunches"] = str(len(pl)); M["jkMachines"] = str(len(set(U.values())))
    rng("jkPlan", [x[3] for x in pl]); rng("jkRatio", [x[1] for x in pl])
    bal = [x for x in pl if x[1] < 0.9]
    rng("jkGainBal", [x[3] - x[2] for x in bal]); M["jkBalN"] = str(len(bal)); M["jkBalMachines"] = str(len({U[x[0]] for x in bal}))
    rng("jkBalRatio", [x[1] for x in bal])
    fast = [x for x in pl if x[1] >= 0.9]
    rng("jkGainFast", [x[3] - x[2] for x in fast])
    slow = [x for x in pl if x[1] < 1 / 3]
    rng("jkSlowFetch", [x[2] for x in slow]); rng("jkSlowPlan", [x[3] for x in slow]); rng("jkSlowRatio", [x[1] for x in slow])
    M["jkSlowLaunches"] = str(len(slow)); M["jkSlowMachines"] = str(len({U[x[0]] for x in slow}))
    # the law's prediction of each online rule's ratio
    pe = [abs(c["ratio_pred"][k] - c["speed"][k][0]) for _, c in cells if "ratio_pred" in c for k in c["ratio_pred"]]
    if pe:
        M["jiRatioPredMax"] = f"{max(pe):.3f}"; M["jiRatioPredMed"] = f"{np.median(pe):.3f}"
    with open(P("paper", "wsg_job106.tex"), "w") as f:
        f.write("% generated by scripts/job106.py from the job 106 hosts (results/106?_onlineadmit@vast)\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    def table(path, label, which, note):
        with open(P("paper", path), "w") as f:
            f.write("% generated by scripts/job106.py\n\\begin{table}[t]\\centering\\footnotesize\n")
            f.write("\\caption{Job 106, registered before the runs" + note + ": the deployed cache's time per token against "
                    "\\cref{eq:sum} (GPU compute profiled on the same host before the timed runs; the run's own counters), and "
                    "the speed relative to the deployed cache of the two online admission rules (\\emph{dk}: the decayed count "
                    "with a larger admission margin; \\emph{lrn}: the learned reuse predictor with a margin; both chosen on "
                    "other text) and of MIN's fewest-admission set copied in the step, with round-level 95\\% intervals "
                    "(rounds, then problems: three rounds at gpt-oss 11\\%, two at 25\\%; Qwen3 ran one round, so its intervals are over "
                    "problems). Hosts sorted by the probe's link-to-CPU ratio.}\\label{" + label + "}\n")
            f.write("\\setlength\\tabcolsep{2.2pt}\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{@{}lrlrrrlll@{}}\\toprule\n")
            f.write(" & Link/ & & $G$ & \\multicolumn{2}{c}{Deployed (ms)} & & & MIN, fewest \\\\\n")
            f.write("Host & CPU & Budget & (ms) & Eq.~\\ref{eq:sum} & meas. & dk & lrn & in the step \\\\\\midrule\n")
            fmt = lambda v: "--" if not v else f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"  # noqa: E731
            for j, h in sorted(which.items(), key=lambda x: x[1]["ratio"]):
                for key in ("g14", "g32", "q16", "q32"):
                    c = h["cells"].get(key)
                    if not c:
                        continue
                    pr = c.get("pred", {}).get("base", {}).get("olap")
                    f.write(f"{NAMES.get(j, j)}{'$^*$' if j in unstable else ''} & {h['ratio']:.2f} & {LAB[(c['tag'], c['C'])].replace('%', chr(92) + '%')} & "
                            f"{(c['G'] or float('nan')):.1f} & {(pr or float('nan')):.1f} & {c['ms']['base']:.1f} & {fmt(c['speed'].get('dk'))} & "
                            f"{fmt(c['speed'].get('lrn'))} & {fmt(c['speed'].get('fetchplan'))} \\\\\n")
            f.write("\\bottomrule\\end{tabular}}\\end{table}\n")
    table("tab_job106.tex", "tab:job106", hs, ", the hosts that ran stably")
    table("tab_job106_all.tex", "tab:job106all", hosts, ", every host ($^*$: deployed cache varied by more than the registered "
          "2\\% between rounds)")
    print(dict(st))
    for c in clauses:
        print(c["status"], c["id"], c["short"], c["measured"], c["ci"])
    print(M)


if __name__ == "__main__":
    main()
