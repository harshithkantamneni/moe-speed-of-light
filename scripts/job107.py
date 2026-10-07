"""Job 107: the time relation with its overlap term registered again on machines never rented before, with validity
gates registered before launch (V0 new GPU and host memory not in use; V1 the deployed cache's loss; V2 the deployed
cache's spread between the gpt-oss 11% rounds); the admission margin (dk) and MIN's fewest-admission set loaded either
way. Scores the predictions in the header of jobs/107_newhosts@vast.sh and writes prereg/job107.json,
prereg/scorecard_107.json, paper/wsg_job107.tex and paper/tab_job107.tex.

    python scripts/job107.py
"""
import glob
import json
import os
import re
import sys
from collections import Counter

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from scripts.panel_099 import host_info, _status  # noqa: E402
from scripts.job106 import law, load as load106, S  # noqa: E402
import scripts.job106 as j106  # noqa: E402

RES = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")
PAT = os.environ.get("MOSL_JOB107_GLOB", "107?_newhosts@vast")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
CELLS = (("g", 14, 2), ("g", 32, 2), ("q", 16, 1), ("q", 32, 1))
LAB = {("g", 14): "gpt-oss 11%", ("g", 32): "gpt-oss 25%", ("q", 16): "Qwen3 12.5%", ("q", 32): "Qwen3 25%"}
SHORT = {("g", 14): "g11", ("g", 32): "g25", ("q", 16): "q12", ("q", 32): "q25"}
REF = {"g": 0.190, "q": 0.0855}
REQUIRED, MEM_GATE, CLASS_CORES = 3, 48, 32   # registered: valid hosts needed, GB in use at the gate; after the fact: the class rule
j106.RNG = np.random.default_rng(107)


def short_cpu(cpu):
    c = re.sub(r"\(R\)|\(TM\)|®|™", "", cpu or "")
    c = re.sub(r"\b(AMD|Intel|Processor|CPU)\b|\d+-Cores?|@.*", "", c)
    c = re.sub(r"\bRyzen Threadripper\b", "Threadripper", c)
    c = re.sub(r"\s+", " ", c).strip()
    if c.startswith("Eng Sample"):
        return "AMD engineering sample"
    return c or "CPU not listed"


def v0(d):
    """the gate's record: (passed, reason, used GB, uuid)"""
    txt = open(f"{d}/v0.txt").read() if os.path.exists(f"{d}/v0.txt") else ""
    m = re.search(r"GPU UUID (\S+); host memory in use (\d+) GB", txt)
    uuid, used = (m.group(1), int(m.group(2))) if m else (None, None)
    if "V0 FAIL" in txt:
        return False, txt.split("V0 FAIL:", 1)[1].split("\n")[0].strip(), used, uuid
    gate = open(f"{d}/gate.txt").read() if os.path.exists(f"{d}/gate.txt") else ""
    m = re.search(r"device read (\d+) GB/s", gate)
    if m and int(m.group(1)) < 1500:
        return False, f"device reads {m.group(1)} GB/s (< 1500)", used, uuid
    m = re.search(r"disk free (\d+) GB", gate)
    if m and int(m.group(1)) < 150:
        return False, f"disk {m.group(1)} GB (< 150)", used, uuid
    if not os.path.exists(f"{d}/concur.txt"):
        return False, "stopped before the probe", used, uuid
    return True, "", used, uuid


def losses(d):
    """the deployed cache's loss per token (mean over problems) in every round: {(tag, C, round): loss}"""
    out = {}
    for f in sorted(glob.glob(f"{d}/ec_*_C*_r*.jsonl")):
        m = re.search(r"ec_([gq])_C(\d+)_r(\d+)\.jsonl", f)
        rows = [json.loads(l) for l in open(f) if l.strip()]
        v = [r["nll_sum"] / r["nll_n"] for r in rows if r["config"].rsplit("stats=", 1)[-1].endswith("_base.json")]
        if v:
            out[(m.group(1), int(m.group(2)), int(m.group(3)))] = float(np.mean(v))
    return out


DAG = "$^{\\dagger}$"
INVALID = {"106c": "wrong outputs", "106d": "rounds 39% apart", "107c": "rounds 3.5% apart", "107e": "rounds 6.0% apart",
           "108b": "rounds 7.0% apart"}


def machine_classes(jobs=r"(09[3-9]|10[0-7])"):
    """every launch of jobs 093-107 that ran the deployed cache at gpt-oss 11% (first round): class (server = more than
    one NUMA node or more than 32 usable physical cores), implied read rate over B_host, the overlap form's error"""
    ra = json.load(open(P("prereg", "reanalysis.json")))
    G0 = ra["sum_law"]["14"]["G"]
    out = []
    for d in sorted(glob.glob(f"{RES}/*@vast")):
        j = os.path.basename(d)
        if not re.match(jobs, j):
            continue
        st = [f for f in (f"{d}/st_g_C14_base.json", f"{d}/st_g_C14_r1_base.json") if os.path.exists(f)]
        if not st or not os.path.exists(f"{d}/concur.txt"):
            continue
        ec = st[0].replace("/st_g_C14", "/ec_g_C14").replace("_base.json", ".jsonl")
        if not os.path.exists(ec):
            continue
        hi = host_info(d)
        s = json.load(open(st[0])); n = max(1, s["steps"]); Mi, A = s["misses"] / n, s["admits"] / n
        T = float(np.mean([r["decode_ms"] / r["n_decode"] for r in map(json.loads, filter(str.strip, open(ec)))
                           if r["config"].rsplit("stats=", 1)[-1].endswith("_base.json")]))
        G = G0
        if os.path.exists(f"{d}/g_prof.json"):
            gp = json.load(open(f"{d}/g_prof.json"))
            G = (gp.get("G14") or gp.get("C14") or {}).get("G_prof_ms", G0)
            if not G or G < 1.0:   # a profile that captured no decode kernels
                G = G0
        B = hi["b_host"]
        t = law(G, Mi, A, B, S["g"])
        eff = (Mi + A * (1 - G / T)) * S["g"] / ((T - G) * 1e-3) / 1e9 / B
        numa = 1
        if os.path.exists(f"{d}/numa.txt"):
            m = re.search(r"available: (\d+) nodes", open(f"{d}/numa.txt").read())
            numa = int(m.group(1)) if m else 1
        uuid = next((ln.split(":", 1)[1].strip() for ln in open(f"{d}/nvidia-smi-q.txt") if "GPU UUID" in ln), j) \
            if os.path.exists(f"{d}/nvidia-smi-q.txt") else j
        out.append(dict(dir=j, cpu=hi["cpu"], cores=hi["cores"], numa=numa, B_host=B, eff=eff, err=t / T - 1, uuid=uuid,
                        cls_why=f"{numa} NUMA nodes, {hi['cores']} cores", valid=j[:4] not in INVALID,
                        **{"class": "server" if numa > 1 or hi["cores"] > CLASS_CORES else "desktop"}))
    return out


def main():
    j106.CELLS = CELLS
    hosts, gated = {}, {}
    for d in sorted(glob.glob(f"{RES}/{PAT}")):
        job = os.path.basename(d)[:4]
        ok, why, used, uuid = v0(d)
        cpu = ""
        if os.path.exists(f"{d}/cpu.txt"):
            m = re.search(r"Model name:\s*(.+)", open(f"{d}/cpu.txt").read())
            cpu = m.group(1).strip() if m else ""
        if not ok:
            gated[job] = dict(dir=os.path.basename(d), reason=why, used_gb=used, uuid=uuid, cpu=cpu)
            continue
        h = load106(d)
        h["name"] = short_cpu(h.get("cpu"))
        h["used_gb"], h["uuid"] = used, uuid
        ls = losses(d)
        h["loss"] = {f"{t}{C}_r{r}": v for (t, C, r), v in ls.items()}
        v1 = [abs(v / REF[t] - 1) for (t, C, r), v in ls.items()]
        h["V1"] = bool(v1) and max(v1) <= 0.02
        b = h["cells"].get("g14", {}).get("base_rounds", [])
        h["V2_spread"] = (max(b) / min(b) - 1) if len(b) >= 2 else None
        h["V2"] = h["V2_spread"] is not None and h["V2_spread"] <= 0.02
        h["valid"] = h["V1"] and h["V2"]
        hosts[job] = h
    valid = {j: h for j, h in hosts.items() if h["valid"]}
    clauses = []

    def add(cid, short, typ, meas, ci, ok, thr, host):
        stt, why = _status(meas, ci, ok)
        clauses.append(dict(id=cid, short=short, type=typ, measured=None if meas is None else round(float(meas), 4),
                            ci=None if ci is None else [round(float(x), 4) for x in ci], threshold=thr, status=stt, why=why, host=host))
    errs_olap, errs_plain, errs_half, errs_miss, dk_all, dk_pred_err, dk_nochange = [], [], [], [], [], [], []
    for job, h in valid.items():
        nm = f"{job} {h['name']}"
        for key, c in h["cells"].items():
            tag, C = c["tag"], c["C"]; g = SHORT[(tag, C)]
            lo, hi = (4.0, 5.0) if tag == "g" else (4.5, 6.5)
            add(f"{job}-P1-{g}", f"profiled GPU compute {lo}-{hi} ms ({nm}, {g})", "band", c["G"], None, lambda v, lo=lo, hi=hi: lo <= v <= hi, f"{lo} to {hi}", nm)
            if "err" in c and "base" in c["err"]:
                e = c["err"]["base"]; errs_olap.append(e["olap"]); errs_plain.append(e["plain"])
                add(f"{job}-P2-{g}", f"overlap form within 6% of base ({nm}, {g})", "band", abs(e["olap"]), None, lambda v: v <= 0.06, "<= 0.06", nm)
                ct = c["counters"]["base"]; tm = lambda r: r * S[tag] / (h["b_host"] * 1e9) * 1e3  # noqa: E731
                errs_half.append((c["G"] + tm(ct["misses"] + 0.5 * ct["admits"])) / c["ms"]["base"] - 1)
                errs_miss.append((c["G"] + tm(ct["misses"])) / c["ms"]["base"] - 1)
            ct, sp = c["counters"], c["speed"]
            rd = lambda k: ct[k]["misses"] + ct[k]["admits"]  # noqa: E731
            if tag == "g" and "dk" in ct and "base" in ct:
                add(f"{job}-P3-reads-{g}", f"dk reads fewer host experts than base ({nm}, {g})", "sign", rd("dk") / rd("base"), None, lambda v: v < 1, "< 1", nm)
            if sp.get("dk"):
                dk_all.append(sp["dk"][0]); dk_nochange.append(abs(sp["dk"][0] - 1))
                add(f"{job}-P3-speed-{g}", f"dk/base >= 0.995 ({nm}, {g})", "band", sp["dk"][0], sp["dk"][1:], lambda v: v >= 0.995, ">= 0.995", nm)
                if "pred" in c and "dk" in c["pred"] and "base" in c["pred"]:
                    pr = c["pred"]["base"]["olap"] / c["pred"]["dk"]["olap"]
                    c["ratio_pred"] = {"dk": pr}; dk_pred_err.append(abs(pr - sp["dk"][0]))
                    add(f"{job}-P4-{g}", f"law-predicted dk/base within 0.03 ({nm}, {g})", "band", abs(pr - sp["dk"][0]), None, lambda v: v <= 0.03, "<= 0.03", nm)
            if (tag, C) == ("g", 14):
                if h["B_p"] >= 20:
                    for k in ("fetchplan", "bypassplan"):
                        if sp.get(k):
                            add(f"{job}-P5-{k}", f"{k}/base > 1 at gpt-oss 11% ({nm})", "sign", sp[k][0], sp[k][1:], lambda v: v > 1, "> 1", nm)
                if sp.get("fetchplan") and sp.get("bypassplan"):
                    d_ = sp["bypassplan"][0] - sp["fetchplan"][0]
                    if h["ratio"] < 0.35:
                        add(f"{job}-P6", f"bypassplan beats fetchplan at ratio {h['ratio']:.2f} ({nm})", "sign", d_, None, lambda v: v > 0, "> 0", nm)
                    elif h["ratio"] >= 0.40:
                        add(f"{job}-P6", f"fetchplan beats bypassplan at ratio {h['ratio']:.2f} ({nm})", "sign", -d_, None, lambda v: v > 0, "> 0", nm)
    if errs_olap:
        add("107-P2-median", "median |error| of the overlap form at most 4%", "band", float(np.median(np.abs(errs_olap))), None, lambda v: v <= 0.04, "<= 0.04", "pooled")
        add("107-P2-plain", "overlap form's median |error| below the plain form's", "sign",
            float(np.median(np.abs(errs_plain)) - np.median(np.abs(errs_olap))), None, lambda v: v > 0, "> 0", "pooled")
    if dk_all:
        add("107-P3-median", "median dk/base over all cells >= 1.01", "band", float(np.median(dk_all)), None, lambda v: v >= 1.01, ">= 1.01", "pooled")
    add("107-valid", "at least 3 valid hosts", "band", len(valid), None, lambda v: v >= 3, ">= 3", "pooled")
    out = dict(gated=gated, hosts=[dict(**{k: v for k, v in h.items() if k != "cells"}, cells=h["cells"]) for h in hosts.values()])
    json.dump(out, open(P("prereg", "job107.json"), "w"), indent=1, default=float)
    json.dump(dict(job="107", script="jobs/107_newhosts@vast.sh", commit="1a0616a", scored_by="scripts/job107.py (machine)", clauses=clauses),
              open(P("prereg", "scorecard_107.json"), "w"), indent=1)
    st = Counter(c["status"] for c in clauses)
    M = dict(jlLaunched=str(len(hosts) + len(gated)), jlGated=str(len(gated)), jlRan=str(len(hosts)), jlValid=str(len(valid)),
             jlInvalid=str(len(hosts) - len(valid)), jlClauses=str(len(clauses)), jlHeld=str(st.get("held", 0)),
             jlPoint=str(st.get("held (point)", 0)), jlFailed=str(st.get("failed", 0)))
    M["jlGatedUuid"] = str(sum("rented before" in g["reason"] for g in gated.values()))
    M["jlGatedMem"] = str(sum("memory in use" in g["reason"] for g in gated.values()))

    def rng(k, vals, fmt="{:.2f}"):
        vals = [v for v in vals if v is not None]
        if vals:
            M[k + "Min"] = fmt.format(min(vals)).replace("-", "$-$"); M[k + "Max"] = fmt.format(max(vals)).replace("-", "$-$")
    num = ["", "one", "two", "three", "four", "five", "six", "seven"]
    M["jlValidWord"] = num[len(valid)] if len(valid) < len(num) else str(len(valid))
    M["jlValidNames"] = ", ".join(h["name"] for h in sorted(valid.values(), key=lambda h: h["ratio"]))
    inv = {j: h for j, h in hosts.items() if not h["valid"]}
    if inv:
        M["jlInvalidNames"] = ", ".join(h["name"] for h in inv.values())
        M["jlInvalidWhy"] = "; ".join(f"{h['name']}: " + ("loss off by more than 2\\%" if not h["V1"] else f"rounds {100 * h['V2_spread']:.0f}\\% apart")
                                       for h in inv.values())
    if errs_olap:
        a = np.abs(errs_olap)
        M["jlCells"] = str(len(a)); M["jlWithin"] = str(int(np.sum(a <= 0.06)))
        M["jlErrMed"] = f"{100 * np.median(a):.1f}"; M["jlErrMax"] = f"{100 * a.max():.1f}"
        rng("jlErr", [100 * e for e in errs_olap], "{:.1f}")
        M["jlPlainMed"] = f"{100 * np.median(np.abs(errs_plain)):.1f}"; M["jlPlainWithin"] = str(int(np.sum(np.abs(errs_plain) <= 0.06)))
        M["jlHalfMed"] = f"{100 * np.median(np.abs(errs_half)):.1f}"; M["jlHalfWithin"] = str(int(np.sum(np.abs(errs_half) <= 0.06)))
        M["jlMissMed"] = f"{100 * np.median(np.abs(errs_miss)):.1f}"; M["jlMissWithin"] = str(int(np.sum(np.abs(errs_miss) <= 0.06)))
    if dk_all:
        M["jlDkMed"] = f"{np.median(dk_all):.3f}"; rng("jlDk", dk_all, "{:.3f}")
    if dk_pred_err:
        M["jlDkPredMed"] = f"{np.median(dk_pred_err):.3f}"; M["jlDkPredMax"] = f"{max(dk_pred_err):.3f}"
        M["jlDkNoChangeMed"] = f"{np.median(dk_nochange):.3f}"
    rng("jlRatio", [h["ratio"] for h in valid.values()]); rng("jlLink", [h["B_p"] for h in valid.values()], "{:.0f}")
    rng("jlBhost", [h["b_host"] for h in valid.values()], "{:.0f}")
    rng("jlG", [c["G"] for h in valid.values() for c in h["cells"].values() if c["tag"] == "g"], "{:.1f}")
    rng("jlGQ", [c["G"] for h in valid.values() for c in h["cells"].values() if c["tag"] == "q"], "{:.1f}")
    rng("jlSpread", [100 * h["V2_spread"] for h in valid.values()], "{:.1f}")
    rng("jlUsed", [h["used_gb"] for h in hosts.values()], "{:.0f}")
    for k, nm in (("fetchplan", "Fetch"), ("bypassplan", "Bypass")):
        rng(f"jl{nm}Plan", [h["cells"]["g14"]["speed"][k][0] for h in valid.values() if "g14" in h["cells"] and h["cells"]["g14"]["speed"].get(k)])
    p6 = [c for c in clauses if "-P6" in c["id"]]
    M["jlCrossN"] = str(len(p6)); M["jlCrossHeld"] = str(sum(c["status"] != "failed" for c in p6))
    # the two valid machines by class, the gate's readings, the unsteady hosts' spreads
    for j, h in valid.items():
        nm = "Serv" if (h["cores"] > 32 or "engineering" in h["name"]) else "Desk"
        e = [100 * abs(c["err"]["base"]["olap"]) for c in h["cells"].values() if "err" in c and "base" in c["err"]]
        if e:
            M[f"jlNew{nm}ErrMin"] = f"{min(e):.0f}"; M[f"jlNew{nm}ErrMax"] = f"{max(e):.0f}"
            M[f"jlNew{nm}Name"] = h["name"]
    if errs_olap:
        M["jlFailCells"] = str(int(np.sum(np.abs(errs_olap) > 0.06)))
        M["jlUnder"] = str(int(np.sum(np.array(errs_olap) < 0)))
    rng("jlUsedGate", [g["used_gb"] for g in gated.values()], "{:.0f}")
    rng("jlInvalidSpread", [100 * h["V2_spread"] for h in inv.values()], "{:.1f}")
    rng("jlInvErr", [100 * abs(h["cells"]["g14"]["err"]["base"]["olap"]) for h in inv.values() if "err" in h["cells"].get("g14", {})], "{:.0f}")
    word = lambda n: num[n] if n < len(num) else str(n)  # noqa: E731
    M["jlRequiredWord"] = word(REQUIRED); M["jlGatedWord"] = word(len(gated)); M["jlRanWord"] = word(len(hosts))
    M["jlMemGate"] = str(MEM_GATE); M["jlClassCores"] = str(CLASS_CORES)
    for j, h in hosts.items():   # per host, by job letter: round spread, the relation's error at gpt-oss 11%, cores
        L = j[-1]
        M[f"jlSpread{L}"] = f"{100 * h['V2_spread']:.1f}" if h["V2_spread"] is not None else "--"
        if "err" in h["cells"].get("g14", {}):
            M[f"jlErrG{L}"] = f"{100 * abs(h['cells']['g14']['err']['base']['olap']):.0f}"
        M[f"jlCores{L}"] = str(h["cores"])
    if errs_olap:
        # no third host could have brought the pooled median under 4%: the median of the known errors with four more
        # cells of zero error (a third host's four cells)
        z = sorted(list(np.abs(errs_olap)) + [0.0] * 4)
        M["jlMedFloor"] = f"{100 * (z[len(z) // 2 - 1] + z[len(z) // 2]) / 2:.1f}"
    # per valid host, each form's median |error| and cells within 6%
    for j, h in valid.items():
        nm = "Serv" if h["cores"] > CLASS_CORES or h.get("numa", 1) > 1 or "engineering" in h["name"] else "Desk"
        f_err = {"Ol": [], "Pl": [], "Half": [], "Miss": []}
        for c in h["cells"].values():
            if "err" not in c or "base" not in c["err"]:
                continue
            ct = c["counters"]["base"]; tm = lambda r, c=c: r * S[c["tag"]] / (h["b_host"] * 1e9) * 1e3  # noqa: E731
            f_err["Ol"].append(c["err"]["base"]["olap"]); f_err["Pl"].append(c["err"]["base"]["plain"])
            f_err["Half"].append((c["G"] + tm(ct["misses"] + 0.5 * ct["admits"])) / c["ms"]["base"] - 1)
            f_err["Miss"].append((c["G"] + tm(ct["misses"])) / c["ms"]["base"] - 1)
        for k, v in f_err.items():
            a = np.abs(v)
            M[f"jlNew{nm}{k}Med"] = f"{100 * np.median(a):.1f}"; M[f"jlNew{nm}{k}Within"] = str(int(np.sum(a <= 0.06)))
        M[f"jlNew{nm}Cells"] = str(len(f_err["Ol"]))
        M[f"jlFail{nm}Word"] = word(int(np.sum(np.abs(f_err["Ol"]) > 0.06)))
        M[f"jlNew{nm}Cores"] = str(h["cores"])
        if nm == "Serv":
            meds = [np.median(np.abs(v)) for v in f_err.values()]
            M["jlNewServFormMin"] = f"{100 * min(meds):.0f}"; M["jlNewServFormMax"] = f"{100 * max(meds):.0f}"
            # the server's best rate rests on one concurrent reading; against its best CPU-only reading instead
            txt = open(f"{RES}/{h['dir']}/concur.txt").read()
            cpu = [float(x) for x in re.findall(r"^cpu_read_gbs t=\d+ ([\d.]+)", txt, re.M)]
            cpuH = re.findall(rf"^cpu_read_gbs t=(\d+) ([\d.]+)", txt, re.M)
            Bb = max(cpu)
            c = h["cells"]["g14"]; ct = c["counters"]["base"]; T = c["ms"]["base"]; G = c["G"]
            t = law(G, ct["misses"], ct["admits"], Bb, S["g"])
            M["jlServBhost"] = f"{h['b_host']:.0f}"; M["jlServCpuBest"] = f"{Bb:.0f}"
            M["jlServCpuBestThreads"] = next(n for n, v in cpuH if float(v) == Bb)
            big = min(cpuH, key=lambda x: abs(int(x[0]) - h["helpers"]))   # the probe at about the engine's thread count
            M["jlServCpuManyThreads"] = big[0]; M["jlServCpuMany"] = f"{float(big[1]):.0f}"; M["jlServHelpers"] = str(h["helpers"])
            M["jlServAltErr"] = f"{100 * abs(t / T - 1):.0f}"
            M["jlServAltEff"] = f"{(ct['misses'] + ct['admits'] * (1 - G / T)) * S['g'] / ((T - G) * 1e-3) / 1e9 / Bb:.2f}"
            M["jlServEffValid"] = f"{(ct['misses'] + ct['admits'] * (1 - G / T)) * S['g'] / ((T - G) * 1e-3) / 1e9 / h['b_host']:.2f}"
    # the three registered tests of the relation, cell by cell within 6% (job 105: the plain form; 106 and 107: overlap)
    e105 = [c["sum_law"]["err"] for h in json.load(open(P("prereg", "job105.json")))["hosts"] for c in h["cells"].values() if "sum_law" in c]
    M["jlRegFiveCells"] = str(len(e105)); M["jlRegFiveWithin"] = str(sum(abs(e) <= 0.06 for e in e105))
    e106 = [c["err"]["base"]["olap"] for h in json.load(open(P("prereg", "job106.json")))["hosts"] for c in h["cells"].values()
            if "err" in c and "base" in c["err"]]
    M["jlRegSixCells"] = str(len(e106)); M["jlRegSixWithin"] = str(sum(abs(e) <= 0.06 for e in e106))
    # the relation's prediction of dk/base where it missed the registered 0.03
    pf = [c["measured"] for c in clauses if "-P4-" in c["id"] and c["status"] == "failed"]
    M["jlDkPredFailN"] = word(len(pf))
    if pf:
        M["jlDkPredFailMin"] = f"{min(pf):.3f}"; M["jlDkPredFailMax"] = f"{max(pf):.3f}"
    dkf = [c for c in clauses if "-P3-speed-" in c["id"] and c["status"] == "failed"]
    M["jlDkFailWord"] = word(len(dkf))
    # the admission margin at gpt-oss over the machines of jobs 106 (the three that ran stably) and 107 (the valid ones)
    dkg = [c["speed"]["dk"][0] for h in valid.values() for c in h["cells"].values() if c["tag"] == "g" and c["speed"].get("dk")]
    n106 = 0
    for h in json.load(open(P("prereg", "job106.json")))["hosts"]:
        if h["dir"][:4] in ("106a", "106b", "106e"):
            n106 += 1
            dkg += [h["cells"][k]["speed"]["dk"][0] for k in ("g14", "g32") if h["cells"].get(k, {}).get("speed", {}).get("dk")]
    if dkg:
        M["jlDkAllN"] = str(len(dkg)); M["jlDkAllMachines"] = str(n106 + len(valid)); M["jlDkAllMed"] = f"{np.median(dkg):.2f}"
        M["jlDkAllLoss"] = str(sum(v < 1 for v in dkg)); M["jlDkAllLossPct"] = f"{100 * (1 - min(dkg)):.0f}"
        M["jlDkAllGainMax"] = f"{100 * (max(dkg) - 1):.0f}"
    # after the fact (not registered): every launch of jobs 093-107 at gpt-oss 11%, the deployed cache's implied read
    # rate (M + A (1 - G/T)) S / (T - G) over the probe's best rate, by machine class
    E = machine_classes()
    json.dump(E, open(P("prereg", "job107_classes.json"), "w"), indent=1)
    for cls, nm in (("desktop", "Desk"), ("server", "Serv")):
        sel = [e for e in E if e["class"] == cls]
        M[f"jl{nm}N"] = str(len(sel)); M[f"jl{nm}Machines"] = str(len({e["uuid"] for e in sel}))
        rng(f"jl{nm}Eff", [e["eff"] for e in sel]); rng(f"jl{nm}Err", [100 * e["err"] for e in sel], "{:.0f}")
        M[f"jl{nm}EffMed"] = f"{np.median([e['eff'] for e in sel]):.2f}"
        M[f"jl{nm}Within"] = str(sum(abs(e["err"]) <= 0.06 for e in sel))
        M[f"jl{nm}Valid"] = str(sum(e["valid"] for e in sel))
    fd = [e for e in E if e["class"] == "desktop" and not e["dir"].startswith("107")]
    M["jlFoundN"] = str(len(fd)); M["jlFoundMachines"] = str(len({e["uuid"] for e in fd}))
    M["jlFoundWithin"] = str(sum(abs(e["err"]) <= 0.06 for e in fd)); rng("jlFoundEff", [e["eff"] for e in fd])
    M["jlFoundMissN"] = str(sum(abs(e["err"]) > 0.06 for e in fd))
    M["jlServValidWord"] = word(sum(e["valid"] for e in E if e["class"] == "server"))
    M["jlServInvalidWord"] = word(sum(not e["valid"] for e in E if e["class"] == "server"))
    de = sorted(e["eff"] for e in E if e["class"] == "desktop")
    M["jlDeskEffSecond"] = f"{de[-2]:.2f}"; M["jlDeskEffLowSecond"] = f"{de[1]:.2f}"
    M["jlServCoresMin"] = str(min(e["cores"] for e in E if e["class"] == "server"))
    M["jlDeskCoresMax"] = str(max(e["cores"] for e in E if e["class"] == "desktop"))
    with open(P("paper", "wsg_job107.tex"), "w") as f:
        f.write("% generated by scripts/job107.py from the job 107 hosts (results/107?_newhosts@vast)\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    with open(P("paper", "tab_job107.tex"), "w") as f:
        f.write("% generated by scripts/job107.py\n\\begin{table}[t]\\centering\\footnotesize\n")
        f.write("\\caption{Job 107, registered before the runs, on the machines never rented before that passed the "
                "registered gate before any download ($^\\dagger$: rounds further apart than the registered 2\\%; such a host "
                "stopped after its gpt-oss 11\\% rounds and is outside the predictions): the deployed cache's time per token against \\cref{eq:sum} (GPU compute profiled on the same "
                "host before the timed runs; the run's own counters), and the speed relative to the deployed cache of the "
                "admission margin (\\emph{dk}) and of MIN's fewest-admission set copied in the step or loaded by the CPU, "
                "with round-level 95\\% intervals (rounds, then problems: two rounds of gpt-oss, one of Qwen3, whose "
                "intervals are over problems). Hosts sorted by the probe's link-to-CPU ratio.}\\label{tab:job107}\n")
        f.write("\\setlength\\tabcolsep{2.2pt}\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{@{}lrlrrrlll@{}}\\toprule\n")
        f.write(" & Link/ & & $G$ & \\multicolumn{2}{c}{Deployed (ms)} & & \\multicolumn{2}{c}{MIN, fewest admissions} \\\\\n")
        f.write("Host & CPU & Budget & (ms) & Eq.~\\ref{eq:sum} & meas. & dk & in the step & by the CPU \\\\\\midrule\n")
        fmt = lambda v: "--" if not v else f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"  # noqa: E731
        for j, h in sorted(hosts.items(), key=lambda x: x[1]["ratio"]):
            for key in ("g14", "g32", "q16", "q32"):
                c = h["cells"].get(key)
                if not c:
                    continue
                pr = c.get("pred", {}).get("base", {}).get("olap")
                f.write(f"{h['name']}{'' if h['valid'] else DAG} & {h['ratio']:.2f} & {LAB[(c['tag'], c['C'])].replace('%', chr(92) + '%')} & "
                        f"{(c['G'] or float('nan')):.1f} & {(pr or float('nan')):.1f} & {c['ms']['base']:.1f} & {fmt(c['speed'].get('dk'))} & "
                        f"{fmt(c['speed'].get('fetchplan'))} & {fmt(c['speed'].get('bypassplan'))} \\\\\n")
        f.write("\\bottomrule\\end{tabular}}\\end{table}\n")
    print("gated:", gated)
    print({j: (h["name"], h["V1"], h["V2_spread"]) for j, h in hosts.items()})
    print(dict(st))
    for c in clauses:
        print(c["status"], c["id"], c["short"], c["measured"], c["ci"])
    print(M)


if __name__ == "__main__":
    main()
