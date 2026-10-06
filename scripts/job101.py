"""Job 101: the in-step copy's loss on the panel's lowest-ratio machine, relaunched (101a = Pf of job 099), and a second
Ryzen 7 9800X3D behind a slower link (101b; job 099's Pb had link 46). Scores the predictions of
jobs/101_crossover@vast.sh by machine and writes prereg/job101.json, prereg/scorecard_101.json and paper/wsg_job101.tex.

    python scripts/job101.py
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from scripts.job100 import COUNT, S, load_host, tmodel  # noqa: E402
from scripts.panel_099 import _status, cell_data  # noqa: E402

RES = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
GL = {14: "g11", 32: "g25"}


def main():
    hosts = {os.path.basename(d)[:4]: load_host(d) for d in sorted(glob.glob(f"{RES}/101?_crossover@vast"))
             if os.path.exists(f"{d}/ec_g_C14.jsonl")}
    clauses = []

    def add(cid, short, typ, meas, ci, ok, thr, host):
        stt, why = _status(meas, ci, ok)
        clauses.append(dict(id=cid, short=short, type=typ, measured=None if meas is None else round(float(meas), 4),
                            ci=None if ci is None else [round(float(x), 4) for x in ci], threshold=thr, status=stt, why=why, host=host))

    def speeds(dirname, C):
        cd = cell_data(f"{RES}/{dirname}", "g", C)
        if not cd:
            return None, None
        b = cd["arr"]["base"].mean()
        return {k: float(b / v.mean()) for k, v in cd["arr"].items()}, float(b)
    M = {}
    # 1. Pf relaunched
    a = hosts.get("101a")
    if a:
        for C in (14, 32):
            if C not in a["cells"]:
                continue
            c = a["cells"][C]
            old, ob = speeds("099f_panel@vast", C)
            if C == 14:
                f = c["speed"]["fetch"]
                add("101a-P1-loss-g11", "fetch/base below 1 at 11% (Pf again)", "sign", f[0], f[1:], lambda v: v < 1, "< 1", "Pf again")
            for k in ("fetch", "foa", "aa", "both3p", "bypass"):
                if k in c["speed"] and old and k in old:
                    add(f"101a-P1-{k}-{GL[C]}", f"{k}/base within 0.03 of job 099f ({GL[C]})", "band", abs(c["speed"][k][0] - old[k]), None,
                        lambda v: v <= 0.03, "<= 0.03", "Pf again")
            if ob:
                add(f"101a-P1-base-{GL[C]}", f"base time within 3% of job 099f ({GL[C]})", "band", abs(c["ms"]["base"] / ob - 1), None,
                    lambda v: v <= 0.03, "<= 3%", "Pf again")
        M["jbPfFetch"] = f"{a['cells'][14]['speed']['fetch'][0]:.2f}"
        M["jbPfFetchLo"] = f"{a['cells'][14]['speed']['fetch'][1]:.2f}"; M["jbPfFetchHi"] = f"{a['cells'][14]['speed']['fetch'][2]:.2f}"
        old, _ = speeds("099f_panel@vast", 14)
        M["jbPfFetchOld"] = f"{old['fetch']:.2f}"
        dv = [abs(a["cells"][C]["speed"][k][0] - speeds("099f_panel@vast", C)[0][k]) for C in a["cells"] for k in ("fetch", "foa", "aa", "both3p", "bypass")
              if k in a["cells"][C]["speed"]]
        M["jbPfRelaunchMax"] = f"{max(dv):.3f}"
        M["jbPfRatio"] = f"{a['B_p'] / a['B_c']:.2f}"
        M["jbPfPaced"] = f"{a['cells'][14]['speed']['both3p'][0]:.3f}"
        # every relaunch of a panel machine (jobs 100 and 101) together
        j100 = json.load(open(P("prereg", "job100.json")))
        rl = [abs(v) for h in j100["hosts"] for c in h["cells"].values() if "relaunch" in c for k, v in c["relaunch"].items() if k != "base_ms_ratio"]
        rb = [abs(c["relaunch"]["base_ms_ratio"] - 1) for h in j100["hosts"] for c in h["cells"].values() if "relaunch" in c]
        nre = sum(1 for h in j100["hosts"] if any("relaunch" in c for c in h["cells"].values())) + 1
        rl += dv
        for C in a["cells"]:
            _, ob = speeds("099f_panel@vast", C)
            rb.append(abs(a["cells"][C]["ms"]["base"] / ob - 1))
        # jobs 103 and 104 relaunched Pf (103a, 104a) and Pg (103d, 104c, the same offer and GPU as 099g)
        machines = {"099d", "099h", "099f"}
        for rj, pj in (("103a", "099f"), ("104a", "099f"), ("103d", "099g"), ("104c", "099g")):
            d = glob.glob(f"{RES}/{rj}_minadm@vast")
            if not d:
                continue
            machines.add(pj)
            for C in (14, 32):
                new, nb = speeds(os.path.basename(d[0]), C)
                old, ob = speeds(f"{pj}_panel@vast", C)
                if not new or not old:
                    continue
                rb.append(abs(nb / ob - 1))
                rl += [abs(new[k] - old[k]) for k in ("fetch", "foa", "bypass") if k in new and k in old]
        nre = len(machines)
        M["jcRelaunchN"] = str(nre)
        M["jcRelaunchRatioMax"] = f"{max(rl):.3f}"
        M["jcRelaunchBaseMax"] = f"{100 * max(rb):.1f}"
    # 2. the 9800X3D behind a slower link
    b = hosts.get("101b")
    if b:
        old, _ = speeds("099b_panel@vast", 14)
        f = b["cells"][14]["speed"]["fetch"]
        add("101b-P2-belowPb", "fetch/base at 11% below Pb's (job 099b)", "sign", old["fetch"] - f[0], None, lambda v: v > 0, "> 0", "9800X3D, slow")
        add("101b-P2-aboveOne", "fetch/base at 11% above 1", "sign", f[0], f[1:], lambda v: v > 1, "> 1", "9800X3D, slow")
        M["jbXFetch"] = f"{f[0]:.2f}"; M["jbXFetchPb"] = f"{old['fetch']:.2f}"; M["jbXRatio"] = f"{b['B_p'] / b['B_c']:.2f}"
    # 3. model signs (frozen G, run counters) and probe-only predictions
    for job, h in hosts.items():
        nm = "Pf again" if job == "101a" else "9800X3D, slow"
        h["link_over_cpu"] = h["B_p"] / h["B_c"]
        for C, c in h["cells"].items():
            pt = {}
            for k in ("base", "fetch", "aa"):
                if k in c["counters"]:
                    m = c["counters"][k]
                    pt[k] = tmodel((m["misses"] - m["fetches"]) * S, m["fetches"] * S, h, C)
            if "fetch" in pt and "base" in pt:
                pr = pt["base"] / pt["fetch"]
                c["pred_fetch_ratio"] = pr
                add(f"{job}-P3-sign-{GL[C]}", f"model sign of fetch/base, predicted {pr:.2f} ({nm}, {GL[C]})", "sign", c["speed"]["fetch"][0],
                    c["speed"]["fetch"][1:], (lambda v, pr=pr: (v > 1) == (pr > 1)), ">1" if pr > 1 else "<1", nm)
            if h["predict"]:
                for k in ("aa", "fetch"):
                    key = f"C{C}_{k}"
                    if key in h["predict"]["pred_ms"] and k in c["ms"]:
                        e = h["predict"]["pred_ms"][key] / c["ms"][k] - 1
                        c.setdefault("probe_err", {})[k] = e
                        if k == "aa":
                            add(f"{job}-P3-aa-{GL[C]}", f"probe-only aa within 12% ({nm}, {GL[C]})", "band", abs(e), None, lambda v: v <= 0.12, "<= 12%", nm)
                        elif job == "101a":
                            add(f"{job}-P3-fetchunder-{GL[C]}", f"probe-only fetch under-predicted ({nm}, {GL[C]})", "sign", e, None, lambda v: v < 0, "< 0", nm)
    out = []
    for job, h in hosts.items():
        hh = {k: v for k, v in h.items() if k != "cells"}
        hh["job"] = job
        hh["cells"] = {str(C): {k: v for k, v in c.items() if k != "arr"} for C, c in h["cells"].items()}
        out.append(hh)
    json.dump(dict(hosts=out), open(P("prereg", "job101.json"), "w"), indent=1)
    json.dump(dict(job="101", script="jobs/101_crossover@vast.sh", commit="8b6c67a", scored_by="scripts/job101.py (machine)", clauses=clauses),
              open(P("prereg", "scorecard_101.json"), "w"), indent=1)
    from collections import Counter
    st = Counter(c["status"] for c in clauses)
    M.update(jbClauses=str(len(clauses)), jbHeld=str(st.get("held", 0)), jbPoint=str(st.get("held (point)", 0)),
             jbFailed=str(st.get("failed", 0)), jbUntested=str(st.get("untested", 0)))
    with open(P("paper", "wsg_job101.tex"), "w") as f:
        f.write("% generated by scripts/job101.py\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    print(dict(st))
    for c in clauses:
        print(c["status"], c["id"], c["short"], c["measured"], c["ci"])
    for job, h in hosts.items():
        print(job, h["cpu"], f"link {h['B_p']:.1f} cpu {h['B_c']:.1f} both {h['B_cp']:.1f}")
        for C, c in h["cells"].items():
            print("  ", C, {k: round(v[0], 3) for k, v in c["speed"].items()}, "probe err", {k: round(v, 3) for k, v in c.get("probe_err", {}).items()})
    print(M)


if __name__ == "__main__":
    main()
