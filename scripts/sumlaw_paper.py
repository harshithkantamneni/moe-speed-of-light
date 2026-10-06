"""Macros for the sum law and the demand bound (paper/wsg_sumlaw.tex), from prereg/reanalysis.json (scripts/reanalysis.py),
the Nsight profiles of job 069c (the GPU's own kernel time per token) and, when present, job 105's profiles.

    python scripts/sumlaw_paper.py
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
S = 13253760


def profiles():
    """per profile: the GPU's own kernel time per token (every category but the helper wait and the expert copies),
    and its non-expert part (also without resident-expert GEMVs and the cache's control kernels)"""
    out = []
    for f in sorted(glob.glob(f"{RES}/069c_profile_*@vast/prof_C14.json") + glob.glob(f"{RES}/069c_profile_*@vast/prof_C32.json")
                    + glob.glob(f"{RES}/105?_sumlaw@vast/prof_C14.json") + glob.glob(f"{RES}/105?_sumlaw@vast/prof_C32.json")):
        d = json.load(open(f))["median_ms"]
        g = sum(v for k, v in d.items() if k.startswith("cat_") and k not in ("cat_ec_wait", "cat_ec_copy"))
        ne = g - d.get("cat_expert_gemv", 0.0) - d.get("cat_ec_ctl", 0.0)
        out.append(dict(file=os.path.relpath(f, RES), G=g, nonexpert=ne))
    return out


def main():
    ra = json.load(open(P("prereg", "reanalysis.json")))
    M = {}
    pr = profiles()
    g = [p["G"] for p in pr]; ne = [p["nonexpert"] for p in pr]
    M["slProfN"] = str(len(pr)); M["slProfHosts"] = str(len({os.path.dirname(p["file"]) for p in pr}))
    M["slGprofMin"] = f"{min(g):.1f}"; M["slGprofMax"] = f"{max(g):.1f}"
    M["slGneMin"] = f"{min(ne):.1f}"; M["slGneMax"] = f"{max(ne):.1f}"
    M["slLaunches"] = str(ra["launches"]); M["slMachines"] = str(ra["machines"])
    for C, nm in (("14", "Low"), ("32", "Mid")):
        s = ra["sum_law"][C]
        b = s["implied_G"]["base"]
        M[f"slG{nm}"] = f"{b['median']:.1f}"; M[f"slG{nm}Qa"] = f"{b['q1']:.1f}"; M[f"slG{nm}Qb"] = f"{b['q3']:.1f}"
        M[f"slGRho{nm}"] = f"{b['rho_ratio']:.2f}".replace("-", "$-$")
        e = s["elasticity"]
        M[f"slElast{nm}"] = f"{e['T_minus_G']['slope']:.2f}".replace("-", "$-$")
        M[f"slElast{nm}Lo"] = f"{e['T_minus_G']['lo']:.2f}".replace("-", "$-$"); M[f"slElast{nm}Hi"] = f"{e['T_minus_G']['hi']:.2f}".replace("-", "$-$")
        M[f"slElastRaw{nm}"] = f"{e['T']['slope']:.2f}".replace("-", "$-$")
        M[f"slLawG{nm}"] = f"{s['G']:.1f}"
        le = s["law_err"]
        M["slLawN"] = str(le["n"]); M[f"slLawWithin{nm}"] = str(le["within6"]); M[f"slLawOut{nm}"] = str(le["n"] - le["within6"])
        M[f"slLawErr{nm}Min"] = f"{100 * le['lo']:.0f}".replace("-", "$-$"); M[f"slLawErr{nm}Max"] = f"{100 * le['hi']:.0f}".replace("-", "$-$")
        M[f"slLawErr{nm}Med"] = f"{100 * le['median']:.0f}".replace("-", "$-$"); M[f"slLawAbs{nm}Med"] = f"{100 * le['median_abs']:.1f}"
        ec = s["eff_over_counted"]
        M[f"slEffC{nm}"] = f"{ec['median']:.2f}"; M[f"slEffC{nm}Qa"] = f"{ec['q1']:.2f}"; M[f"slEffC{nm}Qb"] = f"{ec['q3']:.2f}"
        M[f"slGGap{nm}Min"] = f"{100 * s['G_share_of_gap']['lo']:.0f}"; M[f"slGGap{nm}Max"] = f"{100 * s['G_share_of_gap']['hi']:.0f}"
        M[f"slGTime{nm}Min"] = f"{100 * s['G_share_of_time']['lo']:.0f}"; M[f"slGTime{nm}Max"] = f"{100 * s['G_share_of_time']['hi']:.0f}"
        for k, kn in (("fetch", "Fetch"), ("both3p", "Paced"), ("aa", "Aa"), ("bypass", "Bypass")):
            if k in s["implied_G"]:
                v = s["implied_G"][k]
                M[f"slG{kn}{nm}"] = f"{v['median']:.1f}"
                M[f"slG{kn}Rho{nm}"] = f"{v['rho_ratio']:.2f}".replace("-", "$-$")
    # the demand bound: the GPU's non-expert compute (smallest profiled) in series with MIN's reads at B_host
    hosts = json.load(open(P("prereg", "reanalysis_hosts.json")))["launches"]
    tne = min(ne)
    for C, nm in (("14", "Low"), ("32", "Mid")):
        eff, effmax = [], []
        for h in hosts:
            c = h["cells"].get(C)
            if not c or "limit_ms" not in c:
                continue
            eff.append((tne + c["limit_ms"]) / c["base_ms"])
            effmax.append(c["limit_ms"] / c["base_ms"])
            if "ms_both3p" in c:
                pass
        M[f"slDemEff{nm}Min"] = f"{100 * min(eff):.0f}"; M[f"slDemEff{nm}Max"] = f"{100 * max(eff):.0f}"
        M[f"slMaxEff{nm}Min"] = f"{100 * min(effmax):.0f}"; M[f"slMaxEff{nm}Max"] = f"{100 * max(effmax):.0f}"
    o4 = next(h for h in hosts if h["dir"].startswith("096a"))["cells"]["14"]
    M["slExDemandMs"] = f"{tne + o4['limit_ms']:.1f}"
    M["slExDemandPct"] = f"{100 * (tne + o4['limit_ms']) / o4['base_ms']:.0f}"
    M["slExBoundMs"] = f"{o4['limit_ms']:.1f}"
    # rank correlations across machines
    cor = {(r["outcome"], r["feature"]): r for r in ra["correlations"]}

    def rc(name, o, f):
        r = cor.get((o, f))
        if r:
            M[name] = f"{r['rho']:.2f}".replace("-", "$-$")
            M[name + "Lo"] = f"{r['lo']:.2f}".replace("-", "$-$"); M[name + "Hi"] = f"{r['hi']:.2f}".replace("-", "$-$")
    rc("rcFetchRatio", "fetch/base@14", "ratio")
    rc("rcPacedRatio", "both3p/base@14", "ratio")
    rc("rcAaRatio", "aa/base@14", "ratio")
    rc("rcBypassBp", "bypass/base@14", "B_p")
    rc("rcBypassRatio", "bypass/base@14", "ratio")
    rc("rcFoaBhost", "foa/base@14", "B_host")
    rc("rcBaseBhost", "base_ms@14", "B_host")
    rc("rcEffBc", "eff_base@32", "B_c")
    for name, o, f in (("rcFetchRatio", "fetch/base@14", "ratio"), ("rcPacedRatio", "both3p/base@14", "ratio"),
                       ("rcAaRatio", "aa/base@14", "ratio"), ("rcBypassBp", "bypass/base@14", "B_p")):
        if (o, f) in cor:
            M[name + "N"] = str(cor[(o, f)]["n"])
    pp = ra["per_problem"]
    M["ppWFetchLow"] = f"{pp['fetch@14']['kendall_W']:.2f}"; M["ppWFetchMid"] = f"{pp['fetch@32']['kendall_W']:.2f}"
    M["ppWFoaLow"] = f"{pp['foa@14']['kendall_W']:.2f}"
    M["ppRhoFetchLow"] = f"{pp['fetch@14']['rho_excess_rel']:.2f}"; M["ppRhoFetchMid"] = f"{pp['fetch@32']['rho_excess_rel']:.2f}"
    vf = [pp[k]["var_hosts"] for k in ("fetch@14", "fetch@32")]; vpf = [pp[k]["var_problems"] for k in ("fetch@14", "fetch@32")]
    M["ppVarHostsFetchMin"] = f"{100 * min(vf):.0f}"; M["ppVarHostsFetchMax"] = f"{100 * max(vf):.0f}"
    M["ppVarProbFetchMin"] = f"{100 * min(vpf):.0f}"; M["ppVarProbFetchMax"] = f"{100 * max(vpf):.0f}"; M["ppRhoPacedLow"] = f"{pp['both3p@14']['rho_excess_rel']:.2f}"
    vh = [pp[k]["var_hosts"] for k in ("fetch@14", "fetch@32", "both3p@14", "both3p@32")]
    vp = [pp[k]["var_problems"] for k in ("fetch@14", "fetch@32", "both3p@14", "both3p@32")]
    M["ppVarHostsMin"] = f"{100 * min(vh):.0f}"; M["ppVarHostsMax"] = f"{100 * max(vh):.0f}"
    M["ppVarProbMin"] = f"{100 * min(vp):.0f}"; M["ppVarProbMax"] = f"{100 * max(vp):.0f}"
    M["ppMachines"] = str(pp["fetch@14"]["machines"]); M["ppProblems"] = str(pp["fetch@14"]["problems"])
    uu = {h.get("gpu_uuid") or h["dir"] for h in hosts}
    n105 = 0
    for d in sorted(glob.glob(f"{RES}/105?_sumlaw@vast")):
        if not os.path.exists(f"{d}/ec_g_C14.jsonl"):
            continue
        n105 += 1
        u = [ln.split(":", 1)[1].strip() for ln in open(f"{d}/nvidia-smi-q.txt") if "GPU UUID" in ln]
        uu.add(u[0] if u else d)
    M["slAllLaunches"] = str(ra["launches"] + n105); M["slAllMachines"] = str(len(uu))
    with open(P("paper", "wsg_sumlaw.tex"), "w") as f:
        f.write("% generated by scripts/sumlaw_paper.py\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    print(M)


if __name__ == "__main__":
    main()
