"""Where the seconds go, measured: the deployed cache's time beyond Eq. (1) (the gap), split with the engine's own
oracles, machine by machine, at the two host-bound gpt-oss budgets. Every machine that ran the four states of the 2x2
(deployed set or MIN's set, read twice or once) and MIN prefetched, one launch per GPU:

    set alone     MIN's set, still read twice:          deployed - MIN read twice
    once alone    the deployed set, each admission read once: deployed - deployed read once
    together      MIN's set read once:                  deployed - MIN read once  (= both alone + their interaction)
    ahead         MIN read once with its copies issued a few steps ahead: MIN read once - MIN prefetched
    serial        the GPU's non-expert work in series with the reads: T_GPU (assumed: the median Nsight profile,
                  with the smallest and largest profiles as a band); it is not removed in the engine
    rest          MIN prefetched - Eq. (1) - T_GPU: a residual

together + ahead + serial + rest = the gap. Times are the engine's per-token means over the problems
(prereg/reanalysis_hosts.json); Eq. (1) is the paper's bound on that machine. The split into fast and slow links
(link-to-CPU ratio 0.5) was drawn after the data. Writes paper/wsg_dm.tex, paper/tab_dm.tex,
prereg/decomp_measured.json and paper/figs/decomp_measured.pdf.

    python scripts/decomp_measured.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from scripts.sumlaw_paper import profiles  # noqa: E402

P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
RNG = np.random.default_rng(17)
FAST = 0.5          # link-to-CPU ratio at or above which a machine counts as fast-linked (drawn after the data)
STATES = ("base", "bypass", "foa", "fetch", "both3p")
PARTS = ("setalone", "oncealone", "together", "ahead", "serial", "rest", "left", "extra", "resid")
RES = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")
TLO = [2.9402065]   # set in main() to the smallest profiled T_GPU
S = 13253760


def reads_of(d, C, st):
    """host reads per token of a state, from its counters: misses, admissions and prefetches per step"""
    import glob
    fs = sorted(glob.glob(f"{RES}/{d}/st_g_C{C}_{st}.json")) or sorted(glob.glob(f"{RES}/{d}/st_g_C{C}_r*_{st}.json"))
    v = []
    for f in fs:
        x = json.load(open(f)); v.append((x["misses"] + x["admits"] + x.get("prefetches", 0)) / max(1, x["steps"]))
    return float(np.mean(v)) if v else None


def rows():
    ne = sorted(p["nonexpert"] for p in profiles())
    tmed, tlo, thi = float(np.median(ne)), ne[0], ne[-1]
    rstar = {C: json.load(open(P("prereg", "speed_limit_v2.json")))["models"]["gpt-oss-120b"]["rows"][C]["reads_per_token"]["exact"]
             for C in ("14", "32")}
    out, seen = [], set()
    for h in json.load(open(P("prereg", "reanalysis_hosts.json")))["launches"]:
        u = h.get("gpu_uuid") or h["dir"]
        for C in ("14", "32"):
            c = h["cells"].get(C, {})
            if (u, C) in seen or "limit_ms" not in c or not all(f"ms_{k}" in c for k in STATES):
                continue
            seen.add((u, C))
            t = {k: c[f"ms_{k}"] for k in STATES}
            eq1 = c["limit_ms"]
            gap = t["base"] - eq1
            parts = dict(setalone=t["base"] - t["bypass"], oncealone=t["base"] - t["foa"], together=t["base"] - t["fetch"],
                         ahead=t["fetch"] - t["both3p"], serial=tmed, rest=t["both3p"] - eq1 - tmed, left=t["both3p"] - eq1)
            # what is left, accounted for: the GPU's non-expert time (Eq. (2)'s T_GPU) and the prefetching oracle's reads
            # beyond MIN's at the machine's best rate; the rest is a residual
            rp = reads_of(h["dir"], C, "both3p")
            parts["extra"] = (rp - rstar[C]) * S / (float(h["B_host"]) * 1e9) * 1e3
            parts["resid"] = parts["left"] - tlo - parts["extra"]
            assert abs(parts["together"] + parts["ahead"] + parts["left"] - gap) < 1e-9
            share = {k: v / gap for k, v in parts.items()}
            share["interaction"] = share["together"] - share["setalone"] - share["oncealone"]
            share["serial_lo"], share["serial_hi"] = tlo / gap, thi / gap
            share["tgpu"] = tlo / gap
            out.append(dict(dir=h["dir"], uuid=u, C=int(C), ratio=float(h["ratio"]), cpu=h.get("cpu", ""), t=t, eq1=eq1,
                            eq2=eq1 + tlo, gap=gap, parts=parts, share=share, bound_share=eq1 / t["base"],
                            pref_over_eq2=t["both3p"] / (eq1 + tlo),
                            closed=(t["base"] - min(t["bypass"], t["fetch"], t["both3p"])) / gap))
    return out, (tmed, tlo, thi)


def rows_new(tlo):
    """job 109's valid machines (new desktop-class machines, link-to-CPU ratio >= 0.5 fixed before launch), same parts"""
    p = P("prereg", "job109.json")
    if not os.path.exists(p):
        return []
    J = json.load(open(p))
    rstar = {14: 38.315364583333334, 32: 15.340364583333333}
    out = []
    for h in J["hosts"]:
        if not h["valid"]:
            continue
        for C, c in h["cells"].items():
            C = int(C)
            t = c.get("t", {})
            if not all(k in t for k in STATES) or "eq1" not in c:
                continue
            eq1 = c["eq1"]; gap = t["base"] - eq1
            parts = dict(setalone=t["base"] - t["bypass"], oncealone=t["base"] - t["foa"], together=t["base"] - t["fetch"],
                         ahead=t["fetch"] - t["both3p"], serial=tlo, rest=t["both3p"] - eq1 - tlo, left=t["both3p"] - eq1)
            parts["extra"] = (c["reads"]["both3p"] - rstar[C]) * S / (float(h["B_host"]) * 1e9) * 1e3
            parts["resid"] = parts["left"] - tlo - parts["extra"]
            share = {k: v / gap for k, v in parts.items()}
            share["interaction"] = share["together"] - share["setalone"] - share["oncealone"]
            share["tgpu"] = tlo / gap
            out.append(dict(dir=h["dir"], C=C, ratio=float(h["ratio"]), cpu=h.get("cpu", ""), t=t, eq1=eq1, gap=gap,
                            share=share, reads_both=c["reads"]["both3p"] / rstar[C],
                            closed=(t["base"] - min(t["bypass"], t["fetch"], t["both3p"])) / gap))
    return out


def boot(v):
    """the mean over machines and its 95% t-interval (the machine is the unit; with 4-15 machines a percentile bootstrap
    over machines is too narrow: with 4 machines it has 35 distinct resamples)"""
    from scipy.stats import t as tdist
    v = np.array(v, float)
    h = tdist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(v.mean() - h), float(v.mean() + h)


def pct(x):
    v = int(round(100 * x))
    return str(v).replace("-", "$-$")


def figure(R, path, NEW=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    labels = ["deployed", "MIN-2R\n(or Dep-1R)", "MIN-1R", "read-ahead\noracle", "Eq. (2)", "Eq. (1)"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    blue, dark, orange, green = "#2a78d6", "#1b3f6b", "#eb6834", "#2e9e5b"
    for ax, (C, lab) in zip(axes, ((14, "gpt-oss 11%"), (32, "gpt-oss 25%"))):
        sel = [r for r in R if r["C"] == C]
        A, B = [], []
        for r in sel:
            t = r["t"]
            ya = [1.0, t["bypass"] / t["base"], t["fetch"] / t["base"], t["both3p"] / t["base"], r["eq2"] / t["base"], r["eq1"] / t["base"]]
            fast = r["ratio"] >= FAST
            col = blue if fast else orange
            ax.plot([0, 0.88, 2, 3, 4, 5], ya, color=col, lw=0.8, alpha=0.5 if fast else 0.9, zorder=2 if fast else 3)
            ax.plot([1.12], [t["foa"] / t["base"]], marker="s", ms=2.6, mfc="none", mec=col, mew=0.7, lw=0, zorder=3)
            if fast:
                A.append(ya); B.append(t["foa"] / t["base"])
        for r in (x for x in NEW if x["C"] == C):
            t = r["t"]
            yn = [1.0, t["bypass"] / t["base"], t["fetch"] / t["base"], t["both3p"] / t["base"], (r["eq1"] + TLO[0]) / t["base"], r["eq1"] / t["base"]]
            ax.plot([0, 0.88, 2, 3, 4, 5], yn, color=green, lw=1.1, alpha=0.95, zorder=4)
            ax.plot([1.12], [t["foa"] / t["base"]], marker="s", ms=2.6, mfc="none", mec=green, mew=0.8, lw=0, zorder=4)
        med = np.median(np.array(A), axis=0); mb = float(np.median(B))
        ax.plot([0, 0.88, 2, 3, 4, 5], med, color=dark, lw=2.0, marker="o", ms=3.4, zorder=5)
        ax.plot([0, 1.12, 2], [1.0, mb, med[2]], color=dark, lw=1.6, ls=(0, (3, 1.5)), marker="s", ms=3.4, mfc="white", zorder=5)
        ax.set_xticks(range(6)); ax.set_xticklabels(labels, fontsize=6.3)
        ax.set_title(lab, fontsize=8)
        ax.grid(axis="y", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=6.5, length=2)
        ax.set_ylim(0.2, 1.25)
    axes[0].set_ylabel("time per token / deployed", fontsize=7)
    nf = sum(1 for r in R if r["C"] == 14 and r["ratio"] >= FAST); ns = sum(1 for r in R if r["C"] == 14 and r["ratio"] < FAST)
    h = [Line2D([0], [0], color=dark, lw=2.0, marker="o", ms=3.4, label="median: MIN-2R first (MIN's set, 2 reads)"),
         Line2D([0], [0], color=dark, lw=1.6, ls=(0, (3, 1.5)), marker="s", ms=3.4, mfc="white", label="median: Dep-1R first (deployed set, 1 read)"),
         Line2D([0], [0], color=blue, lw=0.8, label=f"each machine, link/CPU $\\geq$ {FAST} ({nf})"),
         Line2D([0], [0], color=orange, lw=0.8, label=f"link/CPU $<$ {FAST} ({ns})")]
    if NEW:
        n11 = len({r['dir'] for r in NEW if r['C'] == 14}); n25 = len({r['dir'] for r in NEW if r['C'] == 32})
        h.append(Line2D([0], [0], color=green, lw=1.1, label=f"new machines, registered ({n11}; {n25} at 25%)"))
    fig.legend(handles=h, loc="upper center", ncol=3 if NEW else 4, fontsize=6.2, frameon=False, bbox_to_anchor=(0.5, 1.06 if NEW else 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, bbox_inches="tight"); fig.savefig(path.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")


def table(M, has_new=False):
    meas = [("setalone", "MIN's set alone (\\MinTwo)"),
            ("oncealone", "One read alone (\\DepOne)"),
            ("together", "Both (\\MinOne)"),
            ("ahead", "Then the read-ahead oracle"),
            ("left", "Left after all three")]
    attr = [("tgpu", "the GPU's non-expert time ($T_{\\text{GPU}}$)"),
            ("extra", "the oracle's reads beyond MIN's, at $B_{\\mathrm{host}}$"),
            ("resid", "residual")]
    def cell(pre, k, nm):
        K = k.capitalize()
        if pre + K + nm not in M:
            return "--"
        return f"{M[pre + K + nm]} [{M[pre + K + nm + 'Lo']}, {M[pre + K + nm + 'Hi']}]"
    with open(P("paper", "tab_dm.tex"), "w") as f:
        f.write("% generated by scripts/decomp_measured.py\n\\begin{table*}[t]\\centering\\footnotesize\n")
        f.write("\\caption{The deployed cache's gap to \\cref{eq:limit}, split by oracles in the engine (\\cref{fig:staircase}): mean "
                "share of the gap, in percent, with 95\\% $t$-intervals over machines. \\emph{Panel}: the " + M["dmMachines"] + " machines "
                "of jobs 096--101 that ran every state. " + ("\\emph{New}: the " + M["dmNewN"] + " machines of job 109, rented for "
                "this test, whose population (desktop-class, link-to-CPU ratio at least 0.5) and predictions were registered "
                "before any of them started; the job's deadline cut one machine's 25\\% round. " if has_new else "") + "The first two rows each change one thing; the third "
                "changes both. \\emph{Both}, \\emph{then the read-ahead oracle} and \\emph{left} sum to the gap. The last three "
                "rows are attributed, not measured: they split \\emph{left} with $T_{\\text{GPU}}$ as in \\cref{eq:demand} "
                "(" + M["dmTgpu"] + "\\,ms, the smallest profile) and the read-ahead oracle's reads beyond $R^\\star$ at "
                "$B_{\\mathrm{host}}$; the residual is what they leave.}\\label{tab:gap}\n")
        cols = "rrrr" if has_new else "rr"
        f.write("\\setlength\\tabcolsep{4pt}\\begin{tabular}{@{}l" + cols + "@{}}\\toprule\n")
        if has_new:
            f.write(" & \\multicolumn{2}{c}{gpt-oss 11\\%} & \\multicolumn{2}{c}{gpt-oss 25\\%} \\\\\\cmidrule(lr){2-3}\\cmidrule(l){4-5}\n")
            f.write("Part of the gap & Panel (" + M["dmMachines"] + ") & New (" + M["dmNewNLow"] + ") & Panel (" + M["dmMachines"]
                    + ") & New (" + M["dmNewNMid"] + ") \\\\\\midrule\n")
        else:
            f.write("Part of the gap & gpt-oss 11\\% & 25\\% \\\\\\midrule\n")
        f.write("\\multicolumn{" + str(len(cols) + 1) + "}{@{}l}{\\emph{Measured}} \\\\\n")
        for item in meas + [None] + attr:
            if item is None:
                f.write("\\addlinespace\\multicolumn{" + str(len(cols) + 1) + "}{@{}l}{\\emph{Attributed: of what is left}} \\\\\n")
                continue
            k, text = item
            row = [cell("dmAll", k, "Low")]
            if has_new:
                row.append(cell("dmNew", k, "Low"))
            row.append(cell("dmAll", k, "Mid"))
            if has_new:
                row.append(cell("dmNew", k, "Mid"))
            f.write("\\quad " + text + " & " + " & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table*}\n")


def main():
    R, (tmed, tlo, thi) = rows()
    TLO[0] = tlo
    M = {}
    words = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    w = lambda n: words[n] if n < len(words) else str(n)  # noqa: E731
    lo = [r for r in R if r["C"] == 14]
    M["dmMachines"] = str(len(lo)); M["dmFastN"] = str(sum(r["ratio"] >= FAST for r in lo))
    M["dmSlowN"] = w(sum(r["ratio"] < FAST for r in lo)); M["dmFast"] = f"{FAST}"
    M["dmTgpu"] = f"{tlo:.1f}"; M["dmTgpuMed"] = f"{tmed:.1f}"; M["dmTgpuMax"] = f"{thi:.1f}"
    out = {}
    for C, nm in ((14, "Low"), (32, "Mid")):
        sel = [r for r in R if r["C"] == C]
        fast = [r for r in sel if r["ratio"] >= FAST]; slow = [r for r in sel if r["ratio"] < FAST]
        for k in PARTS + ("interaction", "tgpu"):
            m, a, b = boot([r["share"][k] for r in fast])
            K = k.capitalize()
            M[f"dm{K}{nm}"] = pct(m); M[f"dm{K}{nm}Lo"] = pct(a); M[f"dm{K}{nm}Hi"] = pct(b)
            M[f"dm{K}{nm}Min"] = pct(min(r["share"][k] for r in fast)); M[f"dm{K}{nm}Max"] = pct(max(r["share"][k] for r in fast))
        M[f"dmSerial{nm}Plo"] = pct(np.mean([r["share"]["serial_lo"] for r in fast]))
        M[f"dmSerial{nm}Phi"] = pct(np.mean([r["share"]["serial_hi"] for r in fast]))
        for k in ("together", "ahead", "left"):
            m, a, b = boot([r["share"][k] for r in sel])
            M[f"dm{k.capitalize()}All{nm}"] = pct(m); M[f"dm{k.capitalize()}All{nm}Lo"] = pct(a); M[f"dm{k.capitalize()}All{nm}Hi"] = pct(b)
        for k in PARTS + ("interaction", "tgpu"):   # every part over all machines (the table's panel column)
            m, a, b = boot([r["share"][k] for r in sel])
            M[f"dmAll{k.capitalize()}{nm}"] = pct(m); M[f"dmAll{k.capitalize()}{nm}Lo"] = pct(a); M[f"dmAll{k.capitalize()}{nm}Hi"] = pct(b)
        m, a, b = boot([r["closed"] for r in sel]); M[f"dmClosedAll{nm}"] = pct(m)
        m, a, b = boot([r["closed"] for r in fast])
        M[f"dmClosed{nm}"] = pct(m); M[f"dmClosed{nm}Lo"] = pct(a); M[f"dmClosed{nm}Hi"] = pct(b)
        M[f"dmClosedSlow{nm}Max"] = pct(max(r["closed"] for r in slow))
        M[f"dmSlowResid{nm}Min"] = pct(min(r["share"]["resid"] for r in slow)); M[f"dmSlowResid{nm}Max"] = pct(max(r["share"]["resid"] for r in slow))
        M[f"dmPrefEqTwo{nm}Min"] = f"{min(r['pref_over_eq2'] for r in fast):.2f}"
        M[f"dmPrefEqTwo{nm}Max"] = f"{max(r['pref_over_eq2'] for r in fast):.2f}"
        M[f"dmBoundShare{nm}Min"] = pct(min(r["bound_share"] for r in sel)); M[f"dmBoundShare{nm}Max"] = pct(max(r["bound_share"] for r in sel))
        out[nm] = [dict(dir=r["dir"], ratio=r["ratio"], cpu=r["cpu"], times=r["t"], eq1=r["eq1"], share=r["share"]) for r in sel]
    # job 109: the registered replication on new machines (population fixed before launch: desktop-class, ratio >= 0.5)
    NEW = rows_new(tlo)
    M["dmNewN"] = str(len({r["dir"] for r in NEW})); M["dmNewNWord"] = w(len({r["dir"] for r in NEW}))
    for C, nm in ((14, "Low"), (32, "Mid")):
        sel = [r for r in NEW if r["C"] == C]
        M[f"dmNewN{nm}"] = str(len(sel))
        if not sel:
            continue
        for k in PARTS + ("interaction", "tgpu"):
            v = [r["share"][k] for r in sel]
            m, a_, b_ = boot(v) if len(v) > 1 else (v[0], v[0], v[0])
            K = k.capitalize()
            M[f"dmNew{K}{nm}"] = pct(m); M[f"dmNew{K}{nm}Lo"] = pct(a_); M[f"dmNew{K}{nm}Hi"] = pct(b_)
            M[f"dmNew{K}{nm}Min"] = pct(min(v)); M[f"dmNew{K}{nm}Max"] = pct(max(v))
        M[f"dmNewInteractionPos{nm}"] = str(sum(r["share"]["interaction"] > 0 for r in sel))
        m, a_, b_ = boot([r["closed"] for r in sel]) if len(sel) > 1 else (sel[0]["closed"],) * 3
        M[f"dmNewClosed{nm}"] = pct(m)
        M[f"dmNewPrefReads{nm}Min"] = f"{min(r['reads_both'] for r in sel):.2f}"; M[f"dmNewPrefReads{nm}Max"] = f"{max(r['reads_both'] for r in sel):.2f}"
        out["New" + nm] = [dict(dir=r["dir"], ratio=r["ratio"], cpu=r["cpu"], times=r["t"], eq1=r["eq1"], share=r["share"]) for r in sel]
    M["dmNewRatioMin"] = f"{min(r['ratio'] for r in NEW):.2f}" if NEW else "--"
    M["dmNewRatioMax"] = f"{max(r['ratio'] for r in NEW):.2f}" if NEW else "--"
    reg = [r for r in lo if r["dir"][:3] == "099"]
    M["dmRegN"] = str(len(reg)); M["dmRegHeld"] = str(sum(r["share"]["interaction"] > 0 for r in reg))
    v2 = json.load(open(P("prereg", "speed_limit_v2.json")))["models"]["gpt-oss-120b"]["rows"]
    panel = json.load(open(P("prereg", "panel_099.json")))["hosts"]
    for C, lab, nm in ((14, "gpt-oss 11%", "Low"), (32, "gpt-oss 25%", "Mid")):
        rs = v2[str(C)]["reads_per_token"]["exact"]
        q = [h["cells"][lab]["reads"]["both3p"] / rs for h in panel if lab in h["cells"]]
        M[f"dmPrefReads{nm}Min"] = f"{min(q):.2f}"; M[f"dmPrefReads{nm}Max"] = f"{max(q):.2f}"
    M["dmSlowRatioMax"] = f"{max(r['ratio'] for r in lo if r['ratio'] < FAST):.2f}"
    M["dmFastRatioMin"] = f"{min(r['ratio'] for r in lo if r['ratio'] >= FAST):.2f}"
    json.dump(dict(T_gpu_median=tmed, T_gpu_range=[tlo, thi], fast_ratio=FAST, machines=out),
              open(P("prereg", "decomp_measured.json"), "w"), indent=1)
    with open(P("paper", "wsg_dm.tex"), "w") as f:
        f.write("% generated by scripts/decomp_measured.py\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    figure(R, P("paper", "figs", "decomp_measured.pdf"), NEW)
    table(M, bool(NEW))
    for k in sorted(M):
        print(k, M[k])


if __name__ == "__main__":
    main()
