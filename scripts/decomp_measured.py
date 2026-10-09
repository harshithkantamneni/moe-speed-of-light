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


def rows_cpu(tlo):
    """job 114's valid machines: the 2x2 on the CPU-only read path (the in-step fetches off) beside the deployed path,
    registered; shares of each machine's gap (base - Eq. (1)), the CPU-only parts from scripts/job114.py"""
    p = P("prereg", "job114.json")
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
            if "parts" not in c:
                continue
            t, eq1 = c["t"], c["eq1"]; gap = t["base"] - eq1
            share = dict(c["parts"])
            # the parts of what is left are the same on both paths: T_GPU, the oracle's extra reads, the residual
            extra = (c["reads"]["both3p"] - rstar[C]) * S / (float(h["B_host"]) * 1e9) * 1e3
            share["tgpu"] = tlo / gap; share["extra"] = extra / gap
            share["resid"] = (t["both3p"] - eq1 - tlo - extra) / gap
            out.append(dict(dir=h["dir"], C=C, ratio=float(h["ratio"]), cpu=h.get("cpu", ""), t=t, eq1=eq1, gap=gap,
                            share=share, inter_ms=c.get("inter_ms"), inter_ms_ci=c.get("inter_ms_ci")))
    return out


def eq3_left(d, C, t, eq1, tgpu):
    """on one machine: Eq. (3), the ordered bound (CPU reads in series with the GPU's non-expert work, link copies free
    to go ahead; the probe's best CPU and link readings), and the share of the gap to it that the read-ahead oracle
    leaves: (read-ahead - Eq. 3) / (deployed - Eq. 3)"""
    from scripts.fig_decomp import best_rates, dep_bound
    f = os.path.join(RES, d, "concur.txt")
    if not os.path.exists(f):
        return None, None
    bc, bp = best_rates(open(f).read())
    rs = {14: 38.315364583333334, 32: 15.340364583333333}[int(C)] * S
    eq3 = dep_bound(eq1, rs, bc, bp, tgpu)
    return eq3, (t["both3p"] - eq3) / (t["base"] - eq3)


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


def figure(R, path, NEW=(), CPU=()):
    """each machine's staircase relative to its deployed cache: the deployed read path (panel and job 109's machines,
    grey; slow links orange) and the CPU-only read path (job 114's machines, blue), with medians; squares: Dep-1R"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    labels = ["deployed", "MIN's set,\nread twice", "MIN-1R", "read-ahead\noracle", "Eq. (2)", "Eq. (1)"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    grey, dark, orange, blue, dblue = "#a9b4c2", "#1b2a3b", "#eb6834", "#2a78d6", "#0b3f8a"
    X = [0, 0.92, 2, 3, 4, 5]
    for ax, (C, lab) in zip(axes, ((14, "gpt-oss 11%"), (32, "gpt-oss 25%"))):
        A, B = [], []
        for r in [x for x in R if x["C"] == C] + [x for x in NEW if x["C"] == C]:
            t = r["t"]
            eq2 = r.get("eq2", r["eq1"] + TLO[0])
            ya = [1.0, t["bypass"] / t["base"], t["fetch"] / t["base"], t["both3p"] / t["base"], eq2 / t["base"], r["eq1"] / t["base"]]
            fast = r["ratio"] >= FAST
            col = grey if fast else orange
            ax.plot(X, ya, color=col, lw=0.7, alpha=0.8, zorder=2 if fast else 3)
            ax.plot([1.08], [t["foa"] / t["base"]], marker="s", ms=2.4, mfc="none", mec=col, mew=0.6, lw=0, zorder=3)
            if fast:
                A.append(ya); B.append(t["foa"] / t["base"])
        if A:
            ax.plot(X, np.median(np.array(A), axis=0), color=dark, lw=1.9, marker="o", ms=3.2, zorder=5)
            ax.plot([1.08], [float(np.median(B))], marker="s", ms=3.6, mfc="white", mec=dark, mew=1.0, lw=0, zorder=6)
        Z, ZB = [], []
        for r in (x for x in CPU if x["C"] == C):
            t = r["t"]
            yz = [1.0, t["bypass0"] / t["base"], t["fetch"] / t["base"], t["both3p"] / t["base"], (r["eq1"] + TLO[0]) / t["base"], r["eq1"] / t["base"]]
            ax.plot(X, yz, color=blue, lw=0.8, alpha=0.75, zorder=4)
            ax.plot([1.08], [t["foa0"] / t["base"]], marker="s", ms=2.4, mfc="none", mec=blue, mew=0.6, lw=0, zorder=4)
            ax.plot([-0.08], [t["base0"] / t["base"]], marker="D", ms=2.2, mfc="none", mec=blue, mew=0.6, lw=0, zorder=4)
            Z.append(yz); ZB.append(t["foa0"] / t["base"])
        if Z:
            ax.plot(X, np.median(np.array(Z), axis=0), color=dblue, lw=1.9, marker="o", ms=3.2, zorder=6)
            ax.plot([1.08], [float(np.median(ZB))], marker="s", ms=3.6, mfc="white", mec=dblue, mew=1.0, lw=0, zorder=6)
        ax.set_xticks(range(6)); ax.set_xticklabels(labels, fontsize=6.3)
        ax.set_title(lab, fontsize=8)
        ax.grid(axis="y", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=6.5, length=2)
        ax.set_ylim(0.2, 1.25)
    axes[0].set_ylabel("time per token / deployed", fontsize=7)
    nf = sum(1 for r in list(R) + list(NEW) if r["C"] == 14 and r["ratio"] >= FAST)
    ns = sum(1 for r in R if r["C"] == 14 and r["ratio"] < FAST)
    h = [Line2D([0], [0], color=grey, lw=0.8, label=f"fetch table on: each machine, link/CPU $\\geq$ {FAST} ({nf})"),
         Line2D([0], [0], color=dark, lw=1.9, marker="o", ms=3.2, label="fetch table on: median"),
         Line2D([0], [0], color=orange, lw=0.8, label=f"link/CPU $<$ {FAST} ({ns})")]
    if CPU:
        nz = len({r["dir"] for r in CPU if r["C"] == 14})
        h += [Line2D([0], [0], color=blue, lw=0.8, label=f"in-step fetches off: each machine ({nz})"),
              Line2D([0], [0], color=dblue, lw=1.9, marker="o", ms=3.2, label="in-step fetches off: median"),
              Line2D([0], [0], color=dark, lw=0, marker="s", ms=3.4, mfc="white", label="the deployed set, read once (Dep-1R)")]
    fig.legend(handles=h, loc="upper center", ncol=3, fontsize=6.2, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, bbox_inches="tight"); fig.savefig(path.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")


def table(M, has_new=False, has_cpu=False):
    # each row: its key on the deployed path (panel, new machines) and on the CPU-only path (job 114)
    meas = [("setalone", "setalone0", "What is cached: MIN's set, read twice (\\MinTwo)$^\\ast$"),
            ("oncealone", "oncealone0", "How it is read: the deployed set, read once (\\DepOne)"),
            ("together", "together0", "Both: MIN's set, read once (\\MinOne)"),
            ("ahead", "ahead", "Then the read-ahead oracle"),
            ("left", "left", "Left after all three"),
            (None, "table", "The in-step fetches themselves$^\\dagger$")]
    attr = [("tgpu", "tgpu", "the GPU's non-expert time ($T_{\\text{GPU}}$)"),
            ("extra", "extra", "the oracle's reads beyond MIN's, at $B_{\\mathrm{host}}$"),
            ("resid", "resid", "residual")]

    def cell(pre, k, nm):
        if k is None:
            return "--"
        K = k[0].upper() + k[1:]
        if pre == "dmCpu":
            K = K.replace("0", "Zero")
        else:
            K = k.capitalize()
        if pre + K + nm not in M:
            return "--"
        return f"{M[pre + K + nm]} [{M[pre + K + nm + 'Lo']}, {M[pre + K + nm + 'Hi']}]"
    with open(P("paper", "tab_dm.tex"), "w") as f:
        f.write("% generated by scripts/decomp_measured.py\n\\begin{table*}[t]\\centering\\footnotesize\n")
        f.write("\\caption{The deployed cache's gap to \\cref{eq:limit}, split by oracles in the engine (\\cref{fig:staircase}): mean "
                "share of the gap, in percent, with 95\\% $t$-intervals over machines. \\emph{Panel}: the " + M["dmMachines"] + " machines "
                "that ran every state before the registered tests (exploratory). " + ("\\emph{New}: the " + M["dmNewN"] + " machines rented for the registered test "
                "(population and predictions fixed before any started; the deadline cut one 25\\% round). " if has_new else "")
                + ("\\emph{Fetches off}: the machines of the registered tests of \\cref{tab:job114}, whose arms ran with the in-step "
                   "fetches off; their shares are of each machine's gap with its fetch table, like the other columns. " if has_cpu else "")
                + "$^\\ast$On the deployed path (Panel, New) the in-step fetches displace MIN's set, so this row measures MIN's "
                "schedule with the fetch table, not the set held; in the fetches-off column it measures the set held. "
                + ("$^\\dagger$The deployed cache with its fetch table against without it. " if has_cpu else "")
                + "\\emph{Both}, \\emph{then the read-ahead oracle} and \\emph{left} sum to the gap"
                + (" (in the fetches-off column, with $^\\dagger$)" if has_cpu else "") + ". The last three rows split \\emph{left} by "
                "attribution: $T_{\\text{GPU}}$ (" + M["dmTgpu"] + "\\,ms) and the read-ahead oracle's reads beyond $R^\\star$ at "
                "$B_{\\mathrm{host}}$; the residual is what they leave (\\cref{app:limits} gives the probe sensitivity).}\\label{tab:gap}\n")
        groups = []
        if has_cpu:   # the fetches-off machines first: there the rows mean what they say
            groups.append(("dmCpu", "Fetches off", {"Low": M.get("dmCpuNLow", ""), "Mid": M.get("dmCpuNMid", "")}))
        groups.append(("dmAll", "Panel", {"Low": M["dmMachines"], "Mid": M["dmMachines"]}))
        if has_new:
            groups.append(("dmNew", "New", {"Low": M["dmNewNLow"], "Mid": M["dmNewNMid"]}))
        g = len(groups)
        f.write("\\setlength\\tabcolsep{3.5pt}\\resizebox{\\textwidth}{!}{\\begin{tabular}{@{}l" + "r" * (2 * g) + "@{}}\\toprule\n")
        f.write(" & \\multicolumn{" + str(g) + "}{c}{gpt-oss 11\\%} & \\multicolumn{" + str(g) + "}{c}{gpt-oss 25\\%} \\\\"
                "\\cmidrule(lr){2-" + str(1 + g) + "}\\cmidrule(l){" + str(2 + g) + "-" + str(1 + 2 * g) + "}\n")
        if has_cpu:
            on = g - 1
            f.write(" & " + " & ".join(["", "\\multicolumn{" + str(on) + "}{c}{Fetch table on}"] * 2) + " \\\\\n")
        hdr = []
        for nm in ("Low", "Mid"):
            for pre, lab, n in groups:
                hdr.append(f"{lab} ({n[nm]})")
        f.write("Part of the gap & " + " & ".join(hdr) + " \\\\\\midrule\n")
        f.write("\\multicolumn{" + str(1 + 2 * g) + "}{@{}l}{\\emph{Measured}} \\\\\n")
        for item in meas + [None] + attr:
            if item is None:
                f.write("\\addlinespace\\multicolumn{" + str(1 + 2 * g) + "}{@{}l}{\\emph{Attributed: of what is left}} \\\\\n")
                continue
            kd, kc, text = item
            if kd is None and not has_cpu:
                continue
            row = []
            for nm in ("Low", "Mid"):
                for pre, _, _ in groups:
                    row.append(cell(pre, kc if pre == "dmCpu" else kd, nm))
            f.write("\\quad " + text + " & " + " & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\\end{tabular}}\\end{table*}\n")


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
    # sensitivity to the probe: every machine's B_host raised by 10% and by 22% (the most the engine's own reads imply
    # above a probe, Table robust), so Eq. (1) falls by the same factor; the change in the shares, in points
    for C, nm in ((14, "Low"), (32, "Mid")):
        sel = [r for r in R + NEW if r["C"] == C and r["ratio"] >= FAST]
        for f, tag in ((1.10, "Ten"), (1.22, "TwentyTwo")):
            for k, num in (("Together", lambda t, e: t["base"] - t["fetch"]), ("Left", lambda t, e: t["both3p"] - e)):
                d = []
                for r in sel:
                    t, e = r["t"], r["eq1"]
                    d.append(num(t, e / f) / (t["base"] - e / f) - num(t, e) / (t["base"] - e))
                M[f"dmProbe{tag}{k}{nm}Max"] = f"{100 * max(abs(x) for x in d):.0f}"
    # slow links: the gap measured to Eq. (3), the ordered bound, instead of Eq. (1)
    for C, nm in ((14, "Low"), (32, "Mid")):
        sel = [r for r in R if r["C"] == C]
        for grp, rows_ in (("Slow", [r for r in sel if r["ratio"] < FAST]), ("Fast", [r for r in sel if r["ratio"] >= FAST])):
            v = [eq3_left(r["dir"], C, r["t"], r["eq1"], tlo)[1] for r in rows_]
            v = [x for x in v if x is not None]
            if v:
                M[f"dmEqThreeLeft{grp}{nm}Min"] = pct(min(v)); M[f"dmEqThreeLeft{grp}{nm}Max"] = pct(max(v))
                M[f"dmEqThreeLeft{grp}{nm}Med"] = pct(float(np.median(v)))
        p111 = P("prereg", "job111.json")
        if os.path.exists(p111):
            v = []
            for h in json.load(open(p111))["hosts"]:
                c = h["cells"].get(str(C), {})
                if h.get("valid") and c.get("eq1") and all(k in c.get("t", {}) for k in ("base", "both3p")):
                    v.append(eq3_left(h["dir"], C, c["t"], c["eq1"], tlo)[1])
            v = [x for x in v if x is not None]
            if v:
                M[f"dmEqThreeLeftCard{nm}Min"] = pct(min(v)); M[f"dmEqThreeLeftCard{nm}Max"] = pct(max(v))
    # job 114: the 2x2 on the CPU-only path (registered), its shares as dz* macros (scripts/job114.py writes the per-host
    # ratios); the attributed parts of what is left are computed here, with the same T_GPU as the other columns
    CPU = rows_cpu(tlo)
    for C, nm in ((14, "Low"), (32, "Mid")):
        sel = [r for r in CPU if r["C"] == C]
        M[f"dmCpuN{nm}"] = str(len(sel))
        for k in ("table", "setalone0", "oncealone0", "together0", "interaction0", "ahead", "left", "tgpu", "extra", "resid"):
            v = [r["share"][k] for r in sel]
            if not v:
                continue
            m, a_, b_ = boot(v) if len(v) > 1 else (v[0], v[0], v[0])
            K = (k[0].upper() + k[1:]).replace("0", "Zero")
            M[f"dmCpu{K}{nm}"] = pct(m); M[f"dmCpu{K}{nm}Lo"] = pct(a_); M[f"dmCpu{K}{nm}Hi"] = pct(b_)
        if sel:
            out["Cpu" + nm] = [dict(dir=r["dir"], ratio=r["ratio"], cpu=r["cpu"], times=r["t"], eq1=r["eq1"], share=r["share"])
                               for r in sel]
    M["dmCpuN"] = str(len({r["dir"] for r in CPU}))
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
    figure(R, P("paper", "figs", "decomp_measured.pdf"), NEW, CPU)
    table(M, bool(NEW), bool(CPU))
    from scripts.tabnote import split_caption
    split_caption(P("paper", "tab_dm.tex"))   # short caption, the rest as a note below the table
    for k in sorted(M):
        print(k, M[k])


if __name__ == "__main__":
    main()
