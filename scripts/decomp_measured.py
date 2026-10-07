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
PARTS = ("setalone", "oncealone", "together", "ahead", "serial", "rest", "left")


def rows():
    ne = sorted(p["nonexpert"] for p in profiles())
    tmed, tlo, thi = float(np.median(ne)), ne[0], ne[-1]
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
            assert abs(parts["together"] + parts["ahead"] + parts["left"] - gap) < 1e-9
            share = {k: v / gap for k, v in parts.items()}
            share["interaction"] = share["together"] - share["setalone"] - share["oncealone"]
            share["serial_lo"], share["serial_hi"] = tlo / gap, thi / gap
            out.append(dict(dir=h["dir"], uuid=u, C=int(C), ratio=float(h["ratio"]), cpu=h.get("cpu", ""), t=t, eq1=eq1,
                            eq2=eq1 + tlo, gap=gap, parts=parts, share=share, bound_share=eq1 / t["base"],
                            pref_over_eq2=t["both3p"] / (eq1 + tlo),
                            closed=(t["base"] - min(t["bypass"], t["fetch"], t["both3p"])) / gap))
    return out, (tmed, tlo, thi)


def boot(v, stat=np.mean, nb=10000):
    """the statistic over machines and its 95% percentile-bootstrap interval, resampling machines"""
    v = np.array(v, float)
    bs = [stat(v[RNG.integers(0, len(v), len(v))]) for _ in range(nb)]
    return float(stat(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def pct(x):
    v = int(round(100 * x))
    return str(v).replace("-", "$-$")


def figure(R, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    labels = ["deployed", "one change\nalone", "MIN's set,\nread once", "and read\nahead", "Eq. (2)", "Eq. (1)"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    blue, dark, orange = "#2a78d6", "#1b3f6b", "#eb6834"
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
    h = [Line2D([0], [0], color=dark, lw=2.0, marker="o", ms=3.4, label="median: MIN's set first (read twice)"),
         Line2D([0], [0], color=dark, lw=1.6, ls=(0, (3, 1.5)), marker="s", ms=3.4, mfc="white", label="median: one read first (deployed set)"),
         Line2D([0], [0], color=blue, lw=0.8, label=f"each machine, link/CPU $\\geq$ {FAST} ({nf})"),
         Line2D([0], [0], color=orange, lw=0.8, label=f"link/CPU $<$ {FAST} ({ns})")]
    fig.legend(handles=h, loc="upper center", ncol=4, fontsize=6.2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, bbox_inches="tight"); fig.savefig(path.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")


def table(M):
    lab = [("setalone", "MIN's set alone, still read twice"),
           ("oncealone", "One read alone, deployed set"),
           ("together", "Both together: MIN's set read once"),
           ("ahead", "Then reading ahead: copies issued a few steps early"),
           ("left", "Left after all three: no oracle removes it")]
    with open(P("paper", "tab_dm.tex"), "w") as f:
        f.write("% generated by scripts/decomp_measured.py\n\\begin{table}[!t]\\centering\\footnotesize\n")
        f.write("\\caption{The deployed cache's gap to \\cref{eq:limit}, split by oracles in the engine (\\cref{fig:staircase}): mean "
                "share of the gap over the " + M["dmFastN"] + " machines whose link-to-CPU ratio is at least " + M["dmFast"] + " (a "
                "subset drawn after the data), with 95\\% intervals over machines. The first two rows each change one thing; the "
                "third changes both. \\emph{Together}, \\emph{ahead} and \\emph{left} sum to the gap. The GPU's non-expert "
                "time $T_{\\text{GPU}}$, which \\cref{eq:demand} puts in series with the reads, equals " + M["dmSerialLowPlo"] + "--"
                + M["dmSerialLowPhi"] + "\\% of the gap at 11\\% and " + M["dmSerialMidPlo"] + "--" + M["dmSerialMidPhi"] + "\\% at 25\\% "
                "over the profiles' range; the prefetching oracle also reads more experts than MIN.}\\label{tab:gap}\n")
        f.write("\\setlength\\tabcolsep{3pt}\\begin{tabular}{@{}p{0.50\\linewidth}rr@{}}\\toprule\nPart of the gap & gpt-oss 11\\% & 25\\% \\\\\\midrule\n")
        for k, text in lab:
            K = k.capitalize()
            f.write(f"{text} & {M['dm' + K + 'Low']}\\% [{M['dm' + K + 'LowLo']}, {M['dm' + K + 'LowHi']}] & "
                    f"{M['dm' + K + 'Mid']}\\% [{M['dm' + K + 'MidLo']}, {M['dm' + K + 'MidHi']}] \\\\\n")
            if k == "oncealone":
                f.write("\\addlinespace\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n")


def main():
    R, (tmed, tlo, thi) = rows()
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
        for k in PARTS + ("interaction",):
            m, a, b = boot([r["share"][k] for r in fast])
            K = k.capitalize()
            M[f"dm{K}{nm}"] = pct(m); M[f"dm{K}{nm}Lo"] = pct(a); M[f"dm{K}{nm}Hi"] = pct(b)
            M[f"dm{K}{nm}Min"] = pct(min(r["share"][k] for r in fast)); M[f"dm{K}{nm}Max"] = pct(max(r["share"][k] for r in fast))
        M[f"dmSerial{nm}Plo"] = pct(np.mean([r["share"]["serial_lo"] for r in fast]))
        M[f"dmSerial{nm}Phi"] = pct(np.mean([r["share"]["serial_hi"] for r in fast]))
        m, a, b = boot([r["closed"] for r in fast])
        M[f"dmClosed{nm}"] = pct(m); M[f"dmClosed{nm}Lo"] = pct(a); M[f"dmClosed{nm}Hi"] = pct(b)
        M[f"dmClosedSlow{nm}Max"] = pct(max(r["closed"] for r in slow))
        M[f"dmPrefEqTwo{nm}Min"] = f"{min(r['pref_over_eq2'] for r in fast):.2f}"
        M[f"dmPrefEqTwo{nm}Max"] = f"{max(r['pref_over_eq2'] for r in fast):.2f}"
        M[f"dmBoundShare{nm}Min"] = pct(min(r["bound_share"] for r in sel)); M[f"dmBoundShare{nm}Max"] = pct(max(r["bound_share"] for r in sel))
        out[nm] = [dict(dir=r["dir"], ratio=r["ratio"], cpu=r["cpu"], times=r["t"], eq1=r["eq1"], share=r["share"]) for r in sel]
    reg = [r for r in lo if r["dir"][:3] in ("096", "099")]
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
    figure(R, P("paper", "figs", "decomp_measured.pdf"))
    table(M)
    for k in sorted(M):
        print(k, M[k])


if __name__ == "__main__":
    main()
