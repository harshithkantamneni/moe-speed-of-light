"""Where the seconds go, measured: the deployed cache's gap to the bound split by the engine's own oracles, machine by
machine, at the two host-bound gpt-oss budgets. Every machine that ran the four states of the factorial
(deployed, MIN read twice, deployed admissions read once, MIN read once) and MIN prefetched, one launch per GPU:

    choice     what is cached: MIN's set instead of the deployed one (Shapley over the 2x2 with the read path)
    twice      how it is read: each admitted expert read once instead of twice (Shapley over the same 2x2)
    late       when it is read: MIN read once, with its copies issued a few steps ahead instead of in the step
    serial     the GPU's non-expert work in series with the reads: Eq. (2) minus Eq. (1), T_GPU
    rest       MIN prefetched minus Eq. (2): what the engine with full foresight still spends above the demand bound

The parts sum to the gap, deployed minus Eq. (1). Times are the engine's per-token means over the problems
(prereg/reanalysis_hosts.json); Eq. (1) is the paper's bound on that machine; T_GPU is the smallest profiled
non-expert GPU time (scripts/sumlaw_paper.py). Writes paper/wsg_dm.tex, prereg/decomp_measured.json and
paper/figs/decomp_measured.pdf.

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
FAST = 0.5          # link-to-CPU ratio at or above which a machine counts as fast-linked
STATES = ("base", "bypass", "foa", "fetch", "both3p")
PARTS = ("choice", "twice", "late", "serial", "rest")


def rows():
    tne = min(p["nonexpert"] for p in profiles())
    out, seen = [], set()
    for h in json.load(open(P("prereg", "reanalysis_hosts.json")))["launches"]:
        u = h.get("gpu_uuid") or h["dir"]
        for C in ("14", "32"):
            c = h["cells"].get(C, {})
            if (u, C) in seen or "limit_ms" not in c or not all(f"ms_{k}" in c for k in STATES):
                continue
            seen.add((u, C))
            t = {k: c[f"ms_{k}"] for k in STATES}
            eq1 = c["limit_ms"]; eq2 = eq1 + tne
            gap = t["base"] - eq1
            parts = dict(choice=0.5 * ((t["base"] - t["bypass"]) + (t["foa"] - t["fetch"])),
                         twice=0.5 * ((t["base"] - t["foa"]) + (t["bypass"] - t["fetch"])),
                         late=t["fetch"] - t["both3p"], serial=tne, rest=t["both3p"] - eq2)
            assert abs(sum(parts.values()) - gap) < 1e-9
            out.append(dict(dir=h["dir"], uuid=u, C=int(C), ratio=float(h["ratio"]), cpu=h.get("cpu", ""), t=t, eq1=eq1, eq2=eq2,
                            gap=gap, parts=parts, share={k: v / gap for k, v in parts.items()},
                            bound_share=eq1 / t["base"], pref_over_eq2=t["both3p"] / eq2,
                            closed=(t["base"] - min(t["bypass"], t["fetch"], t["both3p"])) / gap))
    return out, tne


def boot(v, stat=np.mean, nb=10000):
    """the statistic over machines and its 95% percentile-bootstrap interval, resampling machines"""
    v = np.array(v, float)
    bs = [stat(v[RNG.integers(0, len(v), len(v))]) for _ in range(nb)]
    return float(stat(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def pct(x):
    return f"{100 * x:.0f}".replace("-", "$-$")


def figure(R, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = ["deployed", "MIN,\n2 reads", "MIN,\n1 read", "MIN,\nprefetched", "Eq. (2)", "Eq. (1)"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    for ax, (C, lab) in zip(axes, ((14, "gpt-oss 11%"), (32, "gpt-oss 25%"))):
        sel = [r for r in R if r["C"] == C]
        ys = {}
        for r in sel:
            t = r["t"]
            y = [1.0, t["bypass"] / t["base"], t["fetch"] / t["base"], t["both3p"] / t["base"], r["eq2"] / t["base"], r["eq1"] / t["base"]]
            fast = r["ratio"] >= FAST
            ax.plot(range(6), y, color="#2a78d6" if fast else "#eb6834", lw=0.9, alpha=0.55 if fast else 0.9,
                    marker="o", ms=2.2, zorder=2 if fast else 3)
            ys.setdefault(fast, []).append(y)
        med = np.median(np.array(ys[True]), axis=0)
        ax.plot(range(6), med, color="#1b3f6b", lw=2.2, marker="o", ms=3.5, zorder=4)
        ax.set_xticks(range(6)); ax.set_xticklabels(steps, fontsize=6.5)
        ax.set_title(lab, fontsize=8)
        ax.grid(axis="y", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=6.5, length=2)
        ax.set_ylim(0.2, 1.25)
    axes[0].set_ylabel("time per token / deployed", fontsize=7)
    from matplotlib.lines import Line2D
    nf = sum(1 for r in R if r["C"] == 14 and r["ratio"] >= FAST); ns = sum(1 for r in R if r["C"] == 14 and r["ratio"] < FAST)
    h = [Line2D([0], [0], color="#2a78d6", lw=0.9, marker="o", ms=2.2, label=f"link $\\geq$ {FAST} of CPU rate ({nf} machines)"),
         Line2D([0], [0], color="#1b3f6b", lw=2.2, marker="o", ms=3.5, label="their median"),
         Line2D([0], [0], color="#eb6834", lw=0.9, marker="o", ms=2.2, label=f"link $<$ {FAST} of CPU rate ({ns} machines)")]
    fig.legend(handles=h, loc="upper center", ncol=3, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, bbox_inches="tight"); fig.savefig(path.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")


def table(M):
    lab = dict(choice="What is cached: MIN's set instead of the deployed one",
               twice="How it is read: each admission read once, not twice",
               late="When it is read: copies issued a few steps ahead",
               serial="GPU's non-expert work in series with the reads ($T_{\\text{GPU}}$)",
               rest="Left above \\cref{eq:demand} with full foresight")
    with open(P("paper", "tab_dm.tex"), "w") as f:
        f.write("% generated by scripts/decomp_measured.py\n\\begin{table}[!t]\\centering\\footnotesize\n")
        f.write("\\caption{Where the deployed cache's time beyond \\cref{eq:limit} goes, measured by removing each part in the engine "
                "(\\cref{fig:staircase}): mean share of the gap over the " + M["dmFastN"] + " machines whose link reads at "
                "least " + M["dmFast"] + " of their CPU rate, with 95\\% intervals over machines. The first two parts are split "
                "order-free over the four combinations of what is cached and how it is read; $T_{\\text{GPU}}$ is profiled, the "
                "rest measured.}\\label{tab:gap}\n")
        f.write("\\setlength\\tabcolsep{3pt}\\begin{tabular}{@{}p{0.52\\linewidth}rr@{}}\\toprule\nPart of the gap & gpt-oss 11\\% & 25\\% \\\\\\midrule\n")
        for k in PARTS:
            K = k.capitalize()
            f.write(f"{lab[k]} & {M['dm' + K + 'Low']}\\% [{M['dm' + K + 'LowLo']}, {M['dm' + K + 'LowHi']}] & "
                    f"{M['dm' + K + 'Mid']}\\% [{M['dm' + K + 'MidLo']}, {M['dm' + K + 'MidHi']}] \\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n")


def main():
    R, tne = rows()
    M = {}
    words = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    w = lambda n: words[n] if n < len(words) else str(n)  # noqa: E731
    lo = [r for r in R if r["C"] == 14]
    M["dmMachines"] = str(len(lo)); M["dmFastN"] = str(sum(r["ratio"] >= FAST for r in lo))
    M["dmSlowN"] = w(sum(r["ratio"] < FAST for r in lo)); M["dmFast"] = f"{FAST}"
    M["dmTgpu"] = f"{tne:.1f}"
    out = {}
    for C, nm in ((14, "Low"), (32, "Mid")):
        sel = [r for r in R if r["C"] == C]
        fast = [r for r in sel if r["ratio"] >= FAST]; slow = [r for r in sel if r["ratio"] < FAST]
        for k in PARTS:
            m, a, b = boot([r["share"][k] for r in fast])
            M[f"dm{k.capitalize()}{nm}"] = pct(m); M[f"dm{k.capitalize()}{nm}Lo"] = pct(a); M[f"dm{k.capitalize()}{nm}Hi"] = pct(b)
            M[f"dm{k.capitalize()}{nm}Min"] = pct(min(r["share"][k] for r in fast)); M[f"dm{k.capitalize()}{nm}Max"] = pct(max(r["share"][k] for r in fast))
        m, a, b = boot([r["closed"] for r in fast])
        M[f"dmClosed{nm}"] = pct(m); M[f"dmClosed{nm}Lo"] = pct(a); M[f"dmClosed{nm}Hi"] = pct(b)
        M[f"dmClosedSlow{nm}Max"] = pct(max(r["closed"] for r in slow))
        M[f"dmPrefEqTwo{nm}Min"] = f"{min(r['pref_over_eq2'] for r in fast):.2f}"
        M[f"dmPrefEqTwo{nm}Max"] = f"{max(r['pref_over_eq2'] for r in fast):.2f}"
        M[f"dmPrefEqTwoSlow{nm}Min"] = f"{min(r['pref_over_eq2'] for r in slow):.2f}"
        M[f"dmTwiceSlow{nm}Min"] = pct(min(r["share"]["twice"] for r in slow))
        M[f"dmBoundShare{nm}Min"] = pct(min(r["bound_share"] for r in sel)); M[f"dmBoundShare{nm}Max"] = pct(max(r["bound_share"] for r in sel))
        out[nm] = [dict(dir=r["dir"], ratio=r["ratio"], cpu=r["cpu"], times=r["t"], eq1=r["eq1"], eq2=r["eq2"], share=r["share"]) for r in sel]
    M["dmSlowRatioMax"] = f"{max(r['ratio'] for r in lo if r['ratio'] < FAST):.2f}"
    M["dmFastRatioMin"] = f"{min(r['ratio'] for r in lo if r['ratio'] >= FAST):.2f}"
    json.dump(dict(T_gpu=tne, fast_ratio=FAST, machines=out), open(P("prereg", "decomp_measured.json"), "w"), indent=1)
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
