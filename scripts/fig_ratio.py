"""Ours / FreeToken against the machine's CPU-to-PCIe read ratio, every same-machine comparison of the Table 1 protocol
(30 AIME-25 problems, 256 tokens, greedy, session, equal GPU expert memory). Job 085's confounded i9-14900K rows are
replaced by job 088's fair rerun (FreeToken's CPU executor on the performance cores).

x: FreeToken's own probe on each machine (ft bench bw ceilings: cpu_stream_read_gbs / pcie_linear_h2d_gbs), the same
   tool on every machine. y: ours / FreeToken's better backend, mean speeds; filled markers use the law's FETCH table
   computed on that machine, hollow ones the fixed table. Confirmation launch where the job had one.

    python scripts/fig_ratio.py [--out paper/figs/ratio.pdf] [--json prereg/ratio_points.json]
"""
import argparse
import glob
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")


def probe(job):
    f = glob.glob(f"{R}/{job}/GPU-*.json")
    if not f:
        return None
    c = json.load(open(f[0]))["ceilings"]
    return c["cpu_stream_read_gbs"], c["pcie_linear_h2d_gbs"]


def load(name):
    p = os.path.join(ROOT, "prereg", name)
    return json.load(open(p)) if os.path.exists(p) else None


def points():
    pts = []

    def add(job, host, model, budget, table, ratio, src):
        pr = probe(job)
        if pr and ratio:
            pts.append(dict(job=job, host=host, model=model, budget=budget, table=table, ratio=ratio, cpu=pr[0], pcie=pr[1],
                            x=pr[0] / pr[1], source=src))
    # job 076 (9950X): gpt-oss, fixed table, launches 2-3 (prereg/table_outcome_076.md)
    for b, r in (("11%", 1.205), ("25%", 1.217), ("40%", 1.118)):
        add("076_table_5090@vast", "9950X", "gpt-oss-120b", b, "fixed", r, "table_outcome_076.md")
    # job 077 (9950X3D, another listing): gpt-oss, fixed table, launch 1 (prereg/competitors_outcome_077.md)
    for b, o, f in (("11%", 56.0, 45.8), ("25%", 92.0, 76.0), ("40%", 133.0, 119.5)):
        add("077_competitors_gptoss@vast", "9950X3D (other)", "gpt-oss-120b", b, "fixed", o / f, "competitors_outcome_077.md")
    q = load("qwen3_078.json")
    for c in q["comparisons"]:
        add("078_qwen3_4way@vast", "9950X", "Qwen3-30B-A3B", c["budget"], "fixed", c["ours_over_ft"][0], "qwen3_078.json")
    add("079_qwen3_profile@vast", "9950X3D (A)", "Qwen3-30B-A3B", "43.75%", "fixed", 0.991, "qwen3_profile_outcome_079.md")
    s = load("split_080.json")
    rt = s["ratios"]
    add("080_fetch_split@vast", "9950X3D (B)", "Qwen3-30B-A3B", "43.75%", "fixed", rt["ours_C56_cur / ft_offload_r0.4375 (L2)"][0], "split_080.json")
    add("080_fetch_split@vast", "9950X3D (B)", "Qwen3-30B-A3B", "43.75%", "law", rt["ours_C56_law / ft_offload_r0.4375 (L2)"][0], "split_080.json")
    add("080_fetch_split@vast", "9950X3D (B)", "Qwen3-30B-A3B", "25%", "law", rt["C32 best ours_C32_law / ft_hybrid_r0.25 (L1)"][0], "split_080.json")
    h = load("headline_081.json")
    for r in h["rows"]:
        add("081_headline_law@vast", "9950X3D (B)", r["model"], r["budget"], "law", r["ours_over_ft_L2"][0], "headline_081.json")
        ftb = max(r["ft_L1"].values())
        add("081_headline_law@vast", "9950X3D (B)", r["model"], r["budget"], "fixed", r["ours_cur_L1"] / ftb, "headline_081.json (launch 1)")
    # job 085's gpt-oss rows (14900K) are confounded: FreeToken's CPU executor spread 23 threads over the P- and E-cores;
    # job 088 reran that machine class with the executor on 8 performance-core threads and replaces them here.
    for name, job, host in (("halfpcie_085b.json", "085b_half_pcie_qwen3@vast", "7900, PCIe 4.0"),
                            ("slowlink_088.json", "088_slowlink_fair@vast", "14900K, slow link, fair")):
        hp = load(name)
        if hp:
            for r in hp["rows"]:
                if "ours_over_ft_L2" in r:
                    add(job, host, r["model"], r["budget"], r["ours_variant"].replace("cur", "fixed"), r["ours_over_ft_L2"][0], name)
    # job 089: Table 1's six cells rerun on a second host (Ryzen 9 9950X, stock-clock card), law table, launch 2
    sc = load("stockclock_089.json")
    if sc:
        for r in sc["rows"]:
            add("089_headline_stockclock@vast", "9950X, stock clock", r["model"], r["budget"], "law", r["ours_over_ft_L2"][0], "stockclock_089.json")
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "figs", "ratio.pdf"))
    ap.add_argument("--json")
    a = ap.parse_args()
    pts = points()
    for p in pts:
        print(f"{p['job'][:3]} {p['host']:16s} {p['model']:14s} {p['budget']:7s} {p['table']:5s} x={p['x']:.2f} "
              f"(CPU {p['cpu']:.1f} / PCIe {p['pcie']:.1f})  ours/FT {p['ratio']:.3f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.9), sharey=True)
    cols = {"11%": "#1f5fa8", "12.5%": "#1f5fa8", "25%": "#2f9e44", "40%": "#c0572b", "43.75%": "#c0572b"}
    for ax, model in zip(axs, ("gpt-oss-120b", "Qwen3-30B-A3B")):
        for p in [p for p in pts if p["model"] == model]:
            ax.scatter(p["x"], p["ratio"], s=16, color=cols[p["budget"]], marker="o",
                       facecolors=cols[p["budget"]] if p["table"] == "law" else "none", linewidths=0.9)
        ax.axhline(1, color="k", lw=0.5, ls=":")
        ax.set_title(model, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("CPU / PCIe read rate", fontsize=6.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("ours / FreeToken", fontsize=6.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", color=c, label=l, markersize=4) for l, c in (("11 or 12.5%", "#1f5fa8"), ("25%", "#2f9e44"), ("40 or 43.75%", "#c0572b"))]
    h += [Line2D([], [], ls="", marker="o", color="k", label="law's table", markersize=4),
          Line2D([], [], ls="", marker="o", color="k", markerfacecolor="none", label="fixed table", markersize=4)]
    fig.legend(handles=h, fontsize=5.2, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.02), columnspacing=0.8, handletextpad=0.2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out)
    if a.json:
        json.dump(pts, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
