"""Figure: decode tok/s against GPU expert budget on the A10 test run, one panel per model.

Series (fixed categorical order, validated palette): mailbox cache, llama.cpp static layers, helpers with the
static layout, split-graph cache. Speed-of-light bound (where the model has a trace) and all-GPU reference in grays.

    python scripts/fig_eval.py --scored prereg/a10_mb/scored.json --sol prereg/a10_mb/sol.json --out paper/figs/eval.pdf
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = [("mailbox_cache", "mailbox cache", "#2a78d6", "o"),
          ("llama_static", "llama.cpp layers", "#eb6834", "s"),
          ("mailbox_layers", "helpers, layers", "#1baf7a", "^"),
          ("split_cache", "split-graph cache", "#eda100", "D")]
TITLES = {"gpt-oss-20b-MXFP4": "gpt-oss-20b (MXFP4)", "Qwen3-30B-A3B-Instruct-2507-Q4_K_M": "Qwen3-30B-A3B (Q4_K_M)",
          "Qwen3-30B-A3B-Instruct-2507-Q8_0": "Qwen3-30B-A3B (Q8_0)", "gpt-oss-120b-MXFP4": "gpt-oss-120b (MXFP4)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--sol", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = json.load(open(a.scored))["results"]
    sol = json.load(open(a.sol))
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.linewidth": 0.8, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.15), sharey=False)
    for ax, tag in zip(axes, TITLES):
        if tag not in res:
            ax.set_visible(False)
            continue
        bud = sorted(res[tag]["budgets"], key=int)
        xs = list(range(len(bud)))
        pct = [int(q) / 8 * 100 for q in bud]
        for key, lab, col, mk in SERIES:
            ys = [res[tag]["budgets"][q][key]["tok_s"] for q in bud]
            ax.plot(xs, ys, color=col, lw=2, marker=mk, ms=4.5, mec="white", mew=1.2, label=lab, zorder=3,
                    solid_capstyle="round", solid_joinstyle="round")
        if tag in sol and tag in ("gpt-oss-20b-MXFP4", "Qwen3-30B-A3B-Instruct-2507-Q4_K_M"):
            ys = [sol[tag]["budgets"][q]["sol_tok_s"] for q in bud]
            ax.plot(xs, ys, color="#3a3a38", lw=1.2, ls=(0, (1, 1.5)), label="speed-of-light bound", zorder=2)
        if "all_gpu" in res[tag]:
            ax.axhline(res[tag]["all_gpu"]["tok_s"], color="#9a9893", lw=1, zorder=1)
            ax.text(xs[0], res[tag]["all_gpu"]["tok_s"], " all experts on GPU", color=MUTED, fontsize=6.5, va="bottom")
        ax.set_title(TITLES[tag], fontsize=8, color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:g}%" for x in pct])
        ax.set_xlim(-0.35, len(bud) - 0.65)
        ax.set_xlabel("experts on GPU")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("decode tok/s")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
