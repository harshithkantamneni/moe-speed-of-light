"""Figure: where the seconds go, order-independent (prereg/shapley_gap.json): one stacked bar per Table 1 cell, the
speed limit followed by each fix's Shapley value (its marginal saving averaged over all 120 orders of the fixes), in
the style and size of paper/figs/gap.pdf (scripts/gap_listingb.py --fig). The whisker inside the foresight segment is
the range of foresight's marginal over the orders. Default variant: exact_max (the exact per-layer optimum of
speed_limit_v2.json at B_host = 87.5 GB/s); `main` is the job 084 optimum of gap_listingb.json.

    python scripts/fig_shapley.py [--json prereg/shapley_gap.json] [--variant exact_max] [--out paper/figs/shapley.pdf]
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLS = ["#2b6cb0", "#90cdf4", "#f6ad55", "#e53e3e", "#9f7aea", "#718096"]   # as gap.pdf, one per segment
LABELS = {"overlap": "no overlap", "foresight": "no foresight", "policy and read paths": "policy and read paths",
          "host work": "host work", "GPU efficiency": "GPU below datasheet (net)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="prereg/shapley_gap.json")
    ap.add_argument("--variant", default="exact_max")
    ap.add_argument("--out", default="paper/figs/shapley.pdf")
    a = ap.parse_args()
    d = json.load(open(a.json))
    var = d["variants"][a.variant]
    fixes = var["fixes"]
    rows = var["cells"]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for i, r in enumerate(rows[::-1]):
        left = r["limit_ms"]
        ax.barh(i, left, color=COLS[0], height=0.62, label="speed limit" if i == 0 else None)
        for j, k in enumerate(fixes):
            v = r["shapley_ms"][k]
            ax.barh(i, v, left=left, color=COLS[j + 1], height=0.62, label=LABELS.get(k, k) if i == 0 else None)
            if k == "foresight":
                lo, hi = r["marginal_min_ms"][k], r["marginal_max_ms"][k]
                ax.plot([left + lo, left + hi], [i, i], color="black", lw=0.6, solid_capstyle="butt")
                for x in (left + lo, left + hi):
                    ax.plot([x, x], [i - 0.14, i + 0.14], color="black", lw=0.6)
            left += v
        ax.text(left + 0.3, i, f"{1e3 / r['measured_ms']:.0f} tok/s", va="center", fontsize=6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["cell"] for r in rows[::-1]], fontsize=6.5)
    ax.set_xlabel("ms per token", fontsize=7)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.legend(fontsize=5.5, frameon=False, loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=3, columnspacing=0.8, handlelength=1.2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlim(0, max(r["measured_ms"] for r in rows) * 1.22)
    fig.tight_layout()
    fig.savefig(a.out)


if __name__ == "__main__":
    main()
