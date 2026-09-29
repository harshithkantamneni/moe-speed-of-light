"""Figure: where the seconds go (prereg/homepc/gap_decomposition_072.json), one stacked bar per budget.

    python scripts/fig_gap.py --out figures/gap_decomposition.pdf
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ["#2b6cb0", "#90cdf4", "#e53e3e", "#f6ad55", "#68d391", "#b794f4", "#a0aec0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="prereg/homepc/gap_decomposition_072.json")
    ap.add_argument("--out", default="figures/gap_decomposition.pdf")
    a = ap.parse_args()
    d = json.load(open(a.json))
    rows = []
    for C, v in d["budgets"].items():
        for mode in ("deployed", "deployed+FETCH"):
            if mode in v:
                rows.append((f"C = {C}" + (" + FETCH" if "FETCH" in mode else ""), v[mode]))
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(rows) + 2.0))
    labels = [k for k, _ in rows[0][1]["steps_ms"]]
    for i, (name, v) in enumerate(rows):
        left = 0.0
        for j, (k, ms) in enumerate(v["steps_ms"]):
            ax.barh(i, ms, left=left, color=COLORS[j], edgecolor="white", label=k if i == 0 else None)
            if ms > 0.9:
                ax.text(left + ms / 2, i, f"{ms:.1f}", ha="center", va="center", fontsize=7, color="black")
            left += ms
        ax.text(left + 0.2, i, f"{v['measured_tok_s']:.0f} tok/s", va="center", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _ in rows])
    ax.invert_yaxis()
    ax.set_xlabel("ms per token (gpt-oss-120b, RTX 5090 + Ryzen 9 7900, own text)")
    ax.legend(fontsize=7, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    ax.set_xlim(0, max(sum(ms for _, ms in v["steps_ms"]) for _, v in rows) * 1.18)
    fig.tight_layout()
    fig.savefig(a.out)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=150)


if __name__ == "__main__":
    main()
