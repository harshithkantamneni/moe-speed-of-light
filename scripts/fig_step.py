"""A schematic of one decode step of the deployed cache, layer by layer: where the GPU's own compute (G) and the host
reads sit. Not data: a drawing for Section 2. Writes paper/figs/step.pdf.

    python scripts/fig_step.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BLUE, GREY, ORANGE, LIGHT = "#3a6ea5", "#7a7974", "#d9822b", "#f3d9bf"


def box(ax, x, w, y, col, text, tc="white", hatch=None):
    ax.add_patch(FancyBboxPatch((x, y - 0.32), w, 0.64, boxstyle="round,pad=0.0,rounding_size=0.06", fc=col, ec="white",
                                lw=0.8, hatch=hatch))
    if text:
        ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=5.6, color=tc)


def main():
    fig, ax = plt.subplots(figsize=(3.4, 1.45))
    lanes = {"GPU": 2.0, "CPU": 1.0, "PCIe": 0.0}
    for nm, y in lanes.items():
        ax.text(-0.12, y, nm, ha="right", va="center", fontsize=6.5)
    # layer l: attention and router, resident experts, then the step waits for the misses' host reads
    box(ax, 0.0, 1.7, 2.0, BLUE, "attn + router")
    box(ax, 1.7, 1.2, 2.0, BLUE, "resident")
    ax.add_patch(FancyBboxPatch((2.9, 1.68), 1.7, 0.64, boxstyle="round,pad=0.0,rounding_size=0.06", fc="white",
                                ec="#a8a7a2", lw=0.6, ls="--"))
    ax.text(3.75, 2.0, "waits", ha="center", va="center", fontsize=5.8, color="#52514e")
    box(ax, 4.6, 0.9, 2.0, BLUE, "fetched")
    box(ax, 5.5, 1.7, 2.0, BLUE, "attn + router")
    box(ax, 7.2, 1.0, 2.0, BLUE, "resident")
    box(ax, 1.7, 2.9, 1.0, GREY, "misses: read and run")
    box(ax, 7.2, 1.2, 1.0, GREY, "misses")
    box(ax, 1.7, 1.7, 0.0, GREY, "fetch: 1 read")
    box(ax, 4.6, 3.6, 0.0, ORANGE, "admission copy: 2nd read", tc="black")
    ax.plot([5.5, 5.5], [-0.42, 2.42], color="#52514e", lw=0.5, ls=":")
    ax.text(2.75, 2.55, "layer $l$", ha="center", va="bottom", fontsize=6.0, color="#52514e")
    ax.text(6.85, 2.55, "layer $l{+}1$", ha="center", va="bottom", fontsize=6.0, color="#52514e")
    ax.set_xlim(-0.9, 8.5); ax.set_ylim(-0.5, 2.85)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    out = os.path.join(ROOT, "paper", "figs", "step.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=200)


if __name__ == "__main__":
    main()
