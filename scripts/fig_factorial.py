"""The measured factorial of foresight in the engine (job 095), as a figure: per cell, the speed of each state as a
percentage of this host's limit, beside the model's values for the states it prices.

    python scripts/fig_factorial.py        -> paper/figs/factorial.pdf

States (bars, left to right): the online policy; fetch (MIN with bypass, admitted misses fetched into their slot: one
read, serialised); both (fetch and the scheduled single-read prefetch, unpaced); paced (both on the paced copy path with
a three-step lead); hit-opt. (the hit-optimal oracle with background admissions: two reads per admitted expert). Marks:
the model's foresight-only state v(F) and its foresight-and-overlap state v(O,F) for the cell.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731


def main():
    d = json.load(open(P("prereg", "foresight_095.json")))
    order = [("gpt-oss-120b", 14), ("gpt-oss-120b", 32), ("gpt-oss-120b", 51), ("qwen3-30b-a3b-bf16", 16), ("qwen3-30b-a3b-bf16", 32), ("qwen3-30b-a3b-bf16", 56)]
    names = {("gpt-oss-120b", 14): "gpt-oss\n11%", ("gpt-oss-120b", 32): "gpt-oss\n25%", ("gpt-oss-120b", 51): "gpt-oss\n40%",
             ("qwen3-30b-a3b-bf16", 16): "Qwen3\n12.5%", ("qwen3-30b-a3b-bf16", 32): "Qwen3\n25%", ("qwen3-30b-a3b-bf16", 56): "Qwen3\n43.75%"}
    cells = {(c["model"], c["C"]): c for c in d["cells"]}
    states = [("base", "online", "#9a9a9a"), ("fetch", "fetch (one read, serialised)", "#5b9bd5"), ("both2", "fetch + prefetch (one read)", "#1f5fa8"),
              ("both3p", "the same, paced, lead 3", "#0b2f5e"), ("hitopt", "hit-optimal (two reads)", "#e39a6b")]
    fig, ax = plt.subplots(figsize=(3.4, 2.55))
    w = 0.15
    for i, key in enumerate(order):
        c = cells[key]
        r = c["runs"]
        mt = c["model_terms"]
        for j, (k, _, col) in enumerate(states):
            if k not in r:
                continue
            x = i + (j - 2) * w
            ax.bar(x, 100 * r[k]["frac_of_limit"], width=w * 0.92, color=col, linewidth=0)
        # the model's states for this cell, as fractions of the limit
        for v, mk in ((mt["v_F_ms"], "_"), (mt["v_OF_ms"], "x")):
            ax.plot([i], [100 * mt["limit_ms"] / v], marker=mk, color="k", ms=7 if mk == "_" else 4, mew=1.2, ls="")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([names[k] + ("$^{\\dagger}$" if cells[k]["model_terms"]["c_at_limit"] <= 0 else "") for k in order], fontsize=6)
    ax.set_ylabel("% of this host's limit", fontsize=6.5)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    handles = [Patch(color=col, label=lab) for _, lab, col in states]
    handles += [Line2D([], [], marker="_", color="k", ls="", ms=7, mew=1.2, label="model: foresight alone, v(F)"),
                Line2D([], [], marker="x", color="k", ls="", ms=4, mew=1.2, label="model: foresight and overlap, v(O,F)")]
    ax.legend(handles=handles, fontsize=5.2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=0.9, handletextpad=0.4, handlelength=1.2, borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(P("paper", "figs", "factorial.pdf"))
    print("wrote paper/figs/factorial.pdf")


if __name__ == "__main__":
    main()
