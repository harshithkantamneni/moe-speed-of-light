"""Figure: claimed speed-up vs normalized speed-up (over the predicted equal-memory llama.cpp baseline), adjudicated
audit rows, with the band.

    python scripts/fig_audit.py prereg/audit/audit.json paper/figs/audit.pdf
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main(src, out):
    A = json.load(open(src))
    adj = [r for r in A["rows"] if not any(l.startswith("not adjudicated") for l in r["labels"])]
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "font.serif": ["Latin Modern Roman"], "mathtext.fontset": "cm", "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for r in adj:
        x = r["claimed_speedup"]
        if not x:
            continue
        lo, mid, hi = r["normalized_speedup"]
        weak = "weak baseline" in r["labels"]
        surv = "gain survives" in r["labels"]
        col = "#1f6f8b" if surv else "#c0392b"
        mk = "o" if weak else "s"
        ax.errorbar([x], [mid], yerr=[[mid - lo], [hi - mid]], fmt=mk, ms=3.8, color=col, mfc=col if weak else "white",
                    mew=0.9, elinewidth=0.6, capsize=1.5, alpha=0.9)
    xs = np.array([1.0, 12])
    ax.plot(xs, xs, ls="--", lw=0.7, color="0.4")
    ax.axhline(1.0, lw=0.7, color="0.2")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1.0, 12); ax.set_ylim(0.07, 12)
    ticks = [1, 2, 4, 8]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t}×" for t in ticks])
    ax.set_yticks([0.125, 0.25, 0.5, 1, 2, 4, 8]); ax.set_yticklabels(["0.125×", "0.25×", "0.5×", "1×", "2×", "4×", "8×"])
    ax.minorticks_off()
    ax.set_xlabel("claimed speed-up (over the paper's baseline)")
    ax.set_ylabel("normalized speed-up $S_n$")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color="#c0392b", ms=3.8, label="weak baseline, not established"),
         Line2D([], [], marker="o", ls="", color="#1f6f8b", ms=3.8, label="weak baseline, survives"),
         Line2D([], [], marker="s", ls="", color="#1f6f8b", mfc="white", ms=3.8, label="baseline at strength, survives"),
         Line2D([], [], marker="s", ls="", color="#c0392b", mfc="white", ms=3.8, label="other, not established")]
    ax.legend(handles=h, loc="upper left", frameon=False, fontsize=6.3, handletextpad=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
