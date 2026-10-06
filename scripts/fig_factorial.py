"""The measured factorial of foresight in the engine (jobs 095-097), as a figure: per cell, the speed of each state
relative to the online policy on the same host, one marker per host (O3: job 095; O4, O5: job 096; with job 097's
learned order on its hosts when present).

    python scripts/fig_factorial.py        -> paper/figs/factorial.pdf

Rows (bottom to top): the ways of spending, or not having, foresight. foa: the online policy with each admission read
once (no foresight); learned: the same with the learned order (job 097); bypass: MIN with bypass, admissions served by
the CPU and then copied (two reads); Belady 2r: Belady prefetch, unpaced (admits what MIN bypasses; two reads);
Belady 1r: the same admissions, each read once (nb2); fetch: MIN with bypass, each admission fetched in the step (one
read); fetch + prefetch: fetch with the paced single-read prefetch (lead 3).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731

# validated categorical slots 1-3 (dataviz palette, light surface), with marker shape as secondary encoding
HOSTS = [("095", "O3", "#2a78d6", "o"), ("096a", "O4", "#eb6834", "s"), ("096b", "O5", "#1baf7a", "^"),
         ("097a", "O4 (097)", "#eb6834", "D"), ("097b", "O5' (097)", "#1baf7a", "v")]
ROWS = [("foa", "single read, online"), ("learned", "learned order"), ("bypass", "MIN, serve-then-copy"),
        ("hitopt", "Belady, two reads"), ("hitoptp", "Belady, two reads, paced"), ("nb2", "Belady, one read"), ("fetch", "MIN, fetch (one read)"),
        ("both3p", "MIN, fetch + prefetch")]
CELLS = ["gpt-oss 11%", "gpt-oss 25%", "gpt-oss 40%", "Qwen3 12.5%", "Qwen3 25%", "Qwen3 43.75%"]
HOSTBOUND = {"gpt-oss 11%", "gpt-oss 25%", "Qwen3 12.5%", "Qwen3 25%"}


def main():
    data = {}
    for job, name, col, mk in HOSTS:
        p = P("prereg", f"foresight_{job}.json")
        if os.path.exists(p):
            data[job] = {c["label"]: c for c in json.load(open(p))["cells"]}
    rows = [r for r in ROWS if any(r[0] in c["runs"] for d in data.values() for c in d.values())]
    fig, axes = plt.subplots(1, 6, figsize=(7.1, 2.75), sharey=True)
    plt.rcParams.update({"font.size": 7})
    for ax, lab in zip(axes, CELLS):
        ax.axvline(1.0, color="#52514e", lw=0.8, zorder=1)
        for y in range(len(rows)):
            ax.axhline(y, color="#e4e3df", lw=0.5, zorder=0)
        present = [h for h in HOSTS if h[0] in data]
        for k, (job, name, col, mk) in enumerate(present):
            c = data.get(job, {}).get(lab)
            if not c:
                continue
            for y, (key, _) in enumerate(rows):
                if key in c["runs"]:
                    r = c["runs"][key]["ratio_to_base"]
                    off = (k - (len(present) - 1) / 2) * 0.15
                    ax.plot([r[1], r[2]], [y + off, y + off], color=col, lw=1.0, zorder=2)
                    ax.plot(r[0], y + off, mk, ms=4.2, mfc=col, mec="#fcfcfb", mew=0.6, zorder=3)
        ax.set_title(lab + ("$^\\dagger$" if lab in HOSTBOUND else ""), fontsize=7)
        ax.set_xlim(0.45, 1.9)
        ax.set_xticks([0.5, 1.0, 1.5])
        ax.tick_params(labelsize=6, length=2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("#b5b4ae"); ax.spines["bottom"].set_color("#b5b4ae")
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[1] for r in rows], fontsize=6.5)
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    fig.text(0.6, 0.015, "speed relative to the online policy on the same host (95% interval)", ha="center", fontsize=7)
    handles = [Line2D([0], [0], marker=mk, color=col, lw=0, ms=5, mec="#fcfcfb", label=name)
               for job, name, col, mk in HOSTS if job in data]
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), fontsize=6.5, frameon=False, bbox_to_anchor=(0.995, 1.02))
    fig.subplots_adjust(left=0.165, right=0.995, top=0.84, bottom=0.15, wspace=0.12)
    os.makedirs(P("paper", "figs"), exist_ok=True)
    fig.savefig(P("paper", "figs", "factorial.pdf"))
    fig.savefig(P("paper", "figs", "factorial.png"), dpi=200)
    print("hosts:", list(data), "rows:", [r[0] for r in rows])


if __name__ == "__main__":
    main()
