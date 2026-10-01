"""Figure: expert reads per token of online cache policies relative to Belady's optimum, against the cache's
turnover time C/k, for 9 MoE models at 3 budgets each (from prereg/policy_study.json, scripts/policy_study.py).
One panel, one series per policy (27 model x budget points each), with a line through each policy's median in
four C/k bands. The style and size follow one panel of scripts/fig_foresight.py.

    python scripts/fig_policy.py --out paper/figs/policy.pdf
"""
import argparse
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = [  # key, label, colour, marker
    ("lru", "LRU", "#e53e3e", "o"),
    ("lfu", "LFU", "#f6ad55", "v"),
    ("df0", "decayed frequency ($\\kappa$=0)", "#2b6cb0", "s"),
    ("dfk", "decayed frequency (deployed $\\kappa$)", "#90cdf4", "D"),
    ("arc", "ARC", "#68d391", "^"),
    ("s3fifo", "S3-FIFO", "#b794f4", "P"),
    ("static", "static top-$C$ (first 10%)", "#a0aec0", "x"),
    ("W4", "Belady within 4 tokens ($W$=4)", "#4a5568", "+"),
]
BANDS = [(0.9, 1.45), (1.45, 2.9), (2.9, 4.6), (4.6, 20)]   # C/k bands for the median lines


def points(path):
    d = json.load(open(path))
    rows = []
    for m, v in d["models"].items():
        for C, cell in v["cells"].items():
            rows.append(dict(model=m, C=int(C), ck=cell["C_over_k"], union=cell["union_to_capacity"],
                             rel={p: q["rel_opt"] for p, q in cell["policies"].items()}))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="prereg/policy_study.json")
    ap.add_argument("--out", default="paper/figs/policy.pdf")
    a = ap.parse_args()
    rows = points(a.json)
    x = np.array([r["ck"] for r in rows])
    fig, ax = plt.subplots(figsize=(4.75, 3.6))
    for key, label, col, mk in SERIES:
        y = np.array([r["rel"][key] for r in rows])
        ax.scatter(x, y, s=14, color=col, marker=mk, lw=0.9, alpha=0.75, label=label, zorder=3)
        bx, by = [], []
        for lo, hi in BANDS:
            sel = (x >= lo) & (x < hi)
            if sel.sum():
                bx.append(np.exp(np.log(x[sel]).mean()))
                by.append(np.median(y[sel]))
        ax.plot(bx, by, color=col, lw=1.0, alpha=0.9, zorder=2)
    ax.axhline(1.0, color="k", lw=0.8, ls="--", zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 12])
    ax.set_xticklabels(["1", "2", "4", "8", "12"])
    yt = [1, 1.5, 2, 3, 5, 10]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{v:g}" for v in yt])
    ax.minorticks_off()
    ax.set_xlabel("cache turnover C/k (slots per layer / experts per token)")
    ax.set_ylabel("host reads per token, relative to Belady")
    ax.set_title("9 MoE models, own text, C = E/8, E/4, 3E/8 per layer", fontsize=10)
    ax.legend(fontsize=6, frameon=False, ncol=2, loc="upper left")
    ymax = max(max(r["rel"][k] for k, *_ in SERIES) for r in rows)
    ax.set_ylim(0.95, ymax * 2.2)
    fig.tight_layout()
    fig.savefig(a.out)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=150)
    print(f"{a.out}: {len(rows)} model x budget points, {len(SERIES)} policies")


if __name__ == "__main__":
    main()
