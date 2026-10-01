"""Figure: what routing foresight is worth to an expert cache, with the exact optimum (from
prereg/foresight/foresight_S_exact.json, scripts/foresight_exact.py). Same two panels as scripts/fig_foresight.py:

Left: fraction of the gap between the best online policy without foresight and Belady's optimum (host reads per
token) closed by a policy that sees the next W decode steps, for each model at 25% of experts per layer.
Right: W50, the foresight at which half the gap is closed, against the cache's turnover time C/k (tokens), all 26
model x budget points (the job 063 trace excluded, as in fig_foresight.py), with the log-log fit; the original
file's fit is drawn dotted for comparison when it is present.

    python scripts/fig_foresight_exact.py --out paper/figs/foresight_exact.pdf
"""
import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.fig_foresight import W, curves, w_at  # noqa: E402


def fit_rows(rows):
    x = np.array([r["ck"] for r in rows])
    y = np.array([w_at(r["g"], 0.5) for r in rows])
    ok = np.isfinite(y)
    b, c0 = np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)
    rr = np.corrcoef(np.log(x[ok]), np.log(y[ok]))[0, 1]
    return x, y, ok, b, c0, rr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="prereg/foresight/foresight_S_exact.json")
    ap.add_argument("--original", default="prereg/foresight/foresight_S.json")
    ap.add_argument("--out", default="paper/figs/foresight_exact.pdf")
    a = ap.parse_args()
    rows = [r for r in curves(a.json) if "job 063" not in r["model"]]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for r in rows:
        if r["C"] * 4 == r["E"]:
            ax[0].plot(W, [100 * x for x in r["g"]], marker="o", ms=3, lw=1.2, label=f"{r['model']} (C/k={r['ck']:.1f})")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("foresight W (decode tokens seen ahead)")
    ax[0].set_ylabel("% of online-to-optimal read gap closed")
    ax[0].set_title("25% of experts per layer on the GPU (exact optimum)", fontsize=10)
    ax[0].axhline(50, color="0.7", lw=0.8, ls="--")
    ax[0].legend(fontsize=6.5, frameon=False)
    x, y, ok, b, c0, rr = fit_rows(rows)
    for m in sorted({r["model"] for r in rows}):
        sel = [i for i, r in enumerate(rows) if r["model"] == m]
        ax[1].scatter(x[sel], y[sel], s=18, label=m)
    xs = np.linspace(x.min() * 0.9, x.max() * 1.1, 50)
    ax[1].plot(xs, np.exp(c0) * xs ** b, color="k", lw=1, label=f"W50 = {np.exp(c0):.2f}(C/k)^{b:.2f}, r = {rr:.3f}")
    if a.original and os.path.exists(a.original):
        orows = [r for r in curves(a.original) if "job 063" not in r["model"]]
        _, _, _, ob, oc0, orr = fit_rows(orows)
        ax[1].plot(xs, np.exp(oc0) * xs ** ob, color="0.5", lw=1, ls=":", label=f"original: {np.exp(oc0):.2f}(C/k)^{ob:.2f}, r = {orr:.3f}")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("cache turnover C/k (tokens)")
    ax[1].set_ylabel("W50: foresight closing half the gap")
    ax[1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(a.out)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=150)
    print(f"{a.out}: n={ok.sum()} fit exponent {b:.2f} prefactor {np.exp(c0):.2f} r={rr:.3f}")


if __name__ == "__main__":
    main()
