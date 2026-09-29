"""Figure: what routing foresight is worth to an expert cache (from prereg/foresight/foresight_S.json).

Left: fraction of the gap between the best online policy without foresight and Belady's optimum (host reads per
token) closed by a policy that sees the next W decode steps, for each model at 25% of experts per layer.
Right: W50, the foresight at which half the gap is closed, against the cache's turnover time C/k (tokens), all 29
model x budget points, with the log-log fit.

    python scripts/fig_foresight.py --out figures/foresight.pdf
"""
import argparse
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

W = [1, 2, 4, 8, 16, 32, 64, 256]


def curves(path):
    d = json.load(open(path))
    rows = []
    for m, v in d.items():
        for C, row in v["budgets"].items():
            C = int(C)
            onl, opt = row["dfa-fetch"]["reads"], row["opt"]["reads"]
            g = [(onl - row[f"W{w}"]["reads"]) / (onl - opt) for w in W]
            rows.append(dict(model=m, C=C, E=v["E"], k=v["k"], ck=C / v["k"], g=g, ratio=opt / onl))
    return rows


def w_at(g, th):
    ws, gs = [0.0] + W, [0.0] + g
    for i in range(1, len(gs)):
        if gs[i] >= th:
            if i == 1:
                return ws[1] * th / gs[1]
            x0, x1 = np.log(ws[i - 1]), np.log(ws[i])
            return float(np.exp(x0 + (th - gs[i - 1]) * (x1 - x0) / (gs[i] - gs[i - 1])))
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="prereg/foresight/foresight_S.json")
    ap.add_argument("--out", default="figures/foresight.pdf")
    a = ap.parse_args()
    rows = [r for r in curves(a.json) if "job 063" not in r["model"]]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for r in rows:
        if r["C"] * 4 == r["E"]:
            ax[0].plot(W, [100 * x for x in r["g"]], marker="o", ms=3, lw=1.2, label=f"{r['model']} (C/k={r['ck']:.1f})")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("foresight W (decode tokens seen ahead)")
    ax[0].set_ylabel("% of online-to-optimal read gap closed")
    ax[0].set_title("25% of experts per layer on the GPU", fontsize=10)
    ax[0].axhline(50, color="0.7", lw=0.8, ls="--")
    ax[0].legend(fontsize=6.5, frameon=False)
    x = np.array([r["ck"] for r in rows])
    y = np.array([w_at(r["g"], 0.5) for r in rows])
    ok = np.isfinite(y)
    b, c0 = np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)
    rr = np.corrcoef(np.log(x[ok]), np.log(y[ok]))[0, 1]
    for m in sorted({r["model"] for r in rows}):
        sel = [i for i, r in enumerate(rows) if r["model"] == m]
        ax[1].scatter(x[sel], y[sel], s=18, label=m)
    xs = np.linspace(x.min() * 0.9, x.max() * 1.1, 50)
    ax[1].plot(xs, np.exp(c0) * xs ** b, color="k", lw=1, label=f"W50 = {np.exp(c0):.2f}(C/k)^{b:.2f}, r = {rr:.3f}")
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
