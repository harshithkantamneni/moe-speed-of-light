"""Paper figures from results/*.json and results/validation.csv."""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("paper/figs", exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "figure.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42})
C = {"static_layer_cpu": "#7f7f7f", "static_hot_xdomain_cpu": "#bcbd22", "lru_fetch": "#d62728",
     "min_fetch_fetch": "#ff9896", "lru_cpu": "#1f77b4", "min_bypass_cpu": "#aec7e8", "speed_of_light": "#000000"}
LBL = {"static_layer_cpu": "static layers, CPU misses (llama.cpp)", "static_hot_xdomain_cpu": "static hot experts, CPU misses",
       "lru_fetch": "LRU, fetch on miss", "min_fetch_fetch": "Belady, fetch on miss",
       "lru_cpu": "LRU, CPU misses", "min_bypass_cpu": "Belady-bypass, CPU misses", "speed_of_light": "speed-of-light bound"}


def fig_validation():
    rows = list(csv.DictReader(open("results/validation.csv")))
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    mk = {"cpu_only": ("o", "CPU only"), "static": ("s", "static CPU offload"), "fetch": ("^", "demand fetch (PCIe)"),
          "numa_holdout": ("x", "2-socket llama.cpp (held out)")}
    for kind, (m, lab) in mk.items():
        r = [x for x in rows if x["kind"] == kind]
        if not r:
            continue
        meas = np.array([float(x["measured"]) for x in r])
        pred = np.array([float(x["predicted_loso"] or x["predicted"]) for x in r])
        ax.scatter(meas, pred, marker=m, s=14, label=lab, alpha=0.8)
    lim = [0.4, 160]
    ax.plot(lim, lim, "k-", lw=0.6)
    for f in (1.25, 0.8):
        ax.plot(lim, [l * f for l in lim], "k:", lw=0.5)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("published decode tok/s"); ax.set_ylabel("predicted tok/s (held-out)")
    ax.legend(loc="upper left", frameon=False)
    fig.savefig("paper/figs/validation.pdf")


def fig_hit_rates(models):
    fig, axes = plt.subplots(1, len(models), figsize=(2.2 * len(models), 2.0), sharey=True)
    axes = np.atleast_1d(axes)
    pol = [("static_layer", "static layers", "#7f7f7f"), ("static_hot_xdomain", "static hot (held-out domain)", "#bcbd22"),
           ("static_hot_oracle", "static hot (oracle)", "#8c564b"), ("lru", "LRU", "#1f77b4"),
           ("min_fetch", "Belady", "#ff9896"), ("min_bypass", "Belady-bypass", "#aec7e8")]
    for ax, m in zip(axes, models):
        a = json.load(open(f"results/analysis_{m}.json"))
        fr = sorted(float(f) for f in a["hit_rate"])
        for key, lab, col in pol:
            ax.plot(fr, [a["hit_rate"][str(f)][key] for f in fr], "-o", ms=2, lw=1, color=col, label=lab)
        ax.set_title(f"{m}\n(E={a['E']}, k={a['k']})", fontsize=7)
        ax.set_xlabel("GPU cache fraction")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("expert hit rate")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.savefig("paper/figs/hit_rates.pdf")


def fig_tok_s(model, tau="tau=20us"):
    a = json.load(open(f"results/analysis_{model}.json"))
    plats = list(a["tok_s"][tau].keys())
    fig, axes = plt.subplots(1, len(plats), figsize=(7.0, 2.3))
    for ax, pn in zip(axes, plats):
        d = a["tok_s"][tau][pn]
        fr = sorted(float(f) for f in d if f not in ("all_gpu", "all_experts_cpu"))
        for key in LBL:
            if key not in d[str(fr[0])]:
                continue
            ax.plot(fr, [d[str(f)][key] for f in fr], "-" if key != "speed_of_light" else "--", marker="o", ms=2,
                    lw=1, color=C[key], label=LBL[key])
        ax.axhline(d["all_experts_cpu"], color="#7f7f7f", lw=0.5, ls=":")
        ax.set_title(pn, fontsize=6.5)
        ax.set_xlabel("GPU cache fraction")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"{model} decode tok/s")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.22))
    fig.savefig(f"paper/figs/toks_{model}.pdf")


if __name__ == "__main__":
    fig_validation()
    models = [m for m in ("olmoe-1b-7b", "qwen3-30b-a3b", "gpt-oss-20b")
              if os.path.exists(f"results/analysis_{m}.json")]
    fig_hit_rates(models)
    for m in models:
        fig_tok_s(m)
    print("figures:", sorted(glob.glob("paper/figs/*.pdf")))
