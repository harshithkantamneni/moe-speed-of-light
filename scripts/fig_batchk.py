"""Figure: what batched verification (speculative decoding, K positions per forward pass) buys an expert cache, from
prereg/batchk_trace.json (scripts/batchk_trace.py).

One panel per model, at C = E/4 slots per layer (Mixtral 3): host reads per committed token relative to plain decoding
(K = 1) against K, for the decayed-frequency online policy, one colour per acceptance rate alpha. Solid: rejected
drafts cost nothing (optimistic); dashed: rejected drafts route like the true token at their position ('same');
dotted (faint): rejected drafts route like a random other token ('random', pessimistic), above 1 everywhere and
leaving the panel at larger K.
Belady with bypass is in the JSON and the outcome table (its 'free' curve is flat at 1: foresight already has the
future, batching adds nothing to it).

    python scripts/fig_batchk.py [--json prereg/batchk_trace.json] [--out paper/figs/batchk.pdf] [--budget 1]
                                 [--policy online]
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
ORDER = ["olmoe", "gpt-oss-20b", "qwen3-30b-a3b", "gpt-oss-120b", "mixtral-8x7b", "deepseek-v2-lite", "qwen1.5-moe",
         "qwen2-57b", "phi3.5-moe"]
NAMES = {"olmoe": "OLMoE-1B-7B", "gpt-oss-20b": "gpt-oss-20b", "qwen3-30b-a3b": "Qwen3-30B-A3B", "gpt-oss-120b": "gpt-oss-120b",
         "mixtral-8x7b": "Mixtral-8x7B", "deepseek-v2-lite": "DeepSeek-V2-Lite", "qwen1.5-moe": "Qwen1.5-MoE",
         "qwen2-57b": "Qwen2-57B-A14B", "phi3.5-moe": "Phi-3.5-MoE"}
COLS = {0.6: "#c0572b", 0.8: "#2f9e44", 0.9: "#1f5fa8"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(ROOT, "prereg", "batchk_trace.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "figs", "batchk.pdf"))
    ap.add_argument("--budget", type=int, default=1, help="index of the budget: 0 = E/8, 1 = E/4, 2 = 3E/8")
    ap.add_argument("--policy", default="online", choices=("online", "belady"))
    a = ap.parse_args()
    d = json.load(open(a.json))
    K = d["K"]
    alphas = d["alphas"]
    models = [m for m in ORDER if m in d["models"]]
    plt.rcParams.update({"font.size": 6.5, "font.family": "serif", "font.serif": ["Latin Modern Roman"], "mathtext.fontset": "cm",
                         "axes.linewidth": 0.5})
    fig, axs = plt.subplots(3, 3, figsize=(3.4, 3.1), sharex=True, sharey=True)
    for ax, key in zip(axs.ravel(), models):
        rec = d["models"][key]
        Cs = sorted(rec["budgets"], key=int)
        b = rec["budgets"][Cs[a.budget]]
        rows = b["rows"]

        def curve(alpha, variant):
            ys = [1.0]
            for k in K[1:]:
                r = next(x for x in rows if x["K"] == k and x["alpha"] == alpha and x["variant"] == variant)
                ys.append(r[a.policy]["rel_to_K1"])
            return ys
        for alpha in alphas:
            ax.plot(K, curve(alpha, "random"), ls=":", lw=0.7, color=COLS[alpha], alpha=0.55, marker="o", ms=1.6, mew=0)
        for alpha in alphas:
            ax.plot(K, curve(alpha, "same"), ls="--", lw=0.8, color=COLS[alpha], marker="o", ms=1.8, mew=0)
            ax.plot(K, curve(alpha, "free"), ls="-", lw=0.9, color=COLS[alpha], marker="o", ms=2.0, mew=0)
        ax.axhline(1.0, color="0.3", lw=0.5, ls="-", zorder=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(K)
        ax.set_xticklabels([str(k) for k in K])
        ax.set_yticks([0.5, 1.0, 1.5, 2.0])
        ax.set_yticklabels(["0.5", "1", "1.5", "2"])
        ax.set_ylim(0.45, 2.05)
        ax.minorticks_off()
        ax.tick_params(labelsize=5.5, length=2, pad=1.5)
        ax.set_title(f"{NAMES[key]}, $C$ = {b['C']}", fontsize=5.5, pad=2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.supxlabel("positions per verification forward pass $K$ ($K$ = 1: plain decoding)", fontsize=6, y=0.1)
    fig.supylabel("host reads per committed token, relative to $K$ = 1", fontsize=6, x=0.012)
    h = [Line2D([], [], color=COLS[al], lw=0.9, label=f"$\\alpha$ = {al}") for al in alphas]
    h += [Line2D([], [], color="0.2", lw=0.9, ls="-", label="rejected drafts free"),
          Line2D([], [], color="0.2", lw=0.8, ls="--", label="route like true token"),
          Line2D([], [], color="0.2", lw=0.7, ls=":", label="route like random token")]
    fig.legend(handles=h, fontsize=5.2, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.53, -0.005),
               columnspacing=0.9, handletextpad=0.35, handlelength=1.6, borderaxespad=0.0)
    fig.tight_layout(rect=(0.02, 0.11, 1, 1), pad=0.25, h_pad=0.5, w_pad=0.4)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=220)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
