"""Figure: every published measurement of the audit as a percentage of its own speed of light, grouped by source, with
the class medians and our own reference points on the headline machine.

Published rows (prereg/audit/audit.json, scripts/audit.py): system_of_sol = measured tok/s x the bound of
mosl.perfmodel.speed_of_light_time at datasheet-physical ceilings (GPU datasheet rate, the top of the host-DRAM band
from channels x MT/s x 8 B, datasheet PCIe, every efficiency 1, no latencies) with MIN-bypass misses at the row's
per-layer capacity. Classes by where the routing came from: trace (our on-policy S trace of that model), iid
(independent uniform routing, an approximation), nocap (no GPU expert capacity: the bound is the CPU-only rate) and
allfit (every expert fits: the bound is the all-in-VRAM rate).
Reference points: ours = prereg/vram_084.json 'check' (Table 1 speed / the speed limit of scripts/speed_limit.py on
the AIME-25 own-text traces, probed host rate 87.5 GB/s and the RTX 5090 datasheet rate); FreeToken and llama.cpp =
prereg/headline_081.json rows / the same limit per cell; the Qwen3 25% and 43.75% FreeToken cells, which the headline
table takes from job 080, come from prereg/split_080.json (no llama.cpp run there).

    python scripts/fig_audit_sol.py [--audit prereg/audit/audit.json] [--out paper/figs/audit_sol.pdf]
                                    [--json prereg/audit/audit_sol.json]
"""
import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
SHORT = [  # (audit 'system' prefix, short label) in the audit table's order
    ("Fiddler", "Fiddler"), ("ProMoE", "ProMoE"), ("KTransformers", "KTrans."), ("HybriMoE", "HybriMoE"),
    ("Fate", "Fate"), ("SP-MoE", "SP-MoE"), ("MoE-SpeQ", "MoE-SpeQ"), ("CPU-GPU collaborative", "CG-collab"),
    ("FreeToken", "FreeToken"), ("SeqMoE", "SeqMoE"), ("Pipelined sharding", "NV pipe"),
    ("MoE expert-cache forks", "lcpp forks"), ("llama.cpp GPU-resident LRU", "lcpp PR"),
]
CLASS = {"trace": "trace-based routing", "iid": "i.i.d. routing", "nocap": "no GPU expert capacity",
         "allfit": "all experts fit"}
CELLS = {  # (model key in vram_084 'check', C) -> (headline_081 model, budget label)
    ("gpt-oss-120b", 14): ("gpt-oss-120b", "11%"), ("gpt-oss-120b", 32): ("gpt-oss-120b", "25%"),
    ("gpt-oss-120b", 51): ("gpt-oss-120b", "40%"), ("qwen3-30b-a3b-bf16", 16): ("Qwen3-30B-A3B", "12.5%"),
    ("qwen3-30b-a3b-bf16", 32): ("Qwen3-30B-A3B", "25%"), ("qwen3-30b-a3b-bf16", 56): ("Qwen3-30B-A3B", "43.75%"),
}
SPLIT080 = {("qwen3-30b-a3b-bf16", 32): "ft_hybrid_r0.25 L1", ("qwen3-30b-a3b-bf16", 56): "ft_offload_r0.4375 L2"}
CAPTION = ("Published MoE-offloading measurements as a percentage of their own speed of light, grouped by source "
           "(audit order), and our reference points. Published rows: the audit bound at datasheet-physical ceilings "
           "(GPU datasheet bandwidth, top of the host-DRAM band, datasheet PCIe, every efficiency 1, no latencies) with "
           "Belady-with-bypass misses at the row's per-layer capacity. Filled: the routing is our on-policy trace of that "
           "model; hollow: independent uniform routing (an approximation); grey triangles: no GPU expert capacity (the "
           "bound is the CPU-only rate); grey crosses: every expert fits (the bound is the all-in-VRAM rate). Horizontal "
           "lines: class medians over the published rows. Right of the divider: our cache, FreeToken and stock llama.cpp on "
           "the headline machine (RTX 5090 + Ryzen 9 9950X3D, Table 1) as percentages of the paper's speed limit for "
           "each cell (left to right within a model: 11-12.5, 25, 40-43.75% of experts resident), computed on the "
           "AIME-25 own-text traces with the machine's probed host read rate and the GPU's datasheet rate. The two "
           "columns use different ceilings: datasheet-physical for the published rows, probed for ours; a probed host "
           "rate is below the datasheet one, so the published percentages would be higher against a probed ceiling, "
           "and the reference points are for orientation, not a like-for-like ranking.")


def cls(r):
    s = r["mstar_source"]
    return "trace" if "on-policy trace" in s else "iid" if "independent uniform" in s else \
        "nocap" if "no GPU expert capacity" in s else "allfit"


def short(system):
    for pre, lab in SHORT:
        if system.startswith(pre):
            return lab
    raise KeyError(system)


def load_rows(audit):
    rows = []
    for r in json.load(open(audit))["rows"]:
        rows.append(dict(id=r["id"], system=r["system"], label=short(r["system"]), model=r["model"], gpu=r["gpu"],
                         C_per_layer=r["C_per_layer"], E=r["E"], k=r["k"], system_tok_s=r["system_tok_s"],
                         sol_tok_s=r["sol_tok_s"], mstar_source=r["mstar_source"], cls=cls(r),
                         pct_of_sol=100.0 * r["system_of_sol"], tier=r["tier"], labels=r["labels"]))
    return rows


def reference_points():
    v = json.load(open(os.path.join(ROOT, "prereg", "vram_084.json")))["check"]
    v2p = os.path.join(ROOT, "prereg", "speed_limit_v2.json")
    v2 = json.load(open(v2p)) if os.path.exists(v2p) else None
    h = {(r["model"], r["budget"]): r for r in json.load(open(os.path.join(ROOT, "prereg", "headline_081.json")))["rows"]}
    s080 = json.load(open(os.path.join(ROOT, "prereg", "split_080.json")))["means"]
    pts = []
    for key, cell in v.items():
        model, C = key.split(" C")
        C = int(C)
        hm, budget = CELLS[(model, C)]
        lim = cell["limit_tok_s"]
        ours = cell["ours_frac_of_limit"] * lim
        src = "vram_084.json check (Table 1 speed / limit)"
        if v2 is not None:   # the exact optimum of speed_limit_v2.py supersedes the 084 optimum (Table 1's limit column)
            lim = v2["models"][model]["rows"][str(C)]["limits"]["exact"]["tok_s"]
            src = "speed_limit_v2.json exact limit; Table 1 speed"
        p = dict(model=hm, key=model, C=C, budget=budget, limit_tok_s=lim, ours_tok_s=ours,
                 ours_pct=100.0 * ours / lim, ours_source=src)
        if (hm, budget) in h:
            r = h[(hm, budget)]
            p.update(freetoken_tok_s=r["ft_L2"], freetoken_variant=r["ft_L2_variant"], freetoken_pct=100.0 * r["ft_L2"] / lim,
                     llama_tok_s=r["llama"], llama_pct=100.0 * r["llama"] / lim, competitor_source="headline_081.json")
        elif (model, C) in SPLIT080:
            ft = s080[SPLIT080[(model, C)]]
            p.update(freetoken_tok_s=ft, freetoken_variant=SPLIT080[(model, C)].split("_")[1], freetoken_pct=100.0 * ft / lim,
                     llama_tok_s=None, llama_pct=None, competitor_source="split_080.json (job 080; no llama.cpp run)")
        pts.append(p)
    order = {"gpt-oss-120b": 0, "Qwen3-30B-A3B": 1}
    pts.sort(key=lambda p: (order[p["model"]], p["C"]))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default=os.path.join(ROOT, "prereg", "audit", "audit.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "figs", "audit_sol.pdf"))
    ap.add_argument("--json", default=os.path.join(ROOT, "prereg", "audit", "audit_sol.json"))
    a = ap.parse_args()
    rows = load_rows(a.audit)
    pts = reference_points()
    labels = [lab for _, lab in SHORT]
    stats = {}
    for c in CLASS:
        v = np.array([r["pct_of_sol"] for r in rows if r["cls"] == c])
        stats[c] = dict(name=CLASS[c], n=int(len(v)), median=float(np.median(v)), q1=float(np.percentile(v, 25)),
                        q3=float(np.percentile(v, 75)), min=float(v.min()), max=float(v.max()))
    allv = np.array([r["pct_of_sol"] for r in rows])
    stats["all"] = dict(name="all published rows", n=int(len(allv)), median=float(np.median(allv)),
                        q1=float(np.percentile(allv, 25)), q3=float(np.percentile(allv, 75)), min=float(allv.min()), max=float(allv.max()))
    for r in rows:
        print(f"{r['label']:10s} {r['cls']:6s} {r['pct_of_sol']:5.1f}%  {r['id']}")
    for c, s in stats.items():
        print(f"{s['name']:26s} n={s['n']:2d} median {s['median']:5.1f}%  IQR [{s['q1']:.1f}, {s['q3']:.1f}]  range [{s['min']:.1f}, {s['max']:.1f}]")
    for p in pts:
        print(f"{p['model']:14s} {p['budget']:7s} C={p['C']:2d} limit {p['limit_tok_s']:6.1f}: ours {p['ours_pct']:4.1f}%"
              + (f"  FreeToken {p['freetoken_pct']:4.1f}% ({p['freetoken_variant']})" if p.get("freetoken_pct") else "")
              + (f"  llama.cpp {p['llama_pct']:4.1f}%" if p.get("llama_pct") else "") + f"  [{p.get('competitor_source', '-')}]")

    plt.rcParams.update({"font.size": 7, "font.family": "serif", "font.serif": ["Latin Modern Roman"], "mathtext.fontset": "cm",
                         "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    col = "#1f6f8b"
    grey = "0.55"
    xs = {lab: i for i, lab in enumerate(labels)}
    for lab in labels:
        grp = [r for r in rows if r["label"] == lab]
        n = len(grp)
        for j, r in enumerate(grp):
            x = xs[lab] + (0 if n == 1 else -0.28 + 0.56 * j / (n - 1))
            y = r["pct_of_sol"]
            if r["cls"] == "trace":
                ax.plot(x, y, "o", ms=3.2, color=col, mfc=col, mew=0.7, zorder=3)
            elif r["cls"] == "iid":
                ax.plot(x, y, "o", ms=3.2, color=col, mfc="white", mew=0.8, zorder=3)
            elif r["cls"] == "nocap":
                ax.plot(x, y, "v", ms=3.0, color=grey, mfc=grey, mew=0.6, zorder=2)
            else:
                ax.plot(x, y, "+", ms=4.0, color=grey, mew=0.9, zorder=2)
    npub = len(labels)
    x0, x1 = -0.6, npub - 0.4
    styles = {"trace": dict(ls="-", color=col, lw=0.8), "iid": dict(ls="--", color=col, lw=0.8),
              "nocap": dict(ls=":", color=grey, lw=0.7), "allfit": dict(ls=":", color=grey, lw=0.7)}
    for c, st in styles.items():
        m = stats[c]["median"]
        ax.plot([x0, x1], [m, m], zorder=1, **st)
        ax.text(x1 + 0.08, m, f"{m:.1f}%" if m < 10 else f"{m:.0f}%", fontsize=5, va="center", ha="left", color=st["color"])
    # our reference points
    ax.axvline(npub + 0.35, color="0.3", lw=0.6)
    ax.axvspan(npub + 0.35, npub + 2.65, color="0.94", zorder=0, lw=0)
    gx = {"gpt-oss-120b": npub + 1.0, "Qwen3-30B-A3B": npub + 2.0}
    for model in gx:
        cells = [p for p in pts if p["model"] == model]
        for j, p in enumerate(cells):
            x = gx[model] - 0.28 + 0.28 * j
            ax.plot(x, p["ours_pct"], "*", ms=5.5, color="#c0572b", mfc="#c0572b", mew=0.5, zorder=4)
            if p.get("freetoken_pct"):
                ax.plot(x, p["freetoken_pct"], "D", ms=2.6, color="#2f9e44", mfc="#2f9e44", mew=0.5, zorder=3)
            if p.get("llama_pct"):
                ax.plot(x, p["llama_pct"], "s", ms=2.6, color="0.2", mfc="0.2", mew=0.5, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(1.5, 130)
    ax.set_yticks([2, 5, 10, 20, 50, 100])
    ax.set_yticklabels(["2", "5", "10", "20", "50", "100"])
    ax.minorticks_off()
    ax.set_ylabel("% of own speed of light", fontsize=7)
    ax.set_xticks(list(range(npub)) + [gx["gpt-oss-120b"], gx["Qwen3-30B-A3B"]])
    ax.set_xticklabels(labels + ["gpt-oss-120b", "Qwen3-30B"], rotation=60, ha="right", fontsize=5.6, rotation_mode="anchor")
    ax.set_xlim(-0.8, npub + 2.75)
    ax.tick_params(axis="x", length=2, pad=1)
    ax.tick_params(axis="y", labelsize=6)
    ax.text(x1 / 2 - 0.3, 108, "published (datasheet ceilings)", fontsize=5.5, ha="center", va="bottom", color="0.25")
    ax.text(npub + 1.5, 108, "ours, headline\nmachine (probed)", fontsize=5.5, ha="center", va="bottom", color="0.25", linespacing=0.9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    h = [Line2D([], [], marker="o", ls="", color=col, mfc=col, ms=3.2, label="trace-based routing"),
         Line2D([], [], marker="o", ls="", color=col, mfc="white", ms=3.2, label="i.i.d. routing"),
         Line2D([], [], marker="v", ls="", color=grey, mfc=grey, ms=3.0, label="no GPU expert capacity"),
         Line2D([], [], marker="+", ls="", color=grey, ms=4.0, mew=0.9, label="all experts fit"),
         Line2D([], [], marker="*", ls="", color="#c0572b", ms=5.5, label="ours"),
         Line2D([], [], marker="D", ls="", color="#2f9e44", ms=2.6, label="FreeToken"),
         Line2D([], [], marker="s", ls="", color="0.2", ms=2.6, label="llama.cpp"),
         Line2D([], [], ls="-", color=col, lw=0.8, label="median, trace"),
         Line2D([], [], ls="--", color=col, lw=0.8, label="median, i.i.d.")]
    fig.legend(handles=h, fontsize=5.0, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.54, -0.005),
               columnspacing=0.7, handletextpad=0.25, handlelength=1.4, borderaxespad=0.0)
    fig.tight_layout(rect=(0, 0.13, 1, 1), pad=0.3)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=220)
    out = dict(
        note=("Published rows: percentage of the audit's speed of light at datasheet-physical ceilings (scripts/audit.py, "
              "mosl.perfmodel.speed_of_light_time: GPU datasheet bandwidth, top of the host-DRAM band, datasheet PCIe, "
              "every efficiency 1, no latencies, MIN-bypass misses at the row's per-layer capacity). Reference points: "
              "percentage of the paper's speed limit (scripts/speed_limit.py) on the headline machine, whose host ceiling "
              "is the probed 87.5 GB/s and whose GPU ceiling is the RTX 5090 datasheet rate; the ceilings differ, so the "
              "two sets are not a like-for-like ranking."),
        caption=CAPTION, classes=CLASS, class_stats=stats,
        by_source={lab: dict(n=len([r for r in rows if r["label"] == lab]),
                             median=float(np.median([r["pct_of_sol"] for r in rows if r["label"] == lab])),
                             classes=sorted({r["cls"] for r in rows if r["label"] == lab})) for lab in labels},
        rows=rows, reference_points=pts, figure=os.path.relpath(a.out, ROOT))
    json.dump(out, open(a.json, "w"), indent=1)
    print(f"wrote {a.out} and {a.json}")


if __name__ == "__main__":
    main()
