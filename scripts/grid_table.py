"""The grid: Table 1's protocol on every card we measured, each cell against its own machine's speed limit.

Reads the per-job statistics (prereg/headline_081.json + speed_limit_v2.json for listing B, stockclock_089.json for the
second 5090 host, grid_091.json and grid_092.json for the RTX 4090 and RTX 3090 when present) and writes
  paper/tab_grid.tex   one row per card and cell: the three systems, ours / FreeToken with its interval, ours / llama.cpp,
                       the limit on that machine (exact optimum reads, the machine's best probed host rate, the card's
                       datasheet rate) with the term that binds it, and ours as a percentage of it
  paper/figs/grid.pdf  ours and FreeToken as a fraction of each machine's limit, per cell, one marker per card
  paper/wsg_grid.tex   macros (gr*)

    python scripts/grid_table.py
"""
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
ORDER = [("gpt-oss-120b", "11%"), ("gpt-oss-120b", "25%"), ("gpt-oss-120b", "40%"),
         ("Qwen3-30B-A3B", "12.5%"), ("Qwen3-30B-A3B", "25%"), ("Qwen3-30B-A3B", "43.75%")]
MACROS = {}


def M(k, v):
    MACROS[k] = v


def load(name):
    p = P("prereg", name)
    return json.load(open(p)) if os.path.exists(p) else None


def listing_b_rows():
    """Listing B: the six cells of Table 1 (headline_081.json, split_080.json, run1_082.json as scripts/stockclock_stats.py
    collects them) against the exact limit (speed_limit_v2.json)."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from stockclock_stats import listing_b
    B = listing_b()
    h = load("headline_081.json")
    sl = load("speed_limit_v2.json")
    lm = {"gpt-oss-120b": "gpt-oss-120b", "Qwen3-30B-A3B": "qwen3-30b-a3b-bf16"}
    CS = {("gpt-oss-120b", "11%"): 14, ("gpt-oss-120b", "25%"): 32, ("gpt-oss-120b", "40%"): 51,
          ("Qwen3-30B-A3B", "12.5%"): 16, ("Qwen3-30B-A3B", "25%"): 32, ("Qwen3-30B-A3B", "43.75%"): 56}
    full = {(r["model"], r["budget"]): r for r in h["rows"]}
    tab = load("stockclock_089.json")["hosts"]["listing B (081)"]
    rows = []
    for key in ORDER:
        b = B[key]
        s = sl["models"][lm[key[0]]]["rows"][str(CS[key])]
        lim = s["limits"]["exact"]
        r = full.get(key)
        rows.append(dict(model=key[0], budget=key[1], ours=b["ours"], ft=b["ft"], llama=b["llama"],
                         ours_over_ft=b["ratio_ci"],
                         ours_over_llama=b.get("ours_over_llama") or [b["ours"] / b["llama"], None, None],
                         limit_tok_s=lim["tok_s"], cpu_at_limit=lim["cpu_at_limit"], frac_ours=s["frac_of_limit"]["exact"]["ours"],
                         frac_ft=s["frac_of_limit"]["exact"]["freetoken"], frac_llama=s["frac_of_limit"]["exact"]["llama"],
                         gpu_only_ms=s["gpu_only"]["datasheet_ms"], t_ms=lim["t_ms"]))
    return dict(card="RTX 5090", host="9950X3D (listing B)", b_host=sl["B_host"]["max"] / 1e9, b_gpu=1792, tables=dict(gptoss=tab["table_gpt"], qwen3=tab["table_qwen"]), rows=rows)


def stockclock_rows():
    d = load("stockclock_089.json")
    rows = []
    for r in d["rows"]:
        L = r["limit"]
        rows.append(dict(model=r["model"], budget=r["budget"], ours=r["ours"], ft=r["ft"], llama=r["llama"], ours_over_ft=r["ours_over_ft_L2"],
                         ours_over_llama=r["ours_over_llama"], limit_tok_s=L["tok_s"], cpu_at_limit=L["cpu_at_limit"], frac_ours=L["frac"]["ours"],
                         frac_ft=L["frac"]["ft"], frac_llama=L["frac"]["llama"], t_ms=L["t_ms"]))
    hb = d["hosts"]["089 (9950X, stock clock)"]
    return dict(card="RTX 5090", host="9950X (job 089)", b_host=d["b_host_max_gbs"], b_gpu=1792, tables=dict(gptoss=hb["table_gpt"], qwen3=hb["table_qwen"]), rows=rows)


def grid_rows(name, host_label):
    d = load(name)
    if not d:
        return None
    rows = []
    for r in d["rows"]:
        L = r["limit"]
        rows.append(dict(model=r["model"], budget=r["budget"], ours=r["ours"], ft=r.get("ft"), llama=r.get("llama"), ours_over_ft=r["ours_over_ft_L2"],
                         ours_over_llama=r.get("ours_over_llama"), limit_tok_s=L["tok_s"], cpu_at_limit=L["cpu_at_limit"], frac_ours=L["frac"]["ours"],
                         frac_ft=L["frac"]["ft"], frac_llama=L["frac"]["llama"], t_ms=L["t_ms"], gpu_only_ms=L["gpu_only_ms"]))
    tb = {k: v["table"] for k, v in d["tables"].items()}
    for r, src in zip(rows, d["rows"]):
        r["hit_rate"] = src.get("cache", {}).get("hit_rate")
    return dict(card=d["card"], host=host_label, b_host=d["probes"]["b_host_max"], b_gpu=d["rows"][0]["limit"]["b_gpu_gbs"] if d["rows"] else None,
                tables=tb, rows=rows, mixtral=d.get("mixtral", []), gpu=d.get("gpu"), skipped=d.get("skipped.txt", ""), ft_probe=d.get("ft_probe"))


def regime(row):
    """What binds the limit. With c experts per token run on the CPU, T = max((D + (Lk - c) S) / B_gpu, max(R*, c) S / B_host).
    When the optimum's reads alone (c = 0) take longer than the GPU's whole read, the host term binds and the GPU has
    slack ('host'); otherwise the limit moves experts to the CPU until the two terms are equal, both paths saturated
    ('both'), which is where the all-in-VRAM speed can be exceeded."""
    return "host" if row["cpu_at_limit"] <= 0 else "both"


def fmt_ratio(x):
    if not x:
        return "--"
    return f"{x[0]:.2f}" if x[1] is None else f"{x[0]:.2f} [{x[1]:.2f}, {x[2]:.2f}]"


def main():
    cards = [listing_b_rows(), stockclock_rows()]
    for name, lab in (("grid_091.json", "i5-12400 (job 091)"), ("grid_092.json", "i9-11900KF (job 092)")):
        g = grid_rows(name, lab)
        if g:
            cards.append(g)
    lines = []
    for c in cards:
        tg = ",".join(map(str, c["tables"].get("gptoss", []))); tq = ",".join(map(str, c["tables"].get("qwen3", [])))
        lines.append(r"\midrule" + "\n" + r"\multicolumn{10}{l}{\textbf{%s}, %s: $B_{\mathrm{host}}$ %.0f\,GB/s, $B_{\mathrm{gpu}}$ %d\,GB/s; tables \texttt{%s} / \texttt{%s}} \\" % (
            c["card"], c["host"], c["b_host"], c["b_gpu"], tg, tq))
        for key in ORDER:
            r = next((x for x in c["rows"] if (x["model"], x["budget"]) == key), None)
            if not r:
                continue
            reg = regime(r)
            dag = r"$^\dagger$" if "3090" in c["card"] else ""
            lines.append("%s & %s & %s & %s & %.1f & %s & %s & %.0f & %s & %.0f \\\\" % (
                r["model"], r["budget"].replace("%", r"\%"), f"{r['llama']:.1f}" if r.get("llama") else "--", f"{r['ft']:.1f}{dag}" if r.get("ft") else "--",
                r["ours"], fmt_ratio(r["ours_over_ft"]) + dag, f"{r['ours_over_llama'][0]:.2f}" if r.get("ours_over_llama") else "--",
                r["limit_tok_s"], reg, 100 * r["frac_ours"]))
        for m in c.get("mixtral", []) or []:
            lines.append("Mixtral-8x7B Q4\\_K\\_M & $C{=}%d$ (%d\\%%) & %.2f & -- & %.2f & -- & %.3f [%.3f, %.3f] & -- & -- & -- \\\\" % (
                m["C"], 100 * m["C"] // 8, m["llama"], m["ours"], m["ours_over_llama"][0], m["ours_over_llama"][1], m["ours_over_llama"][2]))
    tex = r"""\begin{table*}[t]\centering\small
\caption{The grid: \cref{tab:headline}'s protocol on every card and host we measured, each cell against its own
machine's limit (the exact optimum's reads on the same trace, the machine's highest probed host rate, the card's
datasheet rate). \emph{Bound}: what binds the limit; \emph{host} when the optimum's reads alone take longer than the
GPU's whole read, so the GPU has slack, \emph{both} when the limit runs experts on the CPU until the two paths take
equally long, which is where the all-in-VRAM speed can be exceeded. Ours and
FreeToken are launch 2 (FreeToken's backend carried over from the headline machine's selection); the 40 and 43.75\%
budgets do not fit a 24\,GB card. $^\dagger$On the RTX 3090 FreeToken's carried-over hybrid backend ran below
llama.cpp; its own calibration recommends its offload backend there, which we did not run, so those cells are
reported, not counted. Mixtral-8x7B (26\,GB of experts, top-2 of 8) is ours against llama.cpp only, on the RTX 3090,
launch 1.}\label{tab:grid}
\setlength\tabcolsep{3pt}\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrcrrcr}\toprule
Model & Experts & llama.cpp & FreeToken & Ours & Ours $\div$ FreeToken & Ours $\div$ & Limit & Bound & Ours, \% \\
 & on GPU & (tok/s) & (tok/s) & (tok/s) & (95\% CI) & llama.cpp & (tok/s) & & of limit \\
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}}\end{table*}
"""
    open(P("paper", "tab_grid.tex"), "w").write(tex)
    # macros
    allrows = [(c, r) for c in cards for r in c["rows"]]
    M("grCardsN", {1: "one", 2: "two", 3: "three", 4: "four"}[len({c["card"] for c in cards})])
    M("grHostsN", {2: "two", 3: "three", 4: "four", 5: "five"}[len(cards)])
    fr = [r["frac_ours"] for c, r in allrows]
    M("grFracMin", f"{100 * min(fr):.0f}"); M("grFracMax", f"{100 * max(fr):.0f}")
    hostb = [r["frac_ours"] for c, r in allrows if regime(r) == "host"]
    gpub = [r["frac_ours"] for c, r in allrows if regime(r) == "both"]
    if hostb:
        M("grHostFracMin", f"{100 * min(hostb):.0f}"); M("grHostFracMax", f"{100 * max(hostb):.0f}")
    if gpub:
        M("grBothFracMin", f"{100 * min(gpub):.0f}"); M("grBothFracMax", f"{100 * max(gpub):.0f}")
    M("grCellsN", str(len(allrows)))
    lim = [r["limit_tok_s"] for c, r in allrows]
    M("grLimMin", f"{min(lim):.0f}"); M("grLimMax", f"{max(lim):.0f}")
    for c in cards[2:]:
        tag = "Fourk" if "4090" in c["card"] else "Threek"
        rr = c["rows"]
        if not rr:
            continue
        lead = [r for r in rr if r["ours_over_ft"] and r["ours_over_ft"][1] > 1]
        trail = [r for r in rr if r["ours_over_ft"] and r["ours_over_ft"][2] < 1]
        M(f"gr{tag}Cells", str(len(rr))); M(f"gr{tag}Lead", str(len(lead))); M(f"gr{tag}Trail", str(len(trail)))
        rat = [r["ours_over_ft"][0] for r in rr if r["ours_over_ft"]]
        if rat:
            M(f"gr{tag}FtMin", f"{min(rat):.2f}"); M(f"gr{tag}FtMax", f"{max(rat):.2f}")
        ll = [r["ours_over_llama"][0] for r in rr if r.get("ours_over_llama")]
        if ll:
            M(f"gr{tag}LlamaMin", f"{min(ll):.1f}"); M(f"gr{tag}LlamaMax", f"{max(ll):.1f}")
        fr = [r["frac_ours"] for r in rr]
        M(f"gr{tag}FracMin", f"{100 * min(fr):.0f}"); M(f"gr{tag}FracMax", f"{100 * max(fr):.0f}")
        M(f"gr{tag}HostBw", f"{c['b_host']:.0f}")
        if c.get("gpu"):
            M(f"gr{tag}DevRead", f"{c['gpu'].get('device_read_gbs') or 0:.0f}")
        for m in c.get("mixtral", []) or []:
            w = "Two" if m["C"] == 2 else "Four"
            M(f"grMixC{w}", f"{m['ours_over_llama'][0]:.3f}"); M(f"grMixC{w}Lo", f"{m['ours_over_llama'][1]:.3f}"); M(f"grMixC{w}Hi", f"{m['ours_over_llama'][2]:.3f}")
            M(f"grMixOursC{w}", f"{m['ours']:.2f}"); M(f"grMixLlamaC{w}", f"{m['llama']:.2f}")
            if m.get("cache"):
                M(f"grMixHitC{w}", f"{100 * m['cache']['hit_rate']:.0f}"); M(f"grMixPinC{w}", f"{100 * m['cache']['pinned_hit_rate']:.0f}")
        hr = [r["hit_rate"] for r in rr if r.get("hit_rate")]
        if hr:
            M(f"gr{tag}HitMin", f"{100 * min(hr):.0f}"); M(f"gr{tag}HitMax", f"{100 * max(hr):.0f}")
        if c.get("ft_probe"):
            M(f"gr{tag}FtCpu", f"{c['ft_probe']['cpu']:.1f}"); M(f"gr{tag}FtPcie", f"{c['ft_probe']['pcie']:.1f}")
        if "3090" in c["card"]:
            # FreeToken's hybrid on this host ran below llama.cpp: its speeds and the ours / llama.cpp ratios are the comparison
            ft = [r["ft"] for r in rr if r.get("ft")]
            M("grThreekFtSpeedMin", f"{min(ft):.1f}"); M("grThreekFtSpeedMax", f"{max(ft):.1f}")
    # hit rates of the 128-expert models on the 24 GB cards against Mixtral's
    hr = [r["hit_rate"] for c in cards[2:] for r in c["rows"] if r.get("hit_rate")]
    if hr:
        M("grHitMin", f"{100 * min(hr):.0f}"); M("grHitMax", f"{100 * max(hr):.0f}")
    # every cell on the 24 GB cards host-bound?
    reg = [regime(r) for c in cards[2:] for r in c["rows"]]
    M("grSmallHostBound", "every" if reg and all(x == "host" for x in reg) else f"{sum(x == 'host' for x in reg)} of {len(reg)}")
    M("grSmallCellsN", {5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}.get(len(reg), str(len(reg))))
    with open(P("paper", "wsg_grid.tex"), "w") as f:
        f.write("% generated by scripts/grid_table.py\n")
        for k in sorted(MACROS):
            f.write("\\newcommand{\\%s}{%s}\n" % (k, MACROS[k]))
    # figure: fraction of the limit per cell, one marker per card
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    xs = {key: i for i, key in enumerate(ORDER)}
    mk = {"RTX 5090": "o", "RTX 4090": "s", "RTX 3090": "^"}
    col = {"9950X3D (listing B)": "#1f5fa8", "9950X (job 089)": "#5b9bd5", "i5-12400 (job 091)": "#2f9e44", "i9-11900KF (job 092)": "#c0572b"}
    for c in cards:
        for r in c["rows"]:
            x = xs[(r["model"], r["budget"])]
            ax.scatter(x - 0.12, 100 * r["frac_ours"], marker=mk[c["card"]], s=18, color=col.get(c["host"], "k"), zorder=3)
            if r.get("frac_ft") and "3090" not in c["card"]:   # FreeToken's 3090 cells are not a comparison at its best (see the table's note)
                ax.scatter(x + 0.12, 100 * r["frac_ft"], marker=mk[c["card"]], s=18, facecolors="none", edgecolors=col.get(c["host"], "k"), linewidths=0.9, zorder=3)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([f"{'gpt-oss' if m.startswith('gpt') else 'Qwen3'}\n{b}" for m, b in ORDER], fontsize=6)
    ax.set_ylabel("% of the machine's limit", fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.set_ylim(0, None)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker=mk[c["card"]], color=col.get(c["host"], "k"), label=f"{c['card']}, {c['host'].split(' (')[0]}", markersize=4) for c in cards]
    h += [Line2D([], [], ls="", marker="o", color="k", label="ours", markersize=4), Line2D([], [], ls="", marker="o", color="k", markerfacecolor="none", label="FreeToken", markersize=4)]
    fig.legend(handles=h, fontsize=5.2, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02), columnspacing=0.8, handletextpad=0.2)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    os.makedirs(P("paper", "figs"), exist_ok=True)
    fig.savefig(P("paper", "figs", "grid.pdf"))
    print(f"{len(cards)} cards, {len(allrows)} cells; macros: " + ", ".join(f"{k}={v}" for k, v in sorted(MACROS.items())))


if __name__ == "__main__":
    main()
