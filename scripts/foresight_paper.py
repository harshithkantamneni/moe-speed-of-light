"""Macros and the figure for the measured-foresight paragraph (job 093), from prereg/foresight_093.json.

Writes paper/wsg_foresight.tex (macros fsm*; also \\fsMeasuredAbstract and \\fsMeasuredContrib, the clauses the abstract
and the contributions list carry) and paper/figs/foresight.pdf (per host-bound cell: speed as a fraction of this host's
limit against the oracle's window W, with base, the model's foresight-only term and the limit as references).

    python scripts/foresight_paper.py
"""
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
MACROS = {}


def M(k, v):
    MACROS[k] = v


def main():
    d = json.load(open(P("prereg", "foresight_093.json")))
    cells = d["cells"]
    host = [c for c in cells if c["model_terms"]["c_at_limit"] <= 0]        # host-bound at the limit
    gpu = [c for c in cells if c["model_terms"]["c_at_limit"] > 0]
    tagof = {("gpt-oss-120b", 14): "GLow", ("gpt-oss-120b", 32): "GMid", ("gpt-oss-120b", 51): "GHigh",
             ("qwen3-30b-a3b-bf16", 16): "QLow", ("qwen3-30b-a3b-bf16", 32): "QMid", ("qwen3-30b-a3b-bf16", 56): "QHigh"}
    gains, recov, closed, fr0, frb, hit0, hitb, admits, lawerr, noovl, allcpu, w50m = [], [], [], [], [], [], [], [], [], [], [], []
    for c in cells:
        tag = tagof[(c["model"], c["C"])]
        r = c["runs"]
        if "oracle_w0" not in r:
            continue
        o, b = r["oracle_w0"], r["base"]
        M(f"fsmGain{tag}", f"{100 * o['gain']:.0f}"); M(f"fsmGainLo{tag}", f"{100 * (o['ratio_to_base'][1] - 1):.0f}"); M(f"fsmGainHi{tag}", f"{100 * (o['ratio_to_base'][2] - 1):.0f}")
        M(f"fsmBase{tag}", f"{b['mean']:.1f}"); M(f"fsmOracle{tag}", f"{o['mean']:.1f}")
        M(f"fsmFracBase{tag}", f"{100 * b['frac_of_limit']:.0f}"); M(f"fsmFracOracle{tag}", f"{100 * o['frac_of_limit']:.0f}")
        M(f"fsmHitBase{tag}", f"{100 * (b['hit_rate'] or 0):.0f}"); M(f"fsmHitOracle{tag}", f"{100 * (o['hit_rate'] or 0):.0f}")
        M(f"fsmHitOpt{tag}", f"{c['opt_hit_rate']:.0f}")
        M(f"fsmModelF{tag}", f"{1e3 / c['model_terms']['v_F_ms']:.1f}")
        if o["foresight_recovered"] is not None:
            M(f"fsmRecovered{tag}", f"{100 * o['foresight_recovered']:.0f}")
        M(f"fsmClosed{tag}", f"{100 * o['gap_closed']:.0f}")
        gains.append(o["gain"]); closed.append(o["gap_closed"]); fr0.append(o["frac_of_limit"]); frb.append(b["frac_of_limit"])
        hit0.append(o["hit_rate"] or 0); hitb.append(b["hit_rate"] or 0); admits.append(o["admits_per_token"] / max(1e-9, c["opt_reads_per_token"]))
        lawerr.append(o["law_ms"] / o["ms"] - 1)
        if o["foresight_recovered"] is not None:
            recov.append(o["foresight_recovered"])
        if "noovl" in r:
            noovl.append(1 - r["noovl"]["ratio_to_base"][0])
        if "allcpu" in r:
            allcpu.append(1 - r["allcpu"]["ratio_to_base"][0])
        if c.get("w50_measured") is not None and c["w50_measured"] != float("inf"):
            w50m.append((tag, c["w50_measured"], c.get("w50_trace")))
    hostcells = [c for c in host if "oracle_w0" in c["runs"]]
    hg = [c["runs"]["oracle_w0"]["gain"] for c in hostcells]
    hc = [c["runs"]["oracle_w0"]["gap_closed"] for c in hostcells]
    hrec = [c["runs"]["oracle_w0"]["foresight_recovered"] for c in hostcells if c["runs"]["oracle_w0"]["foresight_recovered"] is not None]
    hf0 = [c["runs"]["oracle_w0"]["frac_of_limit"] for c in hostcells]; hfb = [c["runs"]["base"]["frac_of_limit"] for c in hostcells]
    M("fsmHostGainMin", f"{100 * min(hg):.0f}"); M("fsmHostGainMax", f"{100 * max(hg):.0f}")
    M("fsmHostClosedMin", f"{100 * min(hc):.0f}"); M("fsmHostClosedMax", f"{100 * max(hc):.0f}")
    if hrec:
        M("fsmHostRecovMin", f"{100 * min(hrec):.0f}"); M("fsmHostRecovMax", f"{100 * max(hrec):.0f}")
    M("fsmHostFracMin", f"{100 * min(hf0):.0f}"); M("fsmHostFracMax", f"{100 * max(hf0):.0f}")
    M("fsmHostFracBaseMin", f"{100 * min(hfb):.0f}"); M("fsmHostFracBaseMax", f"{100 * max(hfb):.0f}")
    gg = [c["runs"]["oracle_w0"]["gain"] for c in gpu if "oracle_w0" in c["runs"]]
    if gg:
        M("fsmGpuGainMin", f"{100 * min(gg):.0f}"); M("fsmGpuGainMax", f"{100 * max(gg):.0f}")
    if noovl:
        M("fsmNoovlMin", f"{100 * min(noovl):.0f}"); M("fsmNoovlMax", f"{100 * max(noovl):.0f}")
    if allcpu:
        M("fsmAllcpuMin", f"{100 * min(allcpu):.0f}"); M("fsmAllcpuMax", f"{100 * max(allcpu):.0f}")
    M("fsmLawErrMin", f"{100 * min(lawerr):+.0f}"); M("fsmLawErrMax", f"{100 * max(lawerr):+.0f}")
    M("fsmAdmitRatioMin", f"{min(admits):.2f}"); M("fsmAdmitRatioMax", f"{max(admits):.2f}")
    for tag, w, wt in w50m:
        M(f"fsmWfifty{tag}", f"{w:.1f}")
    if w50m:
        M("fsmWfiftyMin", f"{min(w for _, w, _ in w50m):.1f}"); M("fsmWfiftyMax", f"{max(w for _, w, _ in w50m):.1f}")
    hb = d["host"]
    M("fsmHostCpu", f"{hb['B_c']:.0f}"); M("fsmHostLink", f"{hb['B_p']:.0f}"); M("fsmHostBoth", f"{hb['B_cp']:.0f}"); M("fsmHostMax", f"{hb['b_host_max']:.1f}")
    M("fsmCells", str(len([c for c in cells if "oracle_w0" in c["runs"]])))
    # the abstract's and the contributions' clauses
    M("fsMeasuredAbstract", f"; an oracle of future routing in the engine recovers \\\\fsmHostGainMin--\\\\fsmHostGainMax\\\\% of speed at the host-bound cells, "
                            f"\\\\fsmHostClosedMin--\\\\fsmHostClosedMax\\\\% of the gap to the limit")
    M("fsMeasuredContrib", " (it closes \\\\fsmHostClosedMin--\\\\fsmHostClosedMax\\\\% of the gap where the host binds)")
    with open(P("paper", "wsg_foresight.tex"), "w") as f:
        f.write("% generated by scripts/foresight_paper.py from prereg/foresight_093.json\n")
        for k in sorted(MACROS):
            f.write("\\newcommand{\\%s}{%s}\n" % (k, MACROS[k]))
    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    cols = {"GLow": "#1f5fa8", "GMid": "#5b9bd5", "QLow": "#c0572b", "QMid": "#e39a6b"}
    names = {"GLow": "gpt-oss 11%", "GMid": "gpt-oss 25%", "QLow": "Qwen3 12.5%", "QMid": "Qwen3 25%"}
    xs_all = []
    for c in hostcells:
        tag = tagof[(c["model"], c["C"])]
        r = c["runs"]
        pts = [(0.7, r["base"]["frac_of_limit"])]
        for w in (2, 4, 16, 64):
            if f"oracle_w{w}" in r:
                pts.append((w, r[f"oracle_w{w}"]["frac_of_limit"]))
        pts.append((256, r["oracle_w0"]["frac_of_limit"]))
        xs, ys = zip(*pts)
        ax.plot(xs, [100 * y for y in ys], "-o", color=cols[tag], ms=3.5, lw=1, label=names[tag])
        ax.plot([256], [100 * (1 / c["model_terms"]["v_F_ms"]) * (c["model_terms"]["limit_ms"])], marker="_", color=cols[tag], ms=9, mew=1.5, ls="")
        xs_all += list(xs)
    ax.set_xscale("log")
    ax.set_xticks([0.7, 2, 4, 16, 64, 256])
    ax.set_xticklabels(["none", "2", "4", "16", "64", "all"], fontsize=6)
    ax.set_xlabel("tokens of routing foresight W", fontsize=6.5)
    ax.set_ylabel("% of this host's limit", fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.set_ylim(0, 100)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    from matplotlib.lines import Line2D
    h = ax.get_legend_handles_labels()[0] + [Line2D([], [], marker="_", color="k", ls="", ms=9, mew=1.5, label="model: foresight fix alone")]
    ax.legend(handles=h, fontsize=5.2, frameon=False, loc="upper left", ncol=2, columnspacing=0.8, handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(P("paper", "figs", "foresight.pdf"))
    print("macros:", ", ".join(f"{k}={v}" for k, v in sorted(MACROS.items()) if not k.startswith("fsMeasured")))


if __name__ == "__main__":
    main()
