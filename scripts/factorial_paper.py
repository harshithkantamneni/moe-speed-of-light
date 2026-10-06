"""Macros for Section 4 from the randomised factorial (job 096 on hosts O4 and O5, prereg/foresight_096a.json and
foresight_096b.json) and, beside it, job 095 on O3. Writes paper/wsg_factorial.tex.

    python scripts/factorial_paper.py
"""
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
HB = ["gpt-oss 11%", "gpt-oss 25%", "Qwen3 12.5%", "Qwen3 25%"]
GB = ["gpt-oss 40%", "Qwen3 43.75%"]
ALL = HB[:2] + GB[:1] + HB[2:] + GB[1:]
M = {}


def put(k, v):
    assert not any(ch.isdigit() for ch in k), k
    M[k] = v


def reads(r):
    return r["misses_per_token"] + r["admits_per_token"] + r.get("prefetches_per_token", 0.0)


def rng(name, vals, fmt="{:.2f}"):
    put(name + "Min", fmt.format(min(vals))); put(name + "Max", fmt.format(max(vals)))


def main():
    H = {}
    for job, tag in (("096a", "Four"), ("096b", "Five"), ("095", "Three")):
        d = json.load(open(P("prereg", f"foresight_{job}.json")))
        H[tag] = dict(host=d["host"], cells={c["label"]: c for c in d["cells"]})
    for tag in ("Four", "Five", "Three"):
        h = H[tag]["host"]
        put(f"fxLink{tag}", f"{h['B_p']:.0f}"); put(f"fxCpu{tag}", f"{h['B_c']:.0f}"); put(f"fxBest{tag}", f"{h['b_host_max']:.0f}")
    new = [H["Four"]["cells"], H["Five"]["cells"]]
    allh = new + [H["Three"]["cells"]]

    def ratios(cfg, cells, hosts):
        return [h[c]["runs"][cfg]["ratio_to_base"][0] for h in hosts for c in cells if c in h and cfg in h[c]["runs"]]
    for cfg, nm in (("foa", "Foa"), ("fetch", "Fetch"), ("both3p", "Paced"), ("nb2", "Nbtwo"), ("hitopt", "Hitopt"), ("bypass", "Bypass"),
                    ("lead2", "Lead"), ("hitoptp", "Hitoptp")):
        for tag, hs in (("Four", [H["Four"]["cells"]]), ("Five", [H["Five"]["cells"]])):
            v = ratios(cfg, HB, hs)
            if v:
                rng(f"fx{nm}{tag}", v)
            v = ratios(cfg, GB, hs)
            if v:
                rng(f"fx{nm}Gpu{tag}", v)
        v = ratios(cfg, HB, new)
        if v:
            rng(f"fx{nm}Host", v)
    # reads: foa vs base, nb2 vs lead2, fetch vs R* (opt)
    fr = [100 * (1 - reads(h[c]["runs"]["foa"]) / reads(h[c]["runs"]["base"])) for h in new for c in ALL]
    rng("fxFoaFewerReads", fr, "{:.0f}")
    nl = [reads(h[c]["runs"]["nb2"]) / reads(h[c]["runs"]["lead2"]) for h in new for c in ALL]
    rng("fxNbReadsOverLead", nl)
    no = [reads(h[c]["runs"]["nb2"]) / h[c]["opt_reads_per_token"] for h in new for c in ALL]
    rng("fxNbReadsOverOpt", no, "{:.1f}")
    fo = [reads(h[c]["runs"]["fetch"]) / h[c]["opt_reads_per_token"] for h in new for c in ALL]
    rng("fxFetchReadsOverOpt", fo)
    # how often MIN with bypass admits (the fetch oracle's forced fetches) against the online policy's admissions
    ad = [h[c]["runs"]["fetch"]["oracle_forced_per_token"] / h[c]["runs"]["base"]["admits_per_token"] for h in new for c in HB]
    rng("fxMinAdmitsOverOnline", ad, "{:.0f}")
    oa = [h[c]["runs"]["base"]["admits_per_token"] for h in new for c in ALL]
    rng("fxOnlineAdmits", oa, "{:.1f}")
    # nb2 against hitopt (the second read alone), by model
    g = [h[c]["runs"]["nb2"]["mean"] / h[c]["runs"]["hitopt"]["mean"] for h in new for c in ALL if c.startswith("gpt")]
    rng("fxNbOverHitGpt", g)
    q = [h[c]["runs"]["nb2"]["mean"] / h[c]["runs"]["hitopt"]["mean"] for h in new for c in ["Qwen3 12.5%"]]
    rng("fxNbOverHitQlow", q)
    # fetch / base spread between the hosts
    sp = [abs(H["Four"]["cells"][c]["runs"]["fetch"]["ratio_to_base"][0] - H["Five"]["cells"][c]["runs"]["fetch"]["ratio_to_base"][0]) for c in ALL]
    rng("fxFetchSpread", sp)
    # fraction of the limit: online and the best state, host-bound, new hosts
    fb = [100 * h[c]["runs"]["base"]["frac_of_limit"] for h in new for c in HB]
    rng("fxBaseFrac", fb, "{:.0f}")
    bf = [100 * max(h[c]["runs"][k]["frac_of_limit"] for k in ("both2", "both3p")) for h in new for c in HB]
    rng("fxBestFrac", bf, "{:.0f}")
    # largest interval half-width of any ratio on the new hosts
    hw = [0.5 * (r["ratio_to_base"][2] - r["ratio_to_base"][1]) for h in new for c in ALL for k, r in h[c]["runs"].items() if k != "base"]
    put("fxHalfWidthMax", f"{max(hw):.3f}")
    # gains over all cells on all three hosts (percent)
    al = [100 * (h[c]["runs"]["fetch"]["ratio_to_base"][0] - 1) for h in allh for c in ALL if "fetch" in h[c]["runs"]]
    rng("fxFetchAllGain", al, "{:.0f}")
    al = [100 * (h[c]["runs"]["both3p"]["ratio_to_base"][0] - 1) for h in allh for c in ALL if "both3p" in h[c]["runs"]]
    rng("fxPacedAllGain", al, "{:.0f}")
    by = [reads(h[c]["runs"]["bypass"]) / h[c]["opt_reads_per_token"] for h in new for c in ALL]
    rng("fxBypassReadsOverOpt", by, "{:.1f}")
    # the running example on O4
    c = H["Four"]["cells"]["gpt-oss 25%"]["runs"]
    for k, nm in (("foa", "Foa"), ("fetch", "Fetch"), ("both3p", "Paced"), ("nb2", "Nbtwo"), ("hitopt", "Hitopt"), ("bypass", "Bypass")):
        put(f"fx{nm}GMidFour", f"{c[k]['ratio_to_base'][0]:.2f}")
    with open(P("paper", "wsg_factorial.tex"), "w") as f:
        f.write("% generated by scripts/factorial_paper.py from prereg/foresight_095.json, foresight_096a.json, foresight_096b.json\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    for k in sorted(M):
        print(f"{k:28s} {M[k]}")


if __name__ == "__main__":
    main()
