"""Job 096's registered predictions (jobs/096_factorial@vast.sh on the gpu branch, commit 767e07f), split into clauses
and scored per host from prereg/foresight_096<host>.json (scripts/foresight_stats.py), under the scorecard's interval
rule (prereg/scorecard_clauses.json "meta"). Writes the job's entries into prereg/scorecard_clauses.json (replacing
earlier 096 entries); run scripts/scorecard.py afterwards.

    python scripts/scorecard_096.py
"""
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731

HOSTS = [("096a", "O4", "RTX 5090 + Ryzen 9 9950X"), ("096b", "O5", "RTX 5090 + Ryzen 9 9950X3D")]
CELLS = ["gpt-oss 11%", "gpt-oss 25%", "gpt-oss 40%", "Qwen3 12.5%", "Qwen3 25%", "Qwen3 43.75%"]
HOSTBOUND = ["gpt-oss 11%", "gpt-oss 25%", "Qwen3 12.5%", "Qwen3 25%"]
SIM = {"fetch": dict(zip(CELLS, [39.3, 16.0, 7.5, 100.4, 44.9, 14.5])),
       "both2": dict(zip(CELLS, [46.3, 22.3, 12.0, 115.2, 61.4, 23.2]))}
SIM_HIT = dict(zip(CELLS, [72.7, 88.9, 94.8, 73.9, 88.3, 96.2]))


def reads(r):
    return r["misses_per_token"] + r["admits_per_token"] + r.get("prefetches_per_token", 0.0)


def status_band(m, lo, hi, ci=None):
    if not (lo <= m <= hi):
        return "failed"
    if ci is None:
        return "held (point)"
    return "held" if lo <= ci[0] and ci[1] <= hi else "held (point)"


def status_side(m, thr, above=True, ci=None):
    ok = m > thr if above else m < thr
    if not ok:
        return "failed"
    if ci is None:
        return "held (point)"
    return "held" if ((ci[0] > thr) if above else (ci[1] < thr)) else "held (point)"


def main():
    out = []
    per_host = {}
    for job, hname, hdesc in HOSTS:
        path = P("prereg", f"foresight_{job}.json")
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        cells = {c["label"]: c for c in d["cells"]}
        per_host[hname] = cells
        cl = []

        def add(cid, pred, short, clause, typ, qty, thr, meas, ci, st, dec=1, note=""):
            cl.append(dict(id=f"096{hname}-{cid}", short=f"{short} ({hname})", prediction=pred, clause=clause, type=typ, quantity=qty,
                           threshold=thr, source=f"prereg/foresight_{job}.json", measured=round(meas, dec) if meas is not None else None,
                           ci=[round(x, dec) for x in ci] if ci else None, status=st, paper_log="", note=note, decimals=dec))
        for lab in CELLS:
            c = cells.get(lab)
            if not c:
                continue
            R = c["runs"]
            tag = lab.replace(" ", "").replace("%", "")
            if "fetch" in R:
                e = 100 * (reads(R["fetch"]) / SIM["fetch"][lab] - 1)
                add(f"P1a-{tag}", 1, f"fetch reads within 6% of the simulation, {lab}", f"the fetch oracle's reads per token are within 6% of the simulation's ({SIM['fetch'][lab]}) at {lab}",
                    "band", "100 (reads / simulated - 1)", "-6 to +6%", e, None, status_band(e, -6, 6))
                hd = 100 * (R["fetch"]["hit_rate"] or 0) - SIM_HIT[lab]
                add(f"P1c-{tag}", 1, f"fetch hit rate within 2 points of the simulation, {lab}", f"the fetch oracle's hit rate is within 2 points of the simulation's ({SIM_HIT[lab]}%) at {lab}",
                    "band", "hit rate - simulated (points)", "-2 to +2", hd, None, status_band(hd, -2, 2))
            if "both2" in R:
                e = 100 * (reads(R["both2"]) / SIM["both2"][lab] - 1)
                add(f"P1b-{tag}", 1, f"both2 reads within 15% of the simulation, {lab}", f"fetch + lead-2 prefetch reads per token within 15% of the simulation's ({SIM['both2'][lab]}) at {lab}",
                    "band", "100 (reads / simulated - 1)", "-15 to +15%", e, None, status_band(e, -15, 15))
            if "foa" in R and "base" in R:
                e = 100 * (1 - reads(R["foa"]) / reads(R["base"]))
                add(f"P3a-{tag}", 3, f"foa reads 3-22% fewer than base, {lab}", f"fetch-on-admit reads 3-22% fewer host bytes per token than the online policy at {lab}",
                    "band", "100 (1 - foa reads / base reads)", "3 to 22%", e, None, status_band(e, 3, 22))
            if lab in HOSTBOUND:
                if "fetch" in R:
                    r = R["fetch"]["ratio_to_base"]
                    add(f"P2-{tag}", 2, f"fetch 1.15-1.45x base, {lab}", f"the fetch oracle runs 1.15-1.45x the online policy at {lab}", "band", "fetch / base speed", "1.15 to 1.45",
                        r[0], r[1:], status_band(r[0], 1.15, 1.45, r[1:]), 3)
                if "foa" in R:
                    r = R["foa"]["ratio_to_base"]
                    add(f"P3b-{tag}", 3, f"foa 0.97-1.12x base, {lab}", f"fetch-on-admit runs 0.97-1.12x the online policy at {lab}", "band", "foa / base speed", "0.97 to 1.12",
                        r[0], r[1:], status_band(r[0], 0.97, 1.12, r[1:]), 3)
                if all(k in R for k in ("fetch", "foa", "both2", "both3p")):
                    best = max(("both2", "both3p"), key=lambda k: R[k]["mean"])
                    hw = lambda k: 0.5 * (R[k]["ratio_to_base"][2] - R[k]["ratio_to_base"][1])  # noqa: E731
                    rb, rf, ra = R[best]["ratio_to_base"][0], R["fetch"]["ratio_to_base"][0], R["foa"]["ratio_to_base"][0]
                    m1 = rb - rf; m2 = rf - ra
                    t1 = max(hw(best), hw("fetch")); t2 = max(hw("fetch"), hw("foa"))
                    st = "held" if (m1 > t1 and m2 > t2) else ("held (point)" if (m1 > 0 and m2 > 0) else "failed")
                    add(f"P4-{tag}", 4, f"best prefetch > fetch > foa, {lab}", f"at {lab} the faster of both2/both3p beats fetch and fetch beats foa, each by more than the larger interval half-width",
                        "sign", "ratio differences (prefetch - fetch, fetch - foa)", "> half-widths", min(m1 - t1, m2 - t2), None, st, 3,
                        note=f"{best} - fetch = {m1:.3f} (half-width {t1:.3f}); fetch - foa = {m2:.3f} (half-width {t2:.3f})")
                if "lead2" in R and "bypass" in R:
                    m = R["lead2"]["mean"] / R["bypass"]["mean"]
                    add(f"P7b-{tag}", 7, f"lead2 faster than bypass, {lab}", f"the single-read lead-2 prefetch runs faster than serve-then-copy MIN with bypass at {lab}", "sign",
                        "lead2 / bypass speed", "> 1", m, None, status_side(m, 1.0), 3)
            if "nb2" in R and "lead2" in R:
                m = reads(R["nb2"]) / reads(R["lead2"])
                add(f"P5a-{tag}", 5, f"nb2 reads 1.25-1.9x lead2, {lab}", f"Belady single-read prefetch reads 1.25-1.9x the MIN-with-bypass prefetch's at {lab}", "band",
                    "nb2 reads / lead2 reads", "1.25 to 1.9", m, None, status_band(m, 1.25, 1.9), 2)
            if lab in ("gpt-oss 11%", "Qwen3 12.5%"):
                if "nb2" in R and "lead2" in R:
                    m = R["nb2"]["mean"] / R["lead2"]["mean"]
                    add(f"P5b-{tag}", 5, f"nb2 slower than lead2, {lab}", f"nb2 runs slower than lead2 at {lab}", "sign", "nb2 / lead2 speed", "< 1", m, None, status_side(m, 1.0, above=False), 3)
                if "nb2" in R:
                    r = R["nb2"]["ratio_to_base"]
                    add(f"P5c-{tag}", 5, f"nb2 at most 1.05x base, {lab}", f"nb2 runs no faster than 1.05x the online policy at {lab}", "threshold", "nb2 / base speed", "<= 1.05",
                        r[0], r[1:], status_side(r[0], 1.05, above=False, ci=r[1:]), 3)
                if "bypass" in R:
                    r = R["bypass"]["ratio_to_base"]
                    add(f"P7a-{tag}", 7, f"bypass gains at most 8%, {lab}", f"serve-then-copy MIN with bypass gains at most 8% over the online policy at {lab}", "threshold", "bypass / base speed",
                        "<= 1.08", r[0], r[1:], status_side(r[0], 1.08, above=False, ci=r[1:]), 3)
            if lab == "Qwen3 12.5%" and "nb2" in R and "hitopt" in R:
                m = R["nb2"]["mean"] / R["hitopt"]["mean"]
                add(f"P6a-{tag}", 6, f"nb2 faster than hitopt, {lab}", f"nb2 runs faster than the unpaced Belady prefetch at {lab}", "sign", "nb2 / hitopt speed", "> 1", m, None, status_side(m, 1.0), 3)
            if lab.startswith("gpt-oss") and "nb2" in R and "hitopt" in R:
                m = 100 * (R["nb2"]["mean"] / R["hitopt"]["mean"] - 1)
                add(f"P6b-{tag}", 6, f"nb2 within 10% of hitopt, {lab}", f"nb2 runs within 10% of the unpaced Belady prefetch at {lab}", "band", "100 (nb2 / hitopt - 1)", "-10 to +10%", m, None, status_band(m, -10, 10))
            if lab in ("gpt-oss 25%", "Qwen3 25%") and "hitoptp" in R and "hitopt" in R:
                m = R["hitoptp"]["mean"] / R["hitopt"]["mean"]
                add(f"P8-{tag}", 8, f"hitoptp faster than hitopt, {lab}", f"the paced Belady prefetch runs faster than the unpaced one at {lab}", "sign", "hitoptp / hitopt speed", "> 1", m, None, status_side(m, 1.0), 3)
            if lab in HOSTBOUND and all(k in R for k in ("base", "fetch", "both2", "both3p")):
                lim = c["model_terms"]["limit_ms"]
                base_ms = R["base"]["ms"]; gap = base_ms - lim
                b_ms = min(R["both2"]["ms"], R["both3p"]["ms"])
                sb = 100 * (base_ms - R["fetch"]["ms"]) / gap; so = 100 * (R["fetch"]["ms"] - b_ms) / gap
                add(f"P9a-{tag}", 9, f"bytes share 25-55% of the gap, {lab}", f"base -> fetch is 25-55% of the gap to the limit at {lab}", "band", "100 (base - fetch) / (base - limit), ms", "25 to 55%", sb, None, status_band(sb, 25, 55))
                add(f"P9b-{tag}", 9, f"overlap share 5-30% of the gap, {lab}", f"fetch -> the faster prefetch state is 5-30% of the gap at {lab}", "band", "100 (fetch - best) / (base - limit), ms", "5 to 30%", so, None, status_band(so, 5, 30))
        out.append(dict(job=f"096{hname}", era="088-097", script="jobs/096_factorial@vast.sh", commit="767e07f", host=hdesc + f" (host {hname})",
                        outcome_note="prereg/foresight_outcome_096.md", clauses=cl))
    # prediction 10: between hosts
    if len(per_host) == 2:
        (h1, c1), (h2, c2) = per_host.items()
        cl = []
        for lab in CELLS:
            if lab in c1 and lab in c2 and "fetch" in c1[lab]["runs"] and "fetch" in c2[lab]["runs"]:
                m = abs(c1[lab]["runs"]["fetch"]["ratio_to_base"][0] - c2[lab]["runs"]["fetch"]["ratio_to_base"][0])
                tag = lab.replace(" ", "").replace("%", "")
                cl.append(dict(id=f"096-P10-{tag}", short=f"fetch/base differs by at most 0.15 between hosts, {lab}", prediction=10,
                               clause=f"between the two hosts, fetch / base differs by at most 0.15 at {lab}", type="threshold", quantity="|fetch/base (O4) - fetch/base (O5)|",
                               threshold="<= 0.15", source="prereg/foresight_096a.json, prereg/foresight_096b.json", measured=round(m, 3), ci=None,
                               status="held (point)" if m <= 0.15 else "failed", paper_log="", note="", decimals=3))
        out.append(dict(job="096", era="088-097", script="jobs/096_factorial@vast.sh", commit="767e07f", host="O4 and O5", outcome_note="prereg/foresight_outcome_096.md", clauses=cl))
    path = P("prereg", "scorecard_clauses.json")
    d = json.load(open(path))
    d["jobs"] = [j for j in d["jobs"] if not str(j.get("job", "")).startswith("096")] + out
    json.dump(d, open(path, "w"), indent=1, ensure_ascii=False)
    for j in out:
        n = len(j["clauses"]); held = sum(c["status"] == "held" for c in j["clauses"]); pt = sum(c["status"] == "held (point)" for c in j["clauses"])
        print(f"{j['job']}: {n} clauses, {held} held, {pt} held (point), {n - held - pt} failed")
        for c in j["clauses"]:
            if c["status"] == "failed":
                print(f"   FAILED {c['id']}: {c['short']}: {c['measured']} vs {c['threshold']}")


if __name__ == "__main__":
    main()
