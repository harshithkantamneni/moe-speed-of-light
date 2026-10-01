"""Prediction scorecard for GPU jobs 073-087.

Reads prereg/scorecard_clauses.json (every committed prediction split into its separable clauses, each hand-scored
under the interval rule stated in that file's meta.ci_rule) and writes

  prereg/scorecard.json     tallies by job, by era (073-075 / 076-081 / 082-087), by type, by era x type, the
                            prediction-level comparison with the paper's log, and the flat clause list;
  paper/tab_scorecard.tex   a longtable (job, clause, type, measured, status) for the appendix.

    python scripts/scorecard.py [--clauses prereg/scorecard_clauses.json] [--json prereg/scorecard.json] [--tex paper/tab_scorecard.tex]

The script computes nothing statistical: every number comes from the clauses file, whose source fields name the
prereg JSON key or raw-row computation behind it.
"""
import argparse
import json
import os
import re
from collections import OrderedDict

STATUSES = ["held", "held (point)", "failed", "untested", "void"]
TYPES = ["sign", "threshold", "band", "equality"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tally(clauses):
    t = OrderedDict((s, 0) for s in STATUSES)
    for c in clauses:
        t[c["status"]] += 1
    t["clauses"] = len(clauses)
    t["held_any"] = t["held"] + t["held (point)"]
    scored = t["clauses"] - t["untested"] - t["void"]
    t["scored"] = scored
    t["held_share_of_scored"] = round(t["held"] / scored, 3) if scored else None
    t["held_any_share_of_scored"] = round(t["held_any"] / scored, 3) if scored else None
    return t


def by_type(clauses):
    return OrderedDict((ty, tally([c for c in clauses if c["type"] == ty])) for ty in TYPES)


def paper_family(s):
    s = (s or "").lower()
    if s.startswith("held"):
        return "held"
    if s.startswith("failed"):
        return "failed"
    if s.startswith("not scorable"):
        return "untested"
    if s.startswith("half"):
        return "half"
    return s


def compare_with_paper_log(clauses):
    """Prediction-level view: the paper's log gives one verdict per prediction; list where the clause verdicts differ."""
    downgraded, upgraded, half_resolved, held_to_point = [], [], [], []
    for c in clauses:
        if c["status"] == "void":
            continue  # job 084: the paper's log scores the carried-over predictions on 084b, as this scorecard does
        fam, st = paper_family(c["paper_log"]), c["status"]
        entry = {"id": c["id"], "short": c["short"], "paper_log": c["paper_log"], "status": st}
        if fam == "held" and st in ("failed", "untested"):
            downgraded.append(entry)
        elif fam in ("failed", "untested") and st in ("held", "held (point)"):
            upgraded.append(entry)
        elif fam == "half":
            half_resolved.append(entry)
        elif fam == "held" and st == "held (point)":
            held_to_point.append(entry)
    return OrderedDict(downgraded=downgraded, upgraded=upgraded, half_resolved=half_resolved, held_to_point=held_to_point)


def prediction_level(jobs):
    """Reproduce the paper's unit of counting (one verdict per prediction) from the clause file's paper_log field."""
    out = OrderedDict()
    for j in jobs:
        seen = OrderedDict()
        for c in j["clauses"]:
            seen.setdefault(c["prediction"], paper_family(c["paper_log"]))
        out[j["job"]] = OrderedDict(predictions=len(seen), held=sum(v == "held" for v in seen.values()),
                                    failed=sum(v == "failed" for v in seen.values()),
                                    other={k: v for k, v in seen.items() if v not in ("held", "failed")})
    return out


# ---------------------------------------------------------------- LaTeX
def tex(s):
    """Escape a plain-text cell for LaTeX, rendering the few operators the clause texts use."""
    s = str(s)
    marks = OrderedDict([(">=", r"$\ge$"), ("<=", r"$\le$"), ("+-", r"$\pm$"), ("|dNLL|", r"$|\Delta\mathrm{NLL}|$"),
                         ("<", r"$<$"), (">", r"$>$")])
    holes = {}
    for i, (k, v) in enumerate(marks.items()):
        key = "\x00%d\x00" % i
        if k in s:
            s = s.replace(k, key)
            holes[key] = v
    s = re.sub(r"(\d)x\b", lambda m: m.group(1) + "\x00times\x00", s)
    holes["\x00times\x00"] = r"$\times$"
    for ch, rep in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$"),
                    ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(ch, rep)
    for key, v in holes.items():
        s = s.replace(key, v)
    return s


def fmt_num(x, pad=False):
    """Print a number as the clauses file gives it; with pad, ratios and their bounds get three decimals."""
    if isinstance(x, float):
        decimals = len(repr(x).split(".")[1]) if "." in repr(x) else 0
        if pad and abs(x) < 10 and decimals <= 3:
            return "%.3f" % x
        return "%g" % x
    return str(x)


def fmt_measured(c):
    m, ci = c.get("measured"), c.get("ci")
    if m is None:
        return "--"
    if isinstance(m, dict):
        parts = []
        for k, v in m.items():
            cik = ci.get(k) if isinstance(ci, dict) else None
            if isinstance(v, list) and cik:
                vv = "; ".join("%s [%s, %s]" % (fmt_num(x, True), fmt_num(lo, True), fmt_num(hi, True)) for x, (lo, hi) in zip(v, cik))
            elif isinstance(v, list):
                vv = " / ".join(fmt_num(x) for x in v)
            else:
                vv = fmt_num(v)
            parts.append("%s: %s" % (k, vv))
        return "; ".join(parts)
    if isinstance(m, list):
        if ci and isinstance(ci, list) and ci and isinstance(ci[0], list):
            return "; ".join("%s [%s, %s]" % (fmt_num(v, True), fmt_num(lo, True), fmt_num(hi, True)) for v, (lo, hi) in zip(m, ci))
        return " / ".join(fmt_num(v) for v in m)
    if ci and isinstance(ci, list) and len(ci) == 2 and not isinstance(ci[0], list):
        return "%s [%s, %s]" % (fmt_num(m, True), fmt_num(ci[0], True), fmt_num(ci[1], True))
    return fmt_num(m)


def status_tex(s):
    return {"held": "held", "held (point)": "held (point)", "failed": r"\textbf{failed}", "untested": r"\emph{untested}",
            "void": r"\emph{void}"}[s]


def write_tex(d, jobs, path, era_tallies, total):
    lines = [
        "% Generated by scripts/scorecard.py from prereg/scorecard_clauses.json; do not edit by hand.",
        r"\begin{footnotesize}",
        r"\begin{longtable}{@{}p{0.08\textwidth}p{0.355\textwidth}p{0.07\textwidth}p{0.29\textwidth}p{0.10\textwidth}@{}}",
        r"\caption{Prediction scorecard, jobs 073--092: every committed prediction split into its separable clauses and "
        r"scored under one rule. \emph{Held}: the point estimate is on the predicted side and the 95\% paired-bootstrap "
        r"interval excludes the threshold (for a band, lies inside it). \emph{Held (point)}: the point estimate is on the "
        r"predicted side but the interval includes the threshold, or no interval exists. \emph{Untested}: the clause could "
        r"not be evaluated. \emph{Void}: the job was voided (084, a throttled card; its predictions are scored on 084b). "
        r"Types: sign (a direction), threshold (a numeric bar), band (an interval), equality (a deterministic outcome). "
        r"Measured values are ratios of mean speeds with 95\% intervals unless the clause says otherwise; "
        r"\texttt{prereg/scorecard\_clauses.json} names the source of every number. "
        + "Totals: %d clauses, %d held, %d held (point), %d failed, %d untested, %d void."
        % (total["clauses"], total["held"], total["held (point)"], total["failed"], total["untested"], total["void"])
        + r"}\label{tab:scorecard}\\",
        r"\toprule",
        r"Job & Clause & Type & Measured & Status \\\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Job & Clause & Type & Measured & Status \\\midrule",
        r"\endhead",
        r"\midrule\multicolumn{5}{r}{\emph{continued}}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    era_of = {}
    for era, js in d["meta"]["eras"].items():
        for jb in js:
            era_of[jb] = era
    last_era = None
    for j in jobs:
        era = era_of.get(j["job"])
        if era != last_era:
            t = era_tallies[era]
            lines.append(r"\multicolumn{5}{@{}l}{\textbf{Jobs %s}: %d clauses, %d held, %d held (point), %d failed, %d untested, %d void}\\"
                         % (era.replace("-", "--"), t["clauses"], t["held"], t["held (point)"], t["failed"], t["untested"], t["void"]))
            last_era = era
        if not j["clauses"]:
            lines.append(r"%s & \emph{%s} & -- & -- & -- \\" % (tex(j["job"]), tex(j.get("note", "no predictions of its own"))))
            continue
        for c in j["clauses"]:
            lines.append("%s P%s & %s & %s & %s & %s \\\\" % (
                tex(j["job"]), tex(c["id"].split("-P", 1)[1]), tex(c["short"]), tex(c["type"]), tex(fmt_measured(c)), status_tex(c["status"])))
    lines += [r"\end{longtable}", r"\end{footnotesize}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clauses", default=os.path.join(ROOT, "prereg", "scorecard_clauses.json"))
    ap.add_argument("--json", default=os.path.join(ROOT, "prereg", "scorecard.json"))
    ap.add_argument("--tex", default=os.path.join(ROOT, "paper", "tab_scorecard.tex"))
    a = ap.parse_args()
    d = json.load(open(a.clauses), object_pairs_hook=OrderedDict)
    jobs = d["jobs"]
    era_of = {jb: era for era, js in d["meta"]["eras"].items() for jb in js}
    flat = []
    for j in jobs:
        for c in j["clauses"]:
            bad = [k for k in ("id", "short", "prediction", "clause", "type", "status", "paper_log") if k not in c]
            assert not bad, "%s missing %s" % (c.get("id"), bad)
            assert c["status"] in STATUSES, c["id"]
            assert c["type"] in TYPES, c["id"]
            flat.append(OrderedDict(id=c["id"], job=j["job"], era=era_of[j["job"]], prediction=c["prediction"], short=c["short"],
                                    type=c["type"], status=c["status"], paper_log=c["paper_log"], measured=c.get("measured"),
                                    ci=c.get("ci"), source=c.get("source"), note=c.get("note", "")))
    total = tally(flat)
    eras = OrderedDict()
    for era in d["meta"]["eras"]:
        cs = [c for c in flat if c["era"] == era]
        t = tally(cs)
        t["by_type"] = by_type(cs)
        t["jobs"] = d["meta"]["eras"][era]
        eras[era] = t
    per_job = OrderedDict()
    for j in jobs:
        t = tally([c for c in flat if c["job"] == j["job"]])
        t["era"] = era_of[j["job"]]
        t["by_type"] = by_type([c for c in flat if c["job"] == j["job"]])
        per_job[j["job"]] = t
    out = OrderedDict(
        ci_rule=d["meta"]["ci_rule"],
        types=d["meta"]["types"],
        total=total,
        by_era=eras,
        by_type=by_type(flat),
        by_job=per_job,
        vs_paper_log=OrderedDict(prediction_level=prediction_level(jobs), changed=compare_with_paper_log(flat)),
        clauses=flat,
    )
    with open(a.json, "w") as f:
        json.dump(out, f, indent=1)
    write_tex(d, jobs, a.tex, eras, total)

    print("clauses %d: held %d, held (point) %d, failed %d, untested %d, void %d" % (
        total["clauses"], total["held"], total["held (point)"], total["failed"], total["untested"], total["void"]))
    for era, t in eras.items():
        print("  %s: %d clauses: held %d, held (point) %d, failed %d, untested %d, void %d" % (
            era, t["clauses"], t["held"], t["held (point)"], t["failed"], t["untested"], t["void"]))
        for ty, tt in t["by_type"].items():
            if tt["clauses"]:
                print("      %-9s %2d: held %d, held (point) %d, failed %d, untested %d, void %d" % (
                    ty, tt["clauses"], tt["held"], tt["held (point)"], tt["failed"], tt["untested"], tt["void"]))
    ch = out["vs_paper_log"]["changed"]
    for k, v in ch.items():
        print("  %s (%d): %s" % (k, len(v), ", ".join(e["id"] for e in v)))
    print("wrote", a.json, "and", a.tex)


if __name__ == "__main__":
    main()
