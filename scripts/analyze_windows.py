"""Phase 4.5: score the A10 re-run with correct decode windows (whole prompt prefilled, response tokens decoded).

H16: at 25 % the mailbox cache is >= 1.30x llama.cpp static layers on each model (per arm).
H17: the phase-3 step-time model, unchanged (fitted on job 019), with per-step counts from routing traces of
     the same windows, predicts these runs within 10 % median APE.
Also compares every configuration with its phase-3 value (jobs 020a/b, windows partly inside the user prompt).

    python scripts/analyze_windows.py --run <025 results dir> --arm D --out prereg/a10_windows
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl.ecsim import load_steps, simulate  # noqa: E402
from scripts.mb_model import MODELS  # noqa: E402

N_PREFILL = 640          # >= every test prompt: the window starts at the response
N_DECODE = 192


def rate(rows):
    return sum(r["n_decode"] for r in rows) / (sum(r["decode_ms"] for r in rows) / 1000)


def load_runs(d, tag):
    """{'mailbox cache': {q: row list}, 'mailbox static layers': {...}, 'llama.cpp static layers': {...}, 'all-GPU': rows}"""
    L, E, k, K, FR, *_ = MODELS[tag.rsplit("_", 1)[0]]
    out = {"mailbox cache": {}, "mailbox static layers": {}, "llama.cpp static layers": {}}
    p = os.path.join(d, f"{tag}_ec.jsonl")
    if os.path.exists(p):
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            cfg = r["config"]
            if "alloc=" in cfg:
                n = int(cfg.split("alloc=")[1].split(".txt")[0].rsplit("_n", 1)[1])
                q = next(q for q in range(1, 8) if L - L * q // 8 == n)
                out["mailbox static layers"].setdefault(q, []).append(r)
            else:
                C = int(cfg.split("slots=")[1].split(":")[0])
                out["mailbox cache"].setdefault(C * 8 // E, []).append(r)
    for p in glob.glob(os.path.join(d, f"{tag}_static_n*.jsonl")):
        n = int(p.rsplit("_n", 1)[1].split(".")[0])
        rows = [json.loads(l) for l in open(p) if l.strip()]
        if n == 0:
            out["all-GPU"] = rows
        else:
            out["llama.cpp static layers"][next(q for q in range(1, 8) if L - L * q // 8 == n)] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--arm", default="D")
    ap.add_argument("--phase3", default="prereg/a10_mb/scored.json")
    ap.add_argument("--fits", default="prereg/a10_mb/predictions.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fits = json.load(open(a.fits))["fits"]
    p3 = json.load(open(a.phase3))["results"] if os.path.exists(a.phase3) else {}
    res, preds = {}, []
    for tag, (L, E, k, K, FR, tdir, corp, test) in MODELS.items():
        runs = load_runs(a.run, f"{tag}_{a.arm}")
        if not runs["llama.cpp static layers"]:
            continue
        r = res[tag] = {"budgets": {}}
        if "all-GPU" in runs:
            r["all_gpu"] = dict(tok_s=rate(runs["all-GPU"]))
        counts = {}
        if tdir is not None and a.arm == "D":
            R, E2, win = load_steps(tdir, corp, test, N_PREFILL, N_DECODE)
            r["decode_steps_traced"] = int(sum(b - s for _, s, b in win))
            for q in FR:
                h, m, adm = simulate(R, E2, E * q // 8, "dfa", kappa=float(K))
                counts[q] = (float((m > 0).sum(1).mean()), float(m.sum(1).mean()), float(adm.sum(1).mean()),
                             float(h.sum() / (h.sum() + m.sum())))
        f = fits.get(tag)
        for q in FR:
            b = r["budgets"][str(q)] = {}
            for sysname in ("mailbox cache", "mailbox static layers", "llama.cpp static layers"):
                rows = runs[sysname].get(q)
                if rows:
                    b[sysname] = dict(tok_s=rate(rows), steps=sum(x["n_decode"] for x in rows),
                                      nll=sum(x["nll_sum"] for x in rows) / sum(x["nll_n"] for x in rows))
            if "mailbox cache" in b and "llama.cpp static layers" in b:
                b["speedup"] = b["mailbox cache"]["tok_s"] / b["llama.cpp static layers"]["tok_s"]
            old = p3.get(tag, {}).get("budgets", {}).get(str(q), {})
            if old:
                b["phase3"] = {s: old.get(key, {}).get("tok_s") for s, key in
                               (("mailbox cache", "mailbox_cache"), ("llama.cpp static layers", "llama_static"),
                                ("mailbox static layers", "mailbox_layers"))}
            if f and q in counts:
                n = L - L * q // 8
                rr, mu, al, hit = counts[q]
                t_mb = f["T0"] + f["a0"] * rr + f["a1"] * mu + f["beta"] * al
                t_ml = f["T0"] + f["a0"] * n + f["a1"] * n * k
                t_st = f["stock_T0"] + f["stock_s"] * n
                b["sim_hit_rate"] = hit
                for sysname, t in (("mailbox cache", t_mb), ("mailbox static layers", t_ml), ("llama.cpp static layers", t_st)):
                    if sysname in b:
                        pred = 1000 / t
                        b[sysname]["pred_tok_s"] = pred
                        b[sysname]["ape"] = abs(pred - b[sysname]["tok_s"]) / b[sysname]["tok_s"]
                        preds.append(b[sysname]["ape"])
    q25 = {tag: r["budgets"].get("2", {}).get("speedup") for tag, r in res.items()}
    h16 = all(v is not None and v >= 1.30 for v in q25.values())
    h17 = dict(n=len(preds), median_ape=float(np.median(preds)) if preds else None,
               max_ape=float(np.max(preds)) if preds else None)
    h17["holds"] = h17["median_ape"] is not None and h17["median_ape"] <= 0.10
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(arm=a.arm, results=res, H16=dict(speedup_25=q25, holds=h16), H17=h17),
              open(os.path.join(a.out, f"scored_{a.arm}.json"), "w"), indent=1)
    lines = [f"# Phase 4.5, arm {a.arm}: A10, whole prompt prefilled, response tokens decoded\n",
             "| model | budget | llama.cpp | helpers, layers | mailbox cache | speed-up | phase 3 speed-up | DFA hit (sim) | pred. cache / llama |",
             "|---|---|---|---|---|---|---|---|---|"]
    for tag, r in res.items():
        for q, b in r["budgets"].items():
            g = lambda s, f="tok_s": b.get(s, {}).get(f)
            ph = b.get("phase3", {})
            ph_sp = (ph["mailbox cache"] / ph["llama.cpp static layers"]) if ph and ph.get("mailbox cache") and ph.get("llama.cpp static layers") else None
            fmt = lambda v, s="{:.1f}": "–" if v is None else s.format(v)
            lines.append(f"| {tag} | {int(q)/8:.1%} | {fmt(g('llama.cpp static layers'))} | {fmt(g('mailbox static layers'))} | "
                         f"{fmt(g('mailbox cache'))} | {fmt(b.get('speedup'), '{:.2f}x')} | {fmt(ph_sp, '{:.2f}x')} | "
                         f"{fmt(b.get('sim_hit_rate'), '{:.3f}')} | {fmt(g('mailbox cache', 'pred_tok_s'))} / {fmt(g('llama.cpp static layers', 'pred_tok_s'))} |")
        if "all_gpu" in r:
            lines.append(f"| {tag} | all-GPU | {r['all_gpu']['tok_s']:.1f} | | | | | | |")
    lines += ["", f"**H16** (>= 1.30x at 25 % on every model): {'holds' if h16 else 'does not hold'} "
              "(" + ", ".join(f"{t}: {v:.2f}x" for t, v in q25.items() if v) + ").",
              f"**H17** (median APE <= 10 %): {'holds' if h17['holds'] else 'does not hold'} — median {h17['median_ape']*100:.1f} %, "
              f"max {h17['max_ape']*100:.1f} %, n = {h17['n']}." if h17["n"] else "**H17**: no traced configurations."]
    open(os.path.join(a.out, f"scored_{a.arm}.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
