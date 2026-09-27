"""Scores the phase-3 hypotheses (H11-H14, prereg/PROTOCOL.md) on the test run (jobs 020a/020b) and computes the
fraction of the speed-of-light bound reached, with first-party A10 constants.

    python scripts/analyze_phase3.py --runs results/020a results/020b --pred prereg/a10_mb/predictions.json \
        --out prereg/a10_mb
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODELS = {  # tag -> (L, E, k, budgets in 1/8)
    "gpt-oss-20b-MXFP4": (24, 32, 4, [1, 2, 4]),
    "Qwen3-30B-A3B-Instruct-2507-Q4_K_M": (48, 128, 8, [1, 2, 4]),
    "Qwen3-30B-A3B-Instruct-2507-Q8_0": (48, 128, 8, [1, 2, 4]),
    "gpt-oss-120b-MXFP4": (36, 128, 4, [1, 2]),
}
SHORT = {"gpt-oss-20b-MXFP4": "gpt-oss-20b", "Qwen3-30B-A3B-Instruct-2507-Q4_K_M": "Qwen3-30B-A3B Q4_K_M",
         "Qwen3-30B-A3B-Instruct-2507-Q8_0": "Qwen3-30B-A3B Q8_0", "gpt-oss-120b-MXFP4": "gpt-oss-120b"}


def load_runs(path):
    by, order = {}, []
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            c = r.get("config", "")
            if c not in by:
                order.append(c)
            by.setdefault(c, []).append(r)
    return by, order


def agg(rows):
    ms = sum(r["decode_ms"] for r in rows)
    n = sum(r["n_decode"] for r in rows)
    nll = sum(r["nll_sum"] for r in rows) / max(1, sum(r["nll_n"] for r in rows))
    per_seq = [1000 * r["n_decode"] / r["decode_ms"] for r in rows]
    return dict(tok_s=1000 * n / ms, nll=nll, n_seq=len(rows), n_steps=n, per_seq_tok_s=per_seq)


def top1(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    return np.fromfile(path, dtype=np.int32).reshape(-1, 10)[:, 0]


def find(tag, runs_dirs, name):
    for d in runs_dirs:
        p = os.path.join(d, name.format(tag=tag))
        if os.path.exists(p):
            return p, d
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pred = json.load(open(args.pred))["predictions"]
    P = {(p["model"], round(p["budget"], 3), p["system"]): p for p in pred}

    res = {}
    for tag, (L, E, k, FR) in MODELS.items():
        pe, d = find(tag, args.runs, "{tag}_ec.jsonl")
        if pe is None:
            continue
        by, order = load_runs(pe)
        r = dict(budgets={})
        ag, _ = find(tag, args.runs, "{tag}_static_n0.jsonl")
        if ag:
            b0, _ = load_runs(ag)
            r["all_gpu"] = agg(list(b0.values())[0])
            t_ag = top1(os.path.join(d, f"{tag}_static_n0.bin"))
        else:
            t_ag = None
        for qi, q in enumerate(FR):
            C, n = E * q // 8, L - L * q // 8
            ps, _ = find(tag, args.runs, "{tag}_static_n%d.jsonl" % n)
            bs, _ = load_runs(ps)
            st = agg(list(bs.values())[0])
            t_st = top1(os.path.join(d, f"{tag}_static_n{n}.bin"))
            row = dict(C=C, n=n, llama_static=st)
            for j, name in enumerate(["mailbox_cache", "mailbox_layers", "split_cache"]):
                cfg = order[3 * qi + j]
                a = agg(by[cfg])
                stp = os.path.join(d, os.path.basename(cfg.split("stats=")[-1].split(":")[0])) if "stats=" in cfg else None
                if stp and os.path.exists(stp):
                    s = json.load(open(stp))
                    a["hit_rate"] = s["hit_rate"]
                    a["admits_per_step"] = s["admits"] / max(1, s["steps"])
                t = top1(os.path.join(d, f"{tag}_ec.bin.{3 * qi + j}"))
                if t is not None and t_st is not None:
                    m = min(len(t), len(t_st))
                    a["top1_agree_vs_llama"] = float((t[:m] == t_st[:m]).mean())
                a["speedup_vs_llama"] = a["tok_s"] / st["tok_s"]
                a["nll_rel_vs_llama"] = (a["nll"] - st["nll"]) / st["nll"]
                row[name] = a
            if t_ag is not None and t_st is not None:
                m = min(len(t_ag), len(t_st))
                row["all_gpu_top1_agree_vs_llama"] = float((t_ag[:m] == t_st[:m]).mean())
                row["all_gpu_nll_rel_vs_llama"] = (r["all_gpu"]["nll"] - st["nll"]) / st["nll"]
            for sysname, key in [("mailbox cache", "mailbox_cache"), ("mailbox static layers", "mailbox_layers"),
                                 ("llama.cpp static layers", None)]:
                p = P.get((tag, round(q / 8, 3), sysname))
                if p is None:
                    continue
                meas = row[key]["tok_s"] if key else st["tok_s"]
                (row[key] if key else row["llama_static"])["pred_tok_s"] = p["tok_s"]
                (row[key] if key else row["llama_static"])["ape"] = abs(p["tok_s"] - meas) / meas
            r["budgets"][q] = row
        res[tag] = r

    # hypotheses
    h11 = {SHORT[t]: r["budgets"][2]["mailbox_cache"]["speedup_vs_llama"] for t, r in res.items() if 2 in r["budgets"]}
    h12 = {f"{SHORT[t]} {q/8:.3f}": b["mailbox_layers"]["speedup_vs_llama"] for t, r in res.items() for q, b in r["budgets"].items()}
    h13n = {f"{SHORT[t]} {q/8:.3f}": b["mailbox_cache"]["nll_rel_vs_llama"] for t, r in res.items() for q, b in r["budgets"].items()}
    h13a = {f"{SHORT[t]} {q/8:.3f}": b["mailbox_cache"].get("top1_agree_vs_llama") for t, r in res.items() for q, b in r["budgets"].items()}
    apes = [x["ape"] for r in res.values() for b in r["budgets"].values()
            for x in (b["mailbox_cache"], b["mailbox_layers"], b["llama_static"]) if "ape" in x]
    H = dict(
        H11=dict(values=h11, holds=all(v >= 1.30 for v in h11.values()) and len(h11) == 4),
        H12=dict(values=h12, holds=all(v >= 1.05 for v in h12.values())),
        H13=dict(nll_rel=h13n, top1=h13a, holds=all(abs(v) <= 0.01 for v in h13n.values()) and all(v is not None and v >= 0.95 for v in h13a.values())),
        H14=dict(median_ape=float(np.median(apes)) if apes else None, max_ape=float(max(apes)) if apes else None, n=len(apes),
                 holds=bool(apes) and np.median(apes) <= 0.10 and max(apes) <= 0.20),
    )
    os.makedirs(args.out, exist_ok=True)
    json.dump(dict(results=res, hypotheses=H), open(os.path.join(args.out, "scored.json"), "w"), indent=1, default=float)
    with open(os.path.join(args.out, "scored.md"), "w") as f:
        f.write("# Phase 3: test run (020) against the pre-registered hypotheses\n\n")
        f.write("| model | budget | llama.cpp | helpers, static layers | split-graph cache | mailbox cache | speed-up | predicted | hit | NLL Δ | top-1 agree |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for t, r in res.items():
            for q, b in r["budgets"].items():
                mc = b["mailbox_cache"]
                f.write(f"| {SHORT[t]} | {q/8:.1%} | {b['llama_static']['tok_s']:.1f} | {b['mailbox_layers']['tok_s']:.1f} ({b['mailbox_layers']['speedup_vs_llama']:.2f}×) | "
                        f"{b['split_cache']['tok_s']:.1f} | {mc['tok_s']:.1f} | {mc['speedup_vs_llama']:.2f}× | {mc.get('pred_tok_s', float('nan')):.1f} | "
                        f"{mc.get('hit_rate', float('nan')):.3f} | {mc['nll_rel_vs_llama']*100:+.2f}% | {mc.get('top1_agree_vs_llama', float('nan')):.3f} |\n")
            if "all_gpu" in r:
                f.write(f"| {SHORT[t]} | all on GPU | {r['all_gpu']['tok_s']:.1f} | | | | | | | | |\n")
        f.write("\n")
        for h, v in H.items():
            f.write(f"- **{h}**: {'holds' if v['holds'] else 'does not hold'} — {json.dumps({k: v2 for k, v2 in v.items() if k != 'holds'}, default=lambda x: round(float(x), 4))}\n")
    print(open(os.path.join(args.out, "scored.md")).read())


if __name__ == "__main__":
    main()
