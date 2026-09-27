"""Phase 4, section 4.4: classify the gpt-oss-120b teacher-forced NLL anomaly (prereg/PROTOCOL.md).

Inputs, for gpt-oss-20b and -120b, over the 35 traced dataset conversations (1024-token versions):
  vLLM V0  score_<m>_D.jsonl        (the conversation as traced: final channel right after <|start|>assistant)
  vLLM V1  score_<m>_harmony.jsonl  (own greedy analysis message inserted before the final channel)
  vLLM V2  score_<m>_harmony.jsonl  ("Reasoning: low")
  fp32 collector  <m>_tok.npz       (implementation 1: per-token NLL of the same 1024-token sequences)
Regions: user = the user message's tokens; response = the final-channel content tokens.

Classes, applied in order: 1 implementation (vLLM vs collector differ by > 1 nat/token on average over the
affected conversations), 2 user-text, 3 format, 4 genuine (see the protocol for the thresholds).

    python scripts/harmony_class.py --results /home/claude/gpu-branch/results --out prereg/harmony
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl.traces import load_pack  # noqa: E402

HDR = 3            # <|channel|> final <|message|>


def find(results, name):
    h = sorted(glob.glob(os.path.join(results, "*", name)))
    return h[-1] if h else None


def mean_nll(nll, lo, hi):
    """mean NLL of tokens lo..hi-1 (token j is scored by nll[j-1])"""
    v = [x for x in nll[max(lo - 1, 0):hi - 1] if x is not None]
    return (float(np.sum(v)), len(v))


def model_scores(results, m, prompts):
    rows = {r["corpus_idx"]: r for r in map(json.loads, open(prompts)) if r["tok_row"] >= 0}
    d = {r["corpus_idx"]: r["nll"] for r in map(json.loads, open(find(results, f"score_{m}_D.jsonl")))}
    hp = find(results, f"score_{m}_harmony.jsonl")
    h = {r["corpus_idx"]: r for r in map(json.loads, open(hp))} if hp else {}
    pk = find(results, f"{m}_tok.npz")
    col = None
    if pk:
        p = load_pack(pk)
        col, off = {}, 0
        tok_rows = {r["tok_row"]: ci for ci, r in rows.items()}
        for i, L in enumerate(p["seq_lens"]):
            col[tok_rows[i]] = p["nll"][off:off + L - 1].tolist()
            off += L - 1
    out = {}
    for ci, r in rows.items():
        P, U = r["data_prompt_len"], r["user_start"]
        L = min(len(r["data_ids"]), 1024)
        if P + HDR >= L:
            continue
        o = out[ci] = {"user": mean_nll(d[ci], U, P), "resp": mean_nll(d[ci], P + HDR, L)}
        if col is not None:
            o["collector_resp"] = mean_nll(col[ci], P + HDR, L)
            o["collector_user"] = mean_nll(col[ci], U, P)
            a = np.array([x if x is not None else np.nan for x in d[ci][P + HDR - 1:L - 1]], float)
            b = np.array(col[ci][P + HDR - 1:L - 1], float)
            o["impl_absdiff"] = float(np.nanmean(np.abs(a - b)))
        if ci in h:
            for v in ("v1", "v2"):
                ids, nl, n = h[ci][f"{v}_ids"], h[ci][f"{v}_nll"], h[ci]["resp_len"]
                o[f"{v}_resp"] = mean_nll(nl, len(ids) - n, len(ids))
            o["has_analysis"] = h[ci]["has_analysis"]
    return out


def agg(per, key, ids=None):
    ids = ids if ids is not None else list(per)
    s = sum(per[c][key][0] for c in ids if key in per[c])
    n = sum(per[c][key][1] for c in ids if key in per[c])
    return s / n if n else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--prompts", default="data/prompts")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = {m: model_scores(a.results, m, os.path.join(a.prompts, f"{m}.jsonl")) for m in ("gpt-oss-20b", "gpt-oss-120b")
         if find(a.results, f"score_{m}_D.jsonl")}
    rep = {m: {k: agg(per, k) for k in ("user", "resp", "collector_user", "collector_resp", "v1_resp", "v2_resp")}
           for m, per in S.items()}
    for m, per in S.items():
        rep[m]["n_conv"] = len(per)
        d = [per[c]["impl_absdiff"] for c in per if "impl_absdiff" in per[c]]
        rep[m]["impl_absdiff_mean"] = float(np.mean(d)) if d else None
        rep[m]["with_own_analysis"] = sum(per[c].get("has_analysis", False) for c in per)
    cls = None
    if len(S) == 2:
        s, b = rep["gpt-oss-20b"], rep["gpt-oss-120b"]
        # affected conversations: 120b response NLL > 2x 20b's (collector numbers, implementation 1)
        both = [c for c in S["gpt-oss-120b"] if c in S["gpt-oss-20b"]]
        r = lambda m, c, k: (S[m][c][k][0] / S[m][c][k][1]) if k in S[m][c] and S[m][c][k][1] else np.nan
        aff = [c for c in both if r("gpt-oss-120b", c, "collector_resp") > 2 * r("gpt-oss-20b", c, "collector_resp")]
        impl = float(np.mean([S["gpt-oss-120b"][c]["impl_absdiff"] for c in aff])) if aff else None
        ratio = lambda k, k2=None: (b[k] / s[k2 or k]) if b.get(k) and s.get(k2 or k) else None
        rep["ratios"] = dict(resp_V0=ratio("resp"), user_V0=ratio("user"), resp_V1_same=ratio("v1_resp"),
                             resp_V2_same=ratio("v2_resp"), resp_V1_vs_20b_V0=ratio("v1_resp", "resp"),
                             resp_V2_vs_20b_V0=ratio("v2_resp", "resp"), collector_resp=ratio("collector_resp"))
        rep["affected"] = dict(n=len(aff), corpus_idx=aff, impl_absdiff_mean=impl)
        R = rep["ratios"]
        if impl is not None and impl > 1.0:
            cls = "1 implementation"
        elif R["resp_V0"] is not None and R["resp_V0"] <= 1.5 and R["user_V0"] is not None and R["user_V0"] >= 2:
            cls = "2 user-text"
        elif R["resp_V0"] is not None and R["resp_V0"] > 1.5 and any(
                v is not None and v <= 1.5 for v in (R["resp_V1_same"], R["resp_V2_same"])):
            cls = "3 format"
        elif R["resp_V0"] is not None:
            cls = "4 genuine"
        rep["class"] = cls
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(summary=rep, per_conversation=S), open(os.path.join(a.out, "harmony.json"), "w"), indent=1)
    L = ["# Phase 4.4: gpt-oss-120b NLL anomaly (35 traced conversations, nats per token)\n",
         "| model | user (vLLM V0) | response V0 | response V1 (own analysis) | response V2 (reasoning low) | "
         "response, fp32 collector | mean abs vLLM-collector diff |", "|---|---|---|---|---|---|---|"]
    f = lambda v: "–" if v is None else f"{v:.3f}"
    for m in rep:
        if m.startswith("gpt-oss"):
            x = rep[m]
            L.append(f"| {m} | {f(x['user'])} | {f(x['resp'])} | {f(x['v1_resp'])} | {f(x['v2_resp'])} | {f(x['collector_resp'])} | {f(x['impl_absdiff_mean'])} |")
    if "ratios" in rep:
        L += ["", "Ratios 120b / 20b: " + ", ".join(f"{k} {f(v)}" for k, v in rep["ratios"].items()),
              f"Affected conversations (collector response NLL of 120b > 2x 20b's): {rep['affected']['n']}; "
              f"mean abs vLLM-collector difference on them: {f(rep['affected']['impl_absdiff_mean'])} nats.",
              f"\n**Class: {rep['class']}**"]
    open(os.path.join(a.out, "harmony.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
