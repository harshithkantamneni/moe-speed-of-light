"""Fraction of the speed-of-light bound reached on the A10 (phase 4.5 windows: whole prompt prefilled, first <= 192
response tokens decoded), per model and arm, with routing from the matching trace pack (m5, m6 of phase 4).

    python scripts/sol_windows.py --results /home/claude/gpu-branch/results --scored prereg/a10_windows/scored_D.json \
        --arm D --out prereg/a10_windows/sol_D.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl import cachesim, ecsim_fast  # noqa: E402
from mosl.traces import load_pack  # noqa: E402
from scripts.sol_a10 import B_C, B_G, B_P, bound, dense_bytes  # noqa: E402

TEST = {"gpt-oss": [7, 10, 12, 15, 16, 18, 19, 20, 25, 26, 27, 31], "qwen": [7, 10, 12, 15, 16, 18, 19, 20, 25, 26, 27, 32]}
MODELS = {  # A10 tag -> (prompts key, pack names by arm, E, k, kappa, expert bytes, dense bytes incl. LM head, all-GPU tag, budgets)
    "gpt-oss-20b-MXFP4": ("gpt-oss-20b", {"D": "gpt-oss-20b_tok.npz", "S": "gpt-oss-20b_S.npz", "G": "gpt-oss-20b_G.npz"},
                          32, 4, 1.0, 13253760, dense_bytes("gpt-oss-20b-MXFP4"), "gpt-oss-20b-MXFP4", (1, 2, 4)),
    "Qwen3-30B-A3B-Instruct-2507-Q4_K_M": ("qwen3-30b-a3b", {"D": "qwen3-30b-a3b_tok.npz", "S": "qwen3-30b-a3b_fp8_S.npz",
                                                             "G": "qwen3-30b-a3b_fp8_G.npz"},
                                           128, 8, 2.0, 2800000, dense_bytes("Qwen3-30B-A3B-Instruct-2507-Q4_K_M"),
                                           "Qwen3-30B-A3B-Instruct-2507-Q4_K_M", (1, 2, 4)),
    "Qwen3-30B-A3B-Instruct-2507-Q8_0": ("qwen3-30b-a3b", {"D": "qwen3-30b-a3b_tok.npz", "S": "qwen3-30b-a3b_fp8_S.npz",
                                                           "G": "qwen3-30b-a3b_fp8_G.npz"},
                                         128, 8, 2.0, 5013504, dense_bytes("Qwen3-30B-A3B-Instruct-2507-Q8_0"),
                                         "Qwen3-30B-A3B-Instruct-2507-Q4_K_M", (1, 2, 4)),
}
N_DECODE = 192


def find(results, name):
    h = sorted(glob.glob(os.path.join(results, "*", name)))
    return h[-1] if h else None


def windows(pack, prompts_file, test_rows):
    """routes [L, T, k] of the first <= 192 response tokens of the test rows, back to back"""
    rows = {r["tok_row"]: r["corpus_idx"] for r in map(json.loads, open(prompts_file)) if r["tok_row"] >= 0}
    pos = {int(c): i for i, c in enumerate(pack["corpus_idx"])}
    if pack["meta"]["corpus"].startswith("tok_"):   # dataset token file: pack rows are token-file rows
        pos = {rows[i]: i for i in range(len(pack["seq_lens"]))}
    idx = []
    for s in test_rows:
        i = pos[rows[s]]
        P, L, s0 = int(pack["prompt_lens"][i]), int(pack["seq_lens"][i]), int(pack["starts"][i])
        n = min(N_DECODE, L - P)
        idx.append(np.arange(s0 + P, s0 + P + n))
    return pack["routes"][:, np.concatenate(idx)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--prompts", default="data/prompts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bc", type=float, default=B_C / 1e9, help="host read bandwidth of the measured instance, GB/s")
    a = ap.parse_args()
    bc = a.bc * 1e9
    sc = json.load(open(a.scored))["results"]
    out = {}
    for tag, (pk, packs, E, k, K, s, D, ref, FR) in MODELS.items():
        p = find(a.results, packs[a.arm])
        if not p or tag not in sc or ref not in sc or "all_gpu" not in sc[ref]:
            print(tag, "skipped (pack or measurements missing)")
            continue
        R = windows(load_pack(p), os.path.join(a.prompts, f"{pk}.jsonl"), TEST["gpt-oss" if "gpt" in tag else "qwen"])
        L = R.shape[0]
        Lr, sr, Dr = L, MODELS[ref][5], MODELS[ref][6]
        eta = (Dr + Lr * k * sr) * sc[ref]["all_gpu"]["tok_s"] / B_G
        a_, b_, p_ = s / (eta * B_G), s / bc, s / B_P
        TD = D / (eta * B_G)
        rows = {}
        for q in FR:
            C = E * q // 8
            Mstar = float(np.mean([cachesim.simulate(R[l], E, C, "min", bypass=True)[0].mean() for l in range(L)]))
            t_sol = bound(TD, L, k, a_, b_, p_, Mstar)
            res = [ecsim_fast.simulate_layer(R[l], E, C, "dfa", kappa=K) for l in range(L)]
            m = np.stack([r[1] for r in res], 1).astype(float)
            adm = np.stack([r[2] for r in res], 1).astype(float)
            t_dfa = max(TD + float(np.maximum((k - m) * a_, m * b_).sum(1).mean()), float(adm.sum(1).mean()) * p_)
            b = sc[tag]["budgets"].get(str(q), {})
            mb, st = b.get("mailbox cache", {}).get("tok_s"), b.get("llama.cpp static layers", {}).get("tok_s")
            rows[q] = dict(sol_tok_s=1 / t_sol, dfa_ideal_tok_s=1 / t_dfa, mailbox_tok_s=mb, llama_tok_s=st,
                           mailbox_of_sol=mb * t_sol if mb else None, llama_of_sol=st * t_sol if st else None,
                           mailbox_of_dfa_ideal=mb * t_dfa if mb else None,
                           minbypass_miss_per_layer_step=Mstar, dfa_hit=float(1 - m.mean() / k), steps=int(R.shape[1]))
        out[tag] = dict(eta_g=eta, B_C=bc, pack=os.path.relpath(p, a.results), budgets=rows)
        for q, r in rows.items():
            f = lambda v: "–" if v is None else f"{v:.0%}"
            print(f"{tag} {a.arm} {q}/8: SoL {r['sol_tok_s']:.1f} DFA-ideal {r['dfa_ideal_tok_s']:.1f} | mailbox {r['mailbox_tok_s']} "
                  f"({f(r['mailbox_of_sol'])} of SoL, {f(r['mailbox_of_dfa_ideal'])} of DFA-ideal) llama {r['llama_tok_s']} ({f(r['llama_of_sol'])}) "
                  f"| M* {r['minbypass_miss_per_layer_step']:.2f} DFA hit {r['dfa_hit']:.3f} eta_g {eta:.2f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
