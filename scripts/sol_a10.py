"""Speed-of-light bound (paper, Sec. bound) on the A10 test run, with first-party constants.

GPU efficiency eta_g is set so that the model reproduces the measured all-GPU step exactly
(T0 = (D + L k s) / (eta_g B_g)); CPU expert time b = s / B_c with B_c the measured 30-thread read bandwidth
(no per-expert latency: optimistic); a copy costs p = s / B_p with the measured pinned H2D bandwidth. MIN-bypass
misses come from the HF routing traces of the test windows. Also reported: the same accounting for the misses
DFA actually incurs (simulated), i.e. the step time an overhead-free implementation of this policy would reach.

    python scripts/sol_a10.py --scored prereg/a10_mb/scored.json --out prereg/a10_mb/sol.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl.ecsim import load_steps, simulate  # noqa: E402
from scripts.oracle_hits import belady_layer  # noqa: E402

B_G, B_C, B_P = 514.4e9, 151.4e9, 25.2e9   # jobs 001 (device read 1 GiB, STREAM-like read at 30 threads (018), H2D pinned)
MODELS = {  # tag -> (trace dir, corpus, test seqs, L, E, k, expert bytes, dense bytes, all-GPU reference tag, kappa)
    "gpt-oss-20b-MXFP4": ("data/traces/gpt-oss-20b", "data/tok_gpt-oss-20b.jsonl", [7, 10, 12, 15, 16, 18, 19, 20, 25, 26, 27, 31],
                          24, 32, 4, 13253760, 687012096, "gpt-oss-20b-MXFP4", 1.0),
    "Qwen3-30B-A3B-Instruct-2507-Q4_K_M": ("data/traces/qwen3-30b-a3b", "data/tok_qwen3_30b.jsonl", [7, 10, 12, 15, 16, 18, 19, 20, 25, 26, 27, 32],
                                           48, 128, 8, 2800000, 567271424, "Qwen3-30B-A3B-Instruct-2507-Q4_K_M", 2.0),
    "Qwen3-30B-A3B-Instruct-2507-Q8_0": ("data/traces/qwen3-30b-a3b", "data/tok_qwen3_30b.jsonl", [7, 10, 12, 15, 16, 18, 19, 20, 25, 26, 27, 32],
                                         48, 128, 8, 5013504, 1013768192, "Qwen3-30B-A3B-Instruct-2507-Q4_K_M", 2.0),
}


def bound(TD, L, k, a, b, p, Mstar, concurrent=True, grid=801):
    best = np.inf
    for x in np.linspace(0.0, max(Mstar, 1e-9), grid):
        lo = max(0.0, Mstar - x)
        mc = np.linspace(lo, k, grid)
        lay = np.maximum((k - mc) * a, mc * b) if concurrent else (k - mc) * a + mc * b
        best = min(best, max(TD + L * float(lay.min()), L * x * p))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sc = json.load(open(args.scored))["results"]
    out = {}
    for tag, (td, corp, seqs, L, E, k, s, D, ref, K) in MODELS.items():
        if tag not in sc or ref not in sc or "all_gpu" not in sc[ref]:
            continue
        # eta_g from the reference model's all-GPU run (same GPU kernels family for Q8_0 as for Q4_K_M: an assumption)
        Lr, kr, sr, Dr = MODELS[ref][3], MODELS[ref][5], MODELS[ref][6], MODELS[ref][7]
        T0 = 1.0 / sc[ref]["all_gpu"]["tok_s"]
        eta_g = (Dr + Lr * kr * sr) / (T0 * B_G)
        a, b, p = s / (eta_g * B_G), s / B_C, s / B_P
        TD = D / (eta_g * B_G)
        R, E2, _ = load_steps(td, corp, seqs, 128, 192)
        N = len(R[0])
        rows = {}
        for q, bq in sc[tag]["budgets"].items():
            C = E * int(q) // 8
            Mstar = float(np.mean([(1 - belady_layer(r, E, C)) * k for r in R]))     # MIN-bypass misses per layer step
            sol = bound(TD, L, k, a, b, p, Mstar)
            h, m, adm = simulate(R, E2, C, "dfa", kappa=K)
            lay = np.maximum((k - m) * a, m * b)                                     # DFA's own misses, no overheads
            t_dfa = max(TD + float(lay.sum(1).mean()), float(adm.sum(1).mean()) * p)
            meas = 1.0 / bq["mailbox_cache"]["tok_s"]
            st = 1.0 / bq["llama_static"]["tok_s"]
            rows[q] = dict(sol_tok_s=1 / sol, dfa_ideal_tok_s=1 / t_dfa, mailbox_tok_s=1 / meas, llama_tok_s=1 / st,
                           mailbox_of_sol=sol / meas, llama_of_sol=sol / st, mailbox_of_dfa_ideal=t_dfa / meas,
                           minbypass_miss_per_layer_step=Mstar, dfa_miss_per_layer_step=float(m.mean()))
        out[tag] = dict(eta_g=eta_g, a_us=a * 1e6, b_us=b * 1e6, p_us=p * 1e6, TD_ms=TD * 1e3, budgets=rows)
    json.dump(out, open(args.out, "w"), indent=1)
    for tag, o in out.items():
        print(tag, f"eta_g {o['eta_g']:.2f} a {o['a_us']:.1f}us b {o['b_us']:.1f}us p {o['p_us']:.0f}us TD {o['TD_ms']:.2f}ms")
        for q, r in o["budgets"].items():
            print(f"  {int(q)/8:.3f}: SoL {r['sol_tok_s']:.1f}  DFA-ideal {r['dfa_ideal_tok_s']:.1f}  mailbox {r['mailbox_tok_s']:.1f} "
                  f"({r['mailbox_of_sol']:.0%} of SoL, {r['mailbox_of_dfa_ideal']:.0%} of DFA-ideal)  llama {r['llama_tok_s']:.1f} ({r['llama_of_sol']:.0%})")


if __name__ == "__main__":
    main()
