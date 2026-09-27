"""Pre-registered predictions for the audit's anchor runs (llama.cpp --n-cpu-moe on an A100 40 GB).

Run by job 050 on the anchor instance after its platform microbenchmarks and BEFORE job 051 measures anything; the
runner commits job 050's outputs first, so git history orders predictions before measurements. Same predictor and
input rules as phase 1 (scripts/preregister.py): M4 efficiencies fitted on third-party data only, exact bytes from
the GGUF headers, B_g datasheet, B_c best STREAM Triad, B_p negotiated link. Also reported: the A10-refit variant
(sensitivity, scripts/audit.py --with-a10) and the physical floor.

    python scripts/anchor_predict.py <platform dir> <out dir>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.preregister as pr  # noqa: E402
from mosl.gguf_bytes import GGUFS, account  # noqa: E402

pr.HF_REPO.update({"mixtral-8x7b-q4_k_m": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                   "mixtral-8x7b-q8_0": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                   "phi3.5-moe-q4_k_m": "microsoft/Phi-3.5-MoE-instruct",
                   "qwen2-57b-q4_k_m": "Qwen/Qwen2-57B-A14B-Instruct",
                   "dsv2lite-q8_0": "deepseek-ai/DeepSeek-V2-Lite-Chat"})
# --n-cpu-moe values measured by job 051 (block indices 0..n-1; must fit 40 GB with the dense weights)
SWEEP = {"mixtral-8x7b-q4_k_m": [0, 8, 16, 24, 32], "mixtral-8x7b-q8_0": [10, 16, 24, 32],
         "phi3.5-moe-q4_k_m": [0, 8, 16, 24, 32], "qwen2-57b-q4_k_m": [4, 10, 16, 22, 28],
         "dsv2lite-q8_0": [0, 7, 14, 20, 27], "qwen3-30b-a3b-q4_k_m": [0, 12, 24, 36, 48]}
FIRST_MOE = {"dsv2lite-q8_0": 1}   # DeepSeek-V2-Lite: block 0 is dense, so --n-cpu-moe n moves n-1 MoE layers


def main(pdir, out):
    plat = pr.platform(pdir)
    P = pr.params()
    from scripts.a10_rows import A10_ROWS
    from scripts.validate import VARIANTS, fit
    NAMES = VARIANTS["M4 M1 + per-CPU-expert latency"]
    from mosl.validation_set import ROWS
    q = fit(ROWS + A10_ROWS, NAMES)
    P["M4+A10"] = {k: getattr(q, k) for k in NAMES}
    rows = []
    for name, ns in SWEEP.items():
        g = account(*GGUFS[name])
        g.file_key = name
        for n in ns:
            m = max(0, n - FIRST_MOE.get(name, 0))
            rows.append(dict(model=name, n_cpu_moe=n, moe_layers_on_cpu=m,
                             pred_tok_s=1 / pr.step_time(g, m, plat, P["M4"]),
                             pred_tok_s_a10refit=1 / pr.step_time(g, m, plat, P["M4+A10"]),
                             roofline_tok_s=1 / pr.roofline_time(g, m, plat)))
    os.makedirs(out, exist_ok=True)
    json.dump(dict(platform=plat, params=P, sweep=rows, note="scripts/anchor_predict.py, before job 051"),
              open(os.path.join(out, "anchor_predictions.json"), "w"), indent=1)
    L = [f"# Anchor predictions: {plat['gpu']}, B_c {plat['bw_cpu']:.1f} GB/s (STREAM Triad), B_p {plat['bw_pcie']} GB/s\n",
         "| model | --n-cpu-moe | M4 tok/s | M4+A10 tok/s | floor tok/s |", "|---|---|---|---|---|"]
    L += [f"| {r['model']} | {r['n_cpu_moe']} | {r['pred_tok_s']:.1f} | {r['pred_tok_s_a10refit']:.1f} | {r['roofline_tok_s']:.1f} |" for r in rows]
    open(os.path.join(out, "anchor_predictions.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
