"""How far the audit's predictor is from measured llama.cpp on our own hardware, on the audit's basis.

The audit predicts each row's equal-memory llama.cpp baseline from datasheet peaks (GPU and host DRAM) with the M4
efficiencies; the registered anchor predictor (scripts/anchor_predict.py) used STREAM Triad for the host instead.
This script recomputes, for every offloaded first-party configuration (A10 job 025 static runs, A100 and GH200
anchors), measured / predicted on the audit's basis, and writes prereg/anchors/basis.json.

    python scripts/anchor_basis.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import scripts.anchor_predict  # noqa: E402,F401  (registers the anchor models' repos)
import scripts.preregister as pr  # noqa: E402
from mosl.gguf_bytes import GGUFS, account  # noqa: E402
from mosl.validation_set import ROWS  # noqa: E402
from scripts.a10_rows import A10_ROWS  # noqa: E402
from scripts.validate import VARIANTS, fit, predict  # noqa: E402

R = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")
PLATFORMS = {"a100": (f"{R}/050_anchor_platform@anchor", 204.8, "8 x DDR4-3200 (EPYC 7J13)"),
             "gh200": (f"{R}/052_anchor_platform_gh200@trace", 512.0, "LPDDR5X (Grace)")}


def main():
    out = {}
    p4 = fit(ROWS, VARIANTS["M4 M1 + per-CPU-expert latency"])
    a10 = [r["tok_s"] / predict(r, p4)[0] for r in A10_ROWS]
    out["a10"] = dict(n=len(a10), median=float(np.median(a10)), min=float(min(a10)), max=float(max(a10)),
                      basis="datasheet peaks: A10 600 GB/s, 8 x DDR4-3200 204.8 GB/s")
    for name, (pdir, peak, dram) in PLATFORMS.items():
        P = json.load(open(os.path.join(pdir, "anchor_predictions.json")))
        plat = dict(pr.platform(pdir), bw_cpu=peak)
        a = json.load(open(f"prereg/anchors/{name}.json"))
        rat, by = [], {}
        for c in a["configs"]:
            if c["n_cpu_moe"] == 0:
                continue
            g = account(*GGUFS[c["model"]]); g.file_key = c["model"]
            x = c["measured"] * pr.step_time(g, c["moe_layers_on_cpu"], plat, P["params"]["M4"])
            rat.append(x)
            by.setdefault(c["model"], []).append(x)
        out[name] = dict(n=len(rat), median=float(np.median(rat)), min=float(min(rat)), max=float(max(rat)),
                         by_model={m: [float(min(v)), float(max(v))] for m, v in by.items()},
                         basis=f"datasheet peaks: GPU {plat['bw_gpu']} GB/s, host {peak} GB/s ({dram})")
    json.dump(out, open("prereg/anchors/basis.json", "w"), indent=1)
    for k, v in out.items():
        print(k, f"n={v['n']} measured/predicted median {v['median']:.2f} range {v['min']:.2f}-{v['max']:.2f} ({v['basis']})")


if __name__ == "__main__":
    main()
