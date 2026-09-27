"""Per-layer vs shared all-layer (global) expert budget: MIN-bypass misses per layer-step on each model's own
sampled text (arm S, response tokens), at 12.5 % and 25 % of the experts. A shared pool of L*C experts can only
lower M* (Proposition 1 holds for both); this measures by how much.

    python scripts/global_gain.py --results /home/claude/gpu-branch/results --out prereg/global
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl import cachesim  # noqa: E402
from mosl.traces import load_pack  # noqa: E402

PACKS = {"olmoe": "olmoe_S.npz", "gpt-oss-20b": "gpt-oss-20b_S.npz", "qwen3-30b-a3b": "qwen3-30b-a3b_fp8_S.npz",
         "gpt-oss-120b": "gpt-oss-120b_S.npz", "mixtral-8x7b": "mixtral-8x7b_S.npz",
         "deepseek-v2-lite": "deepseek-v2-lite_S.npz", "qwen1.5-moe": "qwen1.5-moe_S.npz",
         "qwen2-57b": "qwen2-57b_S.npz", "phi3.5-moe": "phi3.5-moe_S.npz"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {}
    for key, name in PACKS.items():
        hits = sorted(glob.glob(os.path.join(a.results, "*", name)))
        if not hits:
            continue
        pk = load_pack(hits[-1])
        idx = np.concatenate([np.arange(s + P, s + L) for s, L, P in zip(pk["starts"], pk["seq_lens"], pk["prompt_lens"]) if L - P > 1])
        R = pk["routes"][:, idx]
        E, k, L = int(pk["E"]), R.shape[2], R.shape[0]
        r = out[key] = dict(E=E, k=k, L=L, steps=int(R.shape[1]), pack=os.path.relpath(hits[-1], a.results))
        for q in (1, 2):
            C = max(1, E * q // 8)
            per = float(np.mean([cachesim.simulate(R[l], E, C, "min", bypass=True)[0].mean() for l in range(L)]))
            glo = float(cachesim.simulate_global([R[l] for l in range(L)], E, C, "min", bypass=True)[0].mean())
            r[f"q{q}"] = dict(C=C, mstar_per_layer=per, mstar_global=glo, reduction=1 - glo / per if per else 0.0)
        print(key, json.dumps(r))
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "global.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
