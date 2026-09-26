"""Routing sensitivity to numerics: compare fp32 vs bf16-compute OLMoE traces."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mosl.traces import Trace
from mosl.cachesim import simulate

a = Trace("data/traces/olmoe-1b-7b", "data/tok_olmoe.jsonl")
b = Trace("data/traces/olmoe-1b-7b-bf16", "data/tok_olmoe.jsonl")
diff = np.mean([np.mean(~(Ra == Rb).all(1)) for Ra, Rb in zip(a.R, b.R)])
cap = int(0.25 * a.E)
hit = lambda tr: 1 - sum(simulate(R, tr.E, cap, "lru")[0].sum() for R in tr.R) / (tr.T * tr.L * tr.k)
ha, hb = hit(a), hit(b)
out = dict(frac_token_layer_sets_differ=float(diff), lru25_fp32=float(ha), lru25_bf16=float(hb),
           ppl_fp32=a.meta.get("teacher_forced_ppl"), ppl_bf16=b.meta.get("teacher_forced_ppl"))
json.dump(out, open("results/bf16_check.json", "w"), indent=1)
print(out)
