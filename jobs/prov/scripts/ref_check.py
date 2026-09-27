"""Reference check of real-model traces: the stock transformers model (from_pretrained, fp32, whole model) on the
first N conversations of a trace pack, against the layer-streamed collector's selections and NLL, and the vLLM
score of the same tokens.

    python scripts/ref_check.py --model-dir DIR --pack X_D.npz --corpus corp_X_D.jsonl --score score_X_D.jsonl --n 2
"""
import argparse
import json
import sys
import time

import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mosl.traces import load_pack  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--score", default=None)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    pk = load_pack(a.pack)
    rows = {json.loads(l)["corpus_idx"]: json.loads(l) for l in open(a.corpus)}
    vl = {json.loads(l)["corpus_idx"]: json.loads(l)["nll"] for l in open(a.score)} if a.score else {}
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.float32).to(a.device)
    model.eval()
    print(f"loaded in {time.time() - t0:.0f}s: {type(model).__name__}", flush=True)
    rec = {}
    gates = []
    for li, layer in enumerate(model.model.layers):
        g = getattr(layer.mlp, "gate", None) or getattr(layer.mlp, "router", None)
        if g is not None and hasattr(g, "top_k"):
            g.register_forward_hook(lambda mod, inp, out, li=li: rec.__setitem__(li, out[2].detach().cpu().numpy()))
            gates.append(li)
    layers_pack = list(pk["layers"])
    assert layers_pack == gates, (layers_pack, gates)
    diff = tot = 0
    out = []
    for j in range(a.n):
        ci, s0, L, P = (int(pk[k][j]) for k in ("corpus_idx", "starts", "seq_lens", "prompt_lens"))
        ids = rows[ci]["ids"]
        assert len(ids) == L
        rec.clear()
        t1 = time.time()
        with torch.no_grad():
            logits = model(torch.tensor([ids], device=a.device)).logits[0, :-1].float()
        nll = torch.nn.functional.cross_entropy(logits, torch.tensor(ids[1:], device=a.device), reduction="none").cpu().numpy()
        d = t = 0
        for pos, li in enumerate(gates):
            ref = np.sort(rec[li].reshape(L, -1), axis=1)
            col = np.sort(pk["routes"][pos, s0:s0 + L], axis=1)
            d += int((ref != col).any(axis=1).sum()); t += L
        diff += d; tot += t
        coln = pk["nll"][s0 - j:s0 - j + L - 1]
        r = dict(corpus_idx=ci, tokens=L, tokens_layers_differing=d / t,
                 resp_nll_ref=float(nll[P - 1:].mean()), resp_nll_collector=float(coln[P - 1:].mean()),
                 mean_abs_nll_ref_vs_collector=float(np.abs(nll - coln).mean()), seconds=time.time() - t1)
        if ci in vl:
            v = np.array([x if x is not None else np.nan for x in vl[ci][P - 1:L - 1]], float)
            r["resp_nll_vllm"] = float(np.nanmean(v))
        out.append(r)
        print(json.dumps(r), flush=True)
    print(json.dumps(dict(model_dir=a.model_dir, pack=a.pack, tokens_layers_differing=diff / tot, n=a.n,
                          resp_nll_ref=float(np.mean([r["resp_nll_ref"] for r in out])),
                          resp_nll_collector=float(np.mean([r["resp_nll_collector"] for r in out])),
                          resp_nll_vllm=float(np.mean([r.get("resp_nll_vllm", np.nan) for r in out])))), flush=True)


if __name__ == "__main__":
    sys.exit(main())
