"""The streaming collector must reproduce the reference transformers model's
expert selections exactly, layer by layer, and its final logits."""
import sys, tempfile, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mosl.collect import collect

REPOS = [
    "yujiepan/qwen3-moe-tiny-random",
    "hf-tiny-v2/tiny-random-OlmoeForCausalLM",
    "yujiepan/deepseek-v2-tiny-random",
]


def reference(repo, seqs):
    m = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32, attn_implementation="sdpa")
    m.eval()
    recs = {}
    hooks = []
    for i, layer in enumerate(m.model.layers):
        g = getattr(layer.mlp, "gate", None)
        if g is not None and hasattr(g, "top_k"):
            recs[i] = []
            hooks.append(g.register_forward_hook(lambda mod, inp, out, i=i: recs[i].append(out[2].clone())))
    with torch.no_grad():
        for s in seqs:
            m(torch.tensor([s]))
    return {i: np.sort(torch.cat(v).numpy(), axis=1) for i, v in recs.items()}


def check(repo):
    rng = np.random.default_rng(0)
    from transformers import AutoConfig
    V = AutoConfig.from_pretrained(repo).vocab_size
    seqs = [rng.integers(0, V, n).tolist() for n in (17, 64, 5)]
    ref = reference(repo, seqs)
    with tempfile.TemporaryDirectory() as d:
        meta = collect(repo, seqs, d, keep=True, log=lambda s: None)
        assert set(int(k) for k in meta["layers"]) == set(ref), (meta["layers"].keys(), ref.keys())
        for i, r in ref.items():
            got = np.load(os.path.join(d, f"layer{i:03d}.npy"))
            agree = (got == r).all(1).mean()
            assert agree == 1.0, f"{repo} layer {i}: token agreement {agree:.4f}"
    print(f"OK {repo}: {len(ref)} MoE layers, {sum(map(len, seqs))} tokens, exact match")


if __name__ == "__main__":
    for r in REPOS:
        check(r)
