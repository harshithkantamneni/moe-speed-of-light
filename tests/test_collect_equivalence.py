"""The streaming collector must reproduce the reference transformers model's
expert selections exactly, layer by layer, and its final logits."""
import sys, tempfile, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mosl.collect import collect

REPOS = sys.argv[1:] or [
    "yujiepan/qwen3-moe-tiny-random",
    "hf-tiny-v2/tiny-random-OlmoeForCausalLM",
    "yujiepan/deepseek-v2-tiny-random",
    "yujiepan/gpt-oss-tiny-random",        # MXFP4-packed experts, sliding-window layers
    "yujiepan/gpt-oss-tiny-random-bf16",
    "yujiepan/mixtral-tiny-random",           # pre-v5 checkpoint names (block_sparse_moe.experts.N.w1/w2/w3)
    "yujiepan/phi-moe-tiny-random",           # sparsemixer routing, router named mlp.router
    "hf-tiny-v2/tiny-random-Qwen2MoeForCausalLM",   # shared expert with a sigmoid gate
]


def reference(repo, seqs):
    from transformers import AutoConfig
    impl = "eager" if AutoConfig.from_pretrained(repo).model_type == "gpt_oss" else "sdpa"
    m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32, attn_implementation=impl)
    m.eval()
    recs = {}
    hooks = []
    for i, layer in enumerate(m.model.layers):
        g = getattr(layer.mlp, "gate", None) or getattr(layer.mlp, "router", None)
        if g is not None and hasattr(g, "top_k"):
            recs[i] = []
            hooks.append(g.register_forward_hook(lambda mod, inp, out, i=i: recs[i].append(out[2].clone())))
    with torch.no_grad():
        for s in seqs:
            m(torch.tensor([s]))
    return {i: np.sort(torch.cat(v).numpy(), axis=1) for i, v in recs.items()}


def decode_reference(repo, seqs):
    """Routing observed during genuine autoregressive decode with a KV cache
    (one token per forward call), to show prefill routing == decode routing."""
    from transformers import AutoConfig
    impl = "eager" if AutoConfig.from_pretrained(repo).model_type == "gpt_oss" else "sdpa"
    m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32, attn_implementation=impl).eval()
    recs, cur = {}, []
    for i, layer in enumerate(m.model.layers):
        g = getattr(layer.mlp, "gate", None) or getattr(layer.mlp, "router", None)
        if g is not None and hasattr(g, "top_k"):
            recs[i] = []
            g.register_forward_hook(lambda mod, inp, out, i=i: recs[i].append(out[2].clone()))
    with torch.no_grad():
        for s in seqs:
            past = None
            for t in range(len(s)):
                o = m(torch.tensor([[s[t]]]), past_key_values=past, use_cache=True)
                past = o.past_key_values
    return {i: np.sort(torch.cat(v).numpy(), axis=1) for i, v in recs.items()}


def check(repo):
    rng = np.random.default_rng(0)
    from transformers import AutoConfig
    V = AutoConfig.from_pretrained(repo).vocab_size
    seqs = [rng.integers(0, V, n).tolist() for n in (17, 300, 5)]  # 300 > gpt-oss window (128)
    ref = reference(repo, seqs)
    with tempfile.TemporaryDirectory() as d:
        meta = collect(repo, seqs, d, keep=True, log=lambda s: None, device=os.environ.get("MOSL_DEVICE", "cpu"))
        assert set(int(k) for k in meta["layers"]) == set(ref), (meta["layers"].keys(), ref.keys())
        for i, r in ref.items():
            got = np.load(os.path.join(d, f"layer{i:03d}.npy"))
            agree = (got == r).all(1).mean()
            assert agree == 1.0, f"{repo} layer {i}: token agreement {agree:.4f}"
        dec = decode_reference(repo, seqs)
        for i, r in dec.items():
            got = np.load(os.path.join(d, f"layer{i:03d}.npy"))
            agree = (got == r).all(1).mean()
            assert agree == 1.0, f"{repo} layer {i}: decode-vs-streamed agreement {agree:.4f}"
    print(f"OK {repo}: {len(ref)} MoE layers, {sum(map(len, seqs))} tokens, exact match vs reference prefill AND KV-cache decode (collector on {os.environ.get('MOSL_DEVICE', 'cpu')})")


if __name__ == "__main__":
    for r in REPOS:
        check(r)
