"""Exact parameter accounting for MoE architectures.

Instantiates the transformers model on the `meta` device (no weights, no RAM)
and classifies every parameter, so byte counts used by the performance model
come from the real architecture rather than hand arithmetic.
"""
from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass, asdict


CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "shapes.json")


@dataclass
class Shape:
    repo: str
    n_layers: int
    n_moe_layers: int
    n_experts: int
    top_k: int
    expert_params: int          # params of ONE routed expert (one layer)
    routed_params_total: int    # all routed experts, all layers
    dense_params_per_token: int # attention + shared experts + dense MLPs + routers + norms (all layers)
    lm_head_params: int
    embed_params: int
    total_params: int
    kv_bytes_per_token_ctx: float  # KV-cache bytes read per decode step per context token (bf16)
    sliding_window: int = 0         # layers with a sliding window read min(ctx, window)
    n_sliding_layers: int = 0

    @property
    def active_routed_params(self):
        return self.n_moe_layers * self.top_k * self.expert_params


def _kv_bytes(cfg) -> tuple[float, int, int]:
    L = cfg.num_hidden_layers
    if getattr(cfg, "kv_lora_rank", None):  # MLA: compressed latent + rope key per layer
        per = (cfg.kv_lora_rank + cfg.qk_rope_head_dim) * 2
        return float(per * L), 0, 0
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    kvh = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    per_layer = 2 * kvh * hd * 2
    lt = getattr(cfg, "layer_types", None)
    if lt and getattr(cfg, "sliding_window", None):
        ns = sum(1 for t in lt if "sliding" in t)
        return float(per_layer), int(cfg.sliding_window), ns  # caller combines
    return float(per_layer * L), 0, 0


@functools.lru_cache(maxsize=None)
def shape(repo: str) -> Shape:
    db = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    if repo in db:
        return Shape(**db[repo])
    import torch   # only on a cache miss: machines that only predict need neither torch nor transformers
    from transformers import AutoConfig, AutoModelForCausalLM
    try:
        cfg = AutoConfig.from_pretrained(repo, trust_remote_code=False)
    except ValueError:  # custom-code repos that reuse a native architecture (e.g. Kimi-K2 = DeepSeek-V3)
        from huggingface_hub import hf_hub_download
        from transformers import DeepseekV3Config
        raw = json.load(open(hf_hub_download(repo, "config.json")))
        assert "DeepseekV3" in raw["architectures"][0], raw["architectures"]
        raw.pop("auto_map", None); raw["model_type"] = "deepseek_v3"
        raw.pop("quantization_config", None)
        cfg = DeepseekV3Config(**raw)
    if hasattr(cfg, "text_config") and not hasattr(cfg, "num_hidden_layers"):
        cfg = cfg.text_config
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg)
    routed, dense, head, emb = 0, 0, 0, 0
    moe_layers = set()
    for n, p in m.named_parameters():
        if ".mlp.experts." in n and ".shared" not in n:
            routed += p.numel()
            moe_layers.add(n.split(".layers.")[1].split(".")[0])
        elif n.startswith("lm_head"):
            head += p.numel()
        elif "embed_tokens" in n:
            emb += p.numel()
        else:
            dense += p.numel()
    E = getattr(cfg, "num_local_experts", None) or getattr(cfg, "num_experts", None) or cfg.n_routed_experts
    k = cfg.num_experts_per_tok
    nmoe = len(moe_layers)
    kvb, win, nsl = _kv_bytes(cfg)
    if win:  # per-layer bytes returned; full-attention layers read the whole context
        full = cfg.num_hidden_layers - nsl
        kv_full = kvb * full
    else:
        kv_full = kvb
    if getattr(cfg, "tie_word_embeddings", False) and head == 0:
        head = emb
    s = Shape(repo=repo, n_layers=cfg.num_hidden_layers, n_moe_layers=nmoe, n_experts=int(E), top_k=int(k),
              expert_params=routed // (nmoe * E), routed_params_total=routed, dense_params_per_token=dense,
              lm_head_params=head, embed_params=emb, total_params=routed + dense + head + emb,
              kv_bytes_per_token_ctx=kv_full, sliding_window=win, n_sliding_layers=nsl)
    # per-layer KV bytes for sliding layers are kvb; stash it via window fields
    db[repo] = asdict(s)
    json.dump(db, open(CACHE, "w"), indent=1)
    return s


def kv_bytes(s: Shape, ctx: int, kv_bits: float = 16) -> float:
    """KV bytes read per decode step at context length ctx."""
    b = s.kv_bytes_per_token_ctx * ctx
    if s.sliding_window:
        per = s.kv_bytes_per_token_ctx / max(1, s.n_layers - s.n_sliding_layers)
        b += per * s.n_sliding_layers * min(ctx, s.sliding_window)
    return b * kv_bits / 16


if __name__ == "__main__":
    import sys
    for r in sys.argv[1:]:
        s = shape(r)
        print(f"{r}: total={s.total_params/1e9:.2f}B active_routed={s.active_routed_params/1e9:.2f}B "
              f"dense/token={s.dense_params_per_token/1e9:.2f}B head={s.lm_head_params/1e9:.2f}B "
              f"E={s.n_experts} k={s.top_k} moe_layers={s.n_moe_layers}/{s.n_layers} "
              f"expert={s.expert_params/1e6:.1f}M kv/ctx={s.kv_bytes_per_token_ctx/1e3:.1f}KB")
