"""Prompt token ids for every corpus conversation, per model, for on-policy generation.

The prompt is the model's own chat template applied to the user turn with the generation prompt appended,
exactly as `mosl.tokenize_corpus` computed `prompt_len`. gpt-oss's template stamps the current date into its
system message; we pin it to the date in the existing token file so the prompts of the traced dataset
sequences are reproduced token for token (asserted below).

    python -m mosl.prompts --out-dir data/prompts
"""
import argparse
import json
import os
import re
from collections import Counter

from transformers import AutoTokenizer

MODELS = {  # key -> (repo, existing token file)
    "olmoe": ("allenai/OLMoE-1B-7B-0125-Instruct", "data/tok_olmoe.jsonl"),
    "qwen3-30b-a3b": ("Qwen/Qwen3-30B-A3B-Instruct-2507", "data/tok_qwen3_30b.jsonl"),
    "gpt-oss-20b": ("openai/gpt-oss-20b", "data/tok_gpt-oss-20b.jsonl"),
    "gpt-oss-120b": ("openai/gpt-oss-120b", "data/tok_gpt-oss-120b.jsonl"),
    # models of the audit (no earlier token file)
    "mixtral-8x7b": ("mistralai/Mixtral-8x7B-Instruct-v0.1", None),
    "deepseek-v2-lite": ("deepseek-ai/DeepSeek-V2-Lite-Chat", None),
    "qwen1.5-moe": ("Qwen/Qwen1.5-MoE-A2.7B-Chat", None),
    "qwen2-57b": ("Qwen/Qwen2-57B-A14B-Instruct", None),
    "phi3.5-moe": ("microsoft/Phi-3.5-MoE-instruct", None),
}
DATE_RE = re.compile(r"Current date: \d{4}-\d{2}-\d{2}")


def render(tok, messages, gen, date):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=gen)
    if date:
        text = DATE_RE.sub(f"Current date: {date}", text)
    return tok(text, add_special_tokens=False)["input_ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--out-dir", default="data/prompts")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--tokens-per-domain", type=int, default=6000)
    ap.add_argument("--data-len", type=int, default=2048)
    ap.add_argument("--models", default=None, help="comma-separated subset of MODELS")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    corpus = [json.loads(l) for l in open(a.corpus)]
    for key, (repo, tokf) in MODELS.items():
        if a.models and key not in a.models.split(","):
            continue
        tok = AutoTokenizer.from_pretrained(repo)
        old = [json.loads(l) for l in open(tokf)] if tokf else []
        date = None
        m = DATE_RE.search(tok.decode(old[0]["ids"][:200])) if old else None
        if m:
            date = m.group(0).split(": ")[1]
        budget, row, out = Counter(), 0, []
        for ci, it in enumerate(corpus):
            d = it["domain"]
            pids = render(tok, it["messages"][:1], True, date)
            full = render(tok, it["messages"], False, date)
            in_old = bool(old) and budget[d] < a.tokens_per_domain  # the selection rule of mosl.tokenize_corpus
            tok_row = -1
            if in_old:
                ids = full[: a.max_len]
                assert ids == old[row]["ids"], f"{key}: corpus item {ci} does not reproduce token row {row}"
                budget[d] += len(ids)
                tok_row, row = row, row + 1
            # off-policy arm D: the full dataset conversation (<= --data-len tokens); its response starts where the
            # prompt and the conversation diverge (BPE may merge across the boundary, as in mosl.tokenize_corpus)
            plen = next((j for j, (x, y) in enumerate(zip(full, pids)) if x != y), min(len(full), len(pids)))
            assert len(pids) - plen <= 2, f"{key}: prompt diverges from conversation {ci}"
            empty = render(tok, [{"role": "user", "content": ""}], True, date)
            ustart = next((j for j, (x, y) in enumerate(zip(empty, pids)) if x != y), min(len(empty), len(pids)))
            out.append({"corpus_idx": ci, "domain": d, "tok_row": tok_row, "prompt_ids": pids, "user_start": ustart,
                        "data_ids": full[: a.data_len], "data_prompt_len": min(plen, a.data_len)})
        assert row == len(old), f"{key}: matched {row} of {len(old)} token rows"
        with open(os.path.join(a.out_dir, f"{key}.jsonl"), "w") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        pl = sorted(len(r["prompt_ids"]) for r in out)
        print(f"{key}: {len(out)} prompts, {row} match the traced rows; prompt tokens median {pl[len(pl)//2]}, max {pl[-1]}, date {date}")


if __name__ == "__main__":
    main()
