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
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    corpus = [json.loads(l) for l in open(a.corpus)]
    for key, (repo, tokf) in MODELS.items():
        tok = AutoTokenizer.from_pretrained(repo)
        old = [json.loads(l) for l in open(tokf)]
        date = None
        m = DATE_RE.search(tok.decode(old[0]["ids"][:200]))
        if m:
            date = m.group(0).split(": ")[1]
        budget, row, out = Counter(), 0, []
        for ci, it in enumerate(corpus):
            d = it["domain"]
            pids = render(tok, it["messages"][:1], True, date)
            full = render(tok, it["messages"], False, date)
            in_old = budget[d] < a.tokens_per_domain           # the selection rule of mosl.tokenize_corpus
            tok_row = -1
            if in_old:
                ids = full[: a.max_len]
                assert ids == old[row]["ids"], f"{key}: corpus item {ci} does not reproduce token row {row}"
                budget[d] += len(ids)
                tok_row, row = row, row + 1
            out.append({"corpus_idx": ci, "domain": d, "tok_row": tok_row, "prompt_ids": pids,
                        "response_text": it["messages"][1]["content"]})
        assert row == len(old), f"{key}: matched {row} of {len(old)} token rows"
        with open(os.path.join(a.out_dir, f"{key}.jsonl"), "w") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        pl = sorted(len(r["prompt_ids"]) for r in out)
        print(f"{key}: {len(out)} prompts, {row} match the traced rows; prompt tokens median {pl[len(pl)//2]}, max {pl[-1]}, date {date}")


if __name__ == "__main__":
    main()
