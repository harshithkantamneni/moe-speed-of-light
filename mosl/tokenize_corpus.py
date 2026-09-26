"""Tokenize the conversation corpus with a model's own chat template."""
import argparse
import json
from collections import Counter

from transformers import AutoTokenizer

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--tokens-per-domain", type=int, default=6000)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.repo)
    budget = Counter()
    n = 0
    with open(a.out, "w") as f:
        for line in open(a.corpus):
            it = json.loads(line)
            d = it["domain"]
            if budget[d] >= a.tokens_per_domain:
                continue
            if tok.chat_template:
                text = tok.apply_chat_template(it["messages"], tokenize=False)
                ids = tok(text, add_special_tokens=False)["input_ids"]
                ptxt = tok.apply_chat_template(it["messages"][:1], tokenize=False, add_generation_prompt=True)
                pids = tok(ptxt, add_special_tokens=False)["input_ids"]
                # BPE may merge across the boundary; use the longest common prefix
                plen = next((j for j, (x, y) in enumerate(zip(ids, pids)) if x != y), min(len(ids), len(pids)))
                assert len(pids) - plen <= 2, "prompt diverges from conversation"
            else:
                ids = tok("\n\n".join(m["content"] for m in it["messages"]))["input_ids"]
                plen = len(tok(it["messages"][0]["content"])["input_ids"])
            ids = list(ids)[: a.max_len]
            plen = min(plen, len(ids))
            budget[d] += len(ids)
            n += 1
            f.write(json.dumps({"domain": d, "ids": ids, "prompt_len": plen}) + "\n")
    print(a.repo, n, "seqs", dict(budget))
