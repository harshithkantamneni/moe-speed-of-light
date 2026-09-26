"""Build a contamination-aware, multi-domain conversation corpus.

Each item is an independent (user, assistant) conversation drawn from a public
dataset. No shared system prompt or template text is injected beyond the
model's own chat template, following the workload-contamination checklist of
arXiv:2608.07911. Conversations are tokenized per model at trace time.
"""
import argparse
import json
import random

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

SOURCES = {
    # domain: (repo, file-glob-substring, extractor)
    "chat": ("HuggingFaceH4/ultrachat_200k", "test_sft",
             lambda r: [(m["role"], m["content"]) for m in r["messages"]][:2]),
    "math": ("AI-MO/NuminaMath-CoT", "test",
             lambda r: [("user", r["problem"]), ("assistant", r["solution"])]),
    "code": ("m-a-p/CodeFeedback-Filtered-Instruction", "",
             lambda r: [("user", r["query"]), ("assistant", r["answer"])]),
    "multilingual": ("CohereLabs/aya_dataset", "",
                     lambda r: [("user", r["inputs"]), ("assistant", r["targets"])]),
}


def load(domain, n, min_chars, seed):
    repo, sub, ex = SOURCES[domain]
    files = sorted(f for f in list_repo_files(repo, repo_type="dataset")
                   if f.endswith((".parquet", ".jsonl")) and sub in f and "demographics" not in f)
    path = hf_hub_download(repo, files[0], repo_type="dataset")
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_json(path, lines=True, nrows=20000)
    if domain == "multilingual":
        df = df[df["language"] != "English"]
    rows = df.sample(frac=1.0, random_state=seed).to_dict("records")
    out = []
    for r in rows:
        conv = ex(r)
        if len(conv) == 2 and conv[0][0] == "user" and sum(len(c) for _, c in conv) >= min_chars:
            out.append({"domain": domain, "messages": [{"role": a, "content": c} for a, c in conv]})
        if len(out) >= n:
            break
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="data/corpus.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    items = []
    for d in SOURCES:
        got = load(d, a.n, 1200, a.seed)
        print(d, len(got))
        items += got
    random.Random(a.seed).shuffle(items)
    with open(a.out, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
