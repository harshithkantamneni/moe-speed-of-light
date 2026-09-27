"""On-policy corpora and third-implementation scores with vLLM (phase 4, sections 4.1 and 4.4).

For one model and all prompts in data/prompts/<key>.jsonl:
  G  greedy generation (temperature 0), <= --max-new tokens, with the logprob of every generated token;
  S  generation with the model's own generation-config defaults (LLM.get_default_sampling_params()),
     seed = corpus index, same length cap, same logprobs;
  D  prompt_logprobs of the dataset conversation (data_ids), i.e. teacher-forced NLL of every token;
and, for gpt-oss only, prompt_logprobs of the traced dataset conversations in two more formats:
  V1 the model's own greedy analysis message inserted before the final channel;
  V2 "Reasoning: low" in the system message.
Logprobs are vLLM's raw logprobs (before temperature, top-k/top-p), so they are the model's NLL.

Writes, per arm, a token file in the tracer's format ({domain, ids, prompt_len, corpus_idx, user_start}) and a
score file with per-token NLL (nll[i] = -log p(ids[i+1] | ids[:i+1]); positions without a score are null).

    python scripts/onpolicy_gen.py --model gpt-oss-20b --out results/022
"""
import argparse
import json
import os
import time

MODELS = {"olmoe": "allenai/OLMoE-1B-7B-0125-Instruct", "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
          "gpt-oss-20b": "openai/gpt-oss-20b", "gpt-oss-120b": "openai/gpt-oss-120b"}


def lp_of(entry, tok):
    if entry is None:
        return None
    e = entry.get(tok)
    return None if e is None else float(e.logprob)


def dump(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--prompts", default="data/prompts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--batched-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0, help="first N prompts only (smoke test)")
    ap.add_argument("--model-dir", default=None, help="local checkpoint directory (default: the hub repo)")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    key, repo = a.model, MODELS[a.model]
    rows = [json.loads(l) for l in open(os.path.join(a.prompts, f"{key}.jsonl"))]
    if a.limit:
        rows = rows[: a.limit]
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    llm = LLM(model=a.model_dir or repo, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, seed=0,
              enable_prefix_caching=False, max_num_batched_tokens=a.batched_tokens, enforce_eager=True,
              max_logprobs=1)
    import vllm
    log = {"model": repo, "vllm": vllm.__version__, "load_s": time.time() - t0}
    tok = llm.get_tokenizer()

    def generate(params, tag):
        t = time.time()
        outs = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_ids"]) for r in rows], params, use_tqdm=False)
        corp, score = [], []
        for r, o in zip(rows, outs):
            c = o.outputs[0]
            ids = list(r["prompt_ids"]) + list(c.token_ids)
            lps = [lp_of(e, t_) for e, t_ in zip(c.logprobs or [], c.token_ids)]
            nll = [None] * (len(r["prompt_ids"]) - 1) + [None if x is None else round(-x, 5) for x in lps]
            corp.append({"domain": r["domain"], "ids": ids, "prompt_len": len(r["prompt_ids"]),
                         "corpus_idx": r["corpus_idx"], "user_start": r["user_start"]})
            score.append({"corpus_idx": r["corpus_idx"], "finish": c.finish_reason, "n_new": len(c.token_ids),
                          "nll": nll})
        dump(os.path.join(a.out, f"corp_{key}_{tag}.jsonl"), corp)
        dump(os.path.join(a.out, f"score_{key}_{tag}.jsonl"), score)
        n = sum(s["n_new"] for s in score)
        log[tag] = {"s": time.time() - t, "new_tokens": n,
                    "finish": {f: sum(s["finish"] == f for s in score) for f in {s["finish"] for s in score}}}
        print(tag, json.dumps(log[tag]), flush=True)
        return corp

    def teacher_force(seqs, tag, extra=None):
        t = time.time()
        sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
        outs = llm.generate([TokensPrompt(prompt_token_ids=s) for s in seqs], sp, use_tqdm=False)
        res = []
        for s, o in zip(seqs, outs):
            pl = o.prompt_logprobs or []
            v = [lp_of(pl[i + 1], s[i + 1]) if i + 1 < len(pl) else None for i in range(len(s) - 1)]
            res.append([None if x is None else round(-x, 5) for x in v])
        log[tag] = {"s": time.time() - t, "n_seqs": len(seqs)}
        print(tag, json.dumps(log[tag]), flush=True)
        return res

    # G, S
    g = generate(SamplingParams(temperature=0.0, max_tokens=a.max_new, logprobs=0), "G")
    base = llm.get_default_sampling_params()
    log["S_params"] = repr(base)
    print("S params:", repr(base), flush=True)
    keep = {f: getattr(base, f) for f in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty",
                                           "presence_penalty", "frequency_penalty") if hasattr(base, f)}
    log["S_fields"] = keep
    s_params = [SamplingParams(**keep, seed=r["corpus_idx"], max_tokens=a.max_new, logprobs=0) for r in rows]
    generate(s_params, "S")

    # D: the dataset conversations
    d_nll = teacher_force([r["data_ids"] for r in rows], "D")
    dump(os.path.join(a.out, f"corp_{key}_D.jsonl"),
         [{"domain": r["domain"], "ids": r["data_ids"], "prompt_len": r["data_prompt_len"],
           "corpus_idx": r["corpus_idx"], "user_start": r["user_start"]} for r in rows])
    dump(os.path.join(a.out, f"score_{key}_D.jsonl"),
         [{"corpus_idx": r["corpus_idx"], "nll": x} for r, x in zip(rows, d_nll)])

    json.dump(log, open(os.path.join(a.out, f"log_{key}.json"), "w"), indent=1)
    if key.startswith("gpt-oss"):
        try:
            harmony(a, key, rows, g, tok, teacher_force)
        except Exception:
            import traceback
            traceback.print_exc()
    json.dump(log, open(os.path.join(a.out, f"log_{key}.json"), "w"), indent=1)


def harmony(a, key, rows, g, tok, teacher_force):
    """gpt-oss format variants on the traced conversations (tok_row >= 0): V1 own analysis, V2 reasoning low."""
    if True:
        ids_of = lambda s: tok.encode(s, add_special_tokens=False)
        END, START, CH, MSG = (ids_of(x)[0] for x in ("<|end|>", "<|start|>", "<|channel|>", "<|message|>"))
        FINAL_HDR = ids_of("<|channel|>final<|message|>")
        ANALYSIS_HDR = ids_of("<|channel|>analysis<|message|>")
        v1, v2, meta = [], [], []
        for r, gr in zip(rows, g):
            if r["tok_row"] < 0:
                continue
            d, P = r["data_ids"][:1024], r["data_prompt_len"]            # the traced (1024-token) conversation
            if P + len(FINAL_HDR) >= len(d):
                continue                                                    # no response inside the traced part
            assert d[P:P + len(FINAL_HDR)] == FINAL_HDR, "traced dataset text is not in the final channel"
            content = d[P + len(FINAL_HDR):]                               # response tokens (+ <|return|>)
            gen = gr["ids"][gr["prompt_len"]:]
            ana = None
            if gen[:len(ANALYSIS_HDR)] == ANALYSIS_HDR and END in gen:
                ana = gen[: gen.index(END) + 1]                             # own analysis message incl. <|end|>
            x1 = list(r["prompt_ids"]) + (ana + [START] + ids_of("assistant") if ana else []) + FINAL_HDR + content
            med, low = ids_of(" medium"), ids_of(" low")
            assert len(med) == 1 and len(low) == 1
            i = next(i for i in range(4, 160) if d[i] == med[0] and tok.decode(d[i - 4:i]).endswith("Reasoning:"))
            x2 = d[:i] + low + d[i + 1:]                                    # "Reasoning: medium" -> "Reasoning: low"
            v1.append(x1)
            v2.append(x2)
            meta.append({"corpus_idx": r["corpus_idx"], "tok_row": r["tok_row"], "has_analysis": ana is not None,
                         "resp_len": len(content)})
        n1 = teacher_force(v1, "V1")
        n2 = teacher_force(v2, "V2")
        dump(os.path.join(a.out, f"score_{key}_harmony.jsonl"),
             [dict(m, v1_ids=x1, v1_nll=a1, v2_ids=x2, v2_nll=a2) for m, x1, a1, x2, a2 in zip(meta, v1, n1, v2, n2)])


if __name__ == "__main__":
    main()
