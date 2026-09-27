# s_corpus GEN-DIR CORPUS-KEY PROMPTS-KEY SEQS OUT: the test rows' responses from an on-policy corpus, in --seqs order
s_corpus() {
  python3 - "$1/corp_$2.jsonl" "$J/../prov/data/prompts/$3.jsonl" "$4" > "$5" <<'PY'
import json, sys
corp = {r["corpus_idx"]: r for r in map(json.loads, open(sys.argv[1]))}
rows = {r["tok_row"]: r["corpus_idx"] for r in map(json.loads, open(sys.argv[2])) if r["tok_row"] >= 0}
for s in map(int, sys.argv[3].split(",")):
    c = corp[rows[s]]
    print(json.dumps({"domain": c["domain"], "ids": c["ids"], "prompt_len": c["prompt_len"], "src_row": s}))
PY
}
