#!/bin/bash
# Checkpoint 1 on a Blackwell development host (Vast container, no clock locking): platform numbers, llama.cpp
# 4da6337 built stock and with the rebased expert cache (jobs/ec2/llama.cpp-expert-cache-4da6337.patch: id -1 rows
# zeroed on every CUDA path incl. BF16, stale-payload check in the helpers), op tests with id -1 on every weight type
# and under compute-sanitizer, then gpt-oss-20b on its own text (12 test prompts, whole prompt prefilled, 128 tokens
# decoded): stock --ncmoe and all-GPU against the cache at 25 % of experts. Pass rules (plan, checkpoint 1): op tests
# all OK, 0 memcheck errors, teacher-forced top-1 agreement with stock >= 0.98, |dNLL| <= 1 %.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
platform
build_all
R() { local t=$1; shift; timeout "$t" "$@"; }
R 20m $EC_TESTS -b CUDA0 -o MUL_MAT_ID_EC > $OUT/ops_mmid_ec.txt 2>&1; echo "ops MUL_MAT_ID_EC rc=$?"
R 10m $EC_TESTS -b CUDA0 -o ADD_ID_EC > $OUT/ops_addid_ec.txt 2>&1; echo "ops ADD_ID_EC rc=$?"
R 40m compute-sanitizer --tool memcheck --error-exitcode 9 $EC_TESTS -b CUDA0 -o MUL_MAT_ID_EC > $OUT/memcheck_mmid_ec.txt 2>&1; echo "memcheck rc=$?"
R 40m $EC_TESTS -b CUDA0 -o MUL_MAT_ID > $OUT/ops_mmid_patched.txt 2>&1; echo "ops MUL_MAT_ID (patched build) rc=$?"
R 40m $STOCK_TESTS -b CUDA0 -o MUL_MAT_ID > $OUT/ops_mmid_stock.txt 2>&1; echo "ops MUL_MAT_ID (stock build) rc=$?"
getmodel ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-MXFP4.gguf
CORP=$J/S_gpt-oss-20b.jsonl; SEQ=0,1,2,3,4,5,6,7,8,9,10,11; H=$(( NPROC > 4 ? NPROC - 2 : 2 ))
MB="mailbox=1:tdec=1:helpers=$H"; COMMON="--corpus $CORP --seqs $SEQ -t $NPROC --n-prefill 640 --n-decode 128"
# the cache: warm-up configuration first (discarded), then 25 % of experts (C = 8 of 32 per layer)
R 40m $EC_BIN -m $M/gpt-oss-20b-MXFP4.gguf $COMMON --host-experts \
  --ec "slots=8:policy=dfa:kappa=1:$MB;slots=8:policy=dfa:kappa=1:$MB:stats=$OUT/ec_C8.json" --dump $OUT/ec.bin > $OUT/ec.jsonl 2> $OUT/ec.err
echo "ec rc=$?"
R 40m $EC_BIN -m $M/gpt-oss-20b-MXFP4.gguf $COMMON --ncmoe 18 --dump $OUT/static_n18.bin > $OUT/static_n18.jsonl 2> $OUT/static_n18.err
echo "static n=18 rc=$?"
R 40m $EC_BIN -m $M/gpt-oss-20b-MXFP4.gguf $COMMON --ncmoe 0 --dump $OUT/static_n0.bin > $OUT/static_n0.jsonl 2> $OUT/static_n0.err
echo "all-GPU rc=$?"
python3 - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, sys, os, re, glob
import numpy as np
out = sys.argv[1]
for f in sorted(glob.glob(f"{out}/ops_*.txt") + glob.glob(f"{out}/memcheck_*.txt")):
    t = open(f, errors="replace").read()
    ok = len(re.findall(r"\bOK\b", t)); fail = len(re.findall(r"\bFAIL\b", t))
    err = re.findall(r"ERROR SUMMARY: (\d+) error", t)
    print(f"{os.path.basename(f)}: OK {ok} FAIL {fail}" + (f" memcheck errors {err[-1]}" if err else "") +
          f" | {' '.join(l.strip() for l in t.splitlines() if 'tests passed' in l or 'Backend' in l and 'OK' in l)[:160]}")
def runs(p):
    by = {}
    for l in open(p):
        if l.strip():
            r = json.loads(l); by.setdefault(r.get("config", ""), []).append(r)
    return by
def rate(v): return sum(r["n_decode"] for r in v) / (sum(r["decode_ms"] for r in v) / 1000)
def nll(v): return sum(r["nll_sum"] for r in v) / max(1, sum(r["nll_n"] for r in v))
def top1(p): return np.fromfile(p, dtype=np.int32).reshape(-1, 10)[:, 0] if os.path.exists(p) and os.path.getsize(p) else None
res = {}
for name, path, dump in [("cache_C8", f"{out}/ec.jsonl", f"{out}/ec.bin.1"), ("static_n18", f"{out}/static_n18.jsonl", f"{out}/static_n18.bin"),
                         ("all_gpu", f"{out}/static_n0.jsonl", f"{out}/static_n0.bin")]:
    if not os.path.exists(path): continue
    by = runs(path); key = list(by)[-1]; v = by[key]
    res[name] = dict(tok_s=rate(v), nll=nll(v), n=len(v), steps=sum(r["n_decode"] for r in v), top1=top1(dump))
ref = res.get("static_n18", {}).get("top1")
for k, r in res.items():
    agr = float((r["top1"] == ref).mean()) if r["top1"] is not None and ref is not None and len(r["top1"]) == len(ref) else None
    agr_g = float((r["top1"] == res["all_gpu"]["top1"]).mean()) if "all_gpu" in res and r["top1"] is not None and res["all_gpu"]["top1"] is not None and len(r["top1"]) == len(res["all_gpu"]["top1"]) else None
    print(f"{k:11s} {r['tok_s']:7.1f} tok/s  nll {r['nll']:.4f}  seqs {r['n']} steps {r['steps']}  top1 vs static {agr}  vs all-GPU {agr_g}")
if "cache_C8" in res and "static_n18" in res:
    c, s = res["cache_C8"], res["static_n18"]
    st = json.load(open(f"{out}/ec_C8.json")) if os.path.exists(f"{out}/ec_C8.json") else {}
    print(f"speed-up {c['tok_s']/s['tok_s']:.3f}x  dNLL {100*(c['nll']-s['nll'])/s['nll']:+.2f} %  hit rate {st.get('hit_rate')}")
PY
