#!/bin/bash
# Mailbox mode: the GPU hands a layer's CPU experts to spinning helper threads through pinned memory and waits
# on the GPU, instead of the scheduler splitting the graph (host sync + copies + CPU graph per layer).
# A. correctness: top-5 logits of mailbox vs split-graph cache (DFA) and vs stock CPU experts; ids checked.
# B. speed at equal VRAM, same instance: llama.cpp static layers, split-graph cache (015 best), mailbox cache,
#    mailbox with the static-layer layout (hand-off only), 28 vs 14 helpers.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/ec/setup.sh"
R() { timeout 20m "$@" 2>> $OUT/stderr_runs.txt; }
PROF=0/1/2/3/5/6/9/17
SEQ=4,8,11,14
alloc() {  # file L E n: first n layers on the CPU (0 slots), the rest whole on the GPU
  python3 -c "
L,E,n=$2,$3,$4
print('\n'.join(f'{il} {0 if il < n else E}' for il in range(L)))" > $1
}

# ---- A. correctness (gpt-oss-20b, one sequence, 64 steps)
f=gpt-oss-20b-MXFP4.gguf; corp=$J/tok_gpt-oss-20b.jsonl
alloc $OUT/a_allcpu_24.txt 24 32 24
MB="mailbox=1:helpers=28:tdec=1"
cfg="slots=8:policy=dfa:kappa=1"
cfg="$cfg;slots=8:policy=dfa:kappa=1:$MB:check=1:stats=$OUT/A_mb_dfa.json"
cfg="$cfg;slots=1:alloc=$OUT/a_allcpu_24.txt"
cfg="$cfg;slots=1:alloc=$OUT/a_allcpu_24.txt:$MB:stats=$OUT/A_mb_allcpu.json"
R $EC_BIN -m $M/$f --corpus $corp --seqs 4 --host-experts --ec "$cfg" -t 30 --n-prefill 128 --n-decode 64 --dump $OUT/A.bin > $OUT/A.jsonl
echo "A rc=$?"
grep -E "timed out|GGML_ASSERT|error|mailbox" $OUT/stderr_runs.txt | head -20
python3 - $OUT <<'PY'
import numpy as np, sys, os, json
out = sys.argv[1]
def load(i):
    p = f"{out}/A.bin.{i}"
    if not os.path.exists(p) or os.path.getsize(p) == 0: return None
    r = np.fromfile(p, dtype=np.int32).reshape(-1, 10)
    return r[:, :5], r[:, 5:].view(np.float32)
ok = True
for a, b, what in [(0, 1, "mailbox DFA vs split-graph DFA"), (2, 3, "mailbox all-CPU vs stock all-CPU")]:
    A, B = load(a), load(b)
    if A is None or B is None:
        print(f"{what}: MISSING"); ok = False; continue
    n = min(len(A[0]), len(B[0]))
    agree = float((A[0][:n, 0] == B[0][:n, 0]).mean())
    d = float(np.abs(A[1][:n, 0] - B[1][:n, 0]).max())
    print(f"{what}: {n} steps, top-1 agreement {agree:.3f}, max top-1 logit diff {d:.4f}")
    ok = ok and n >= 60 and agree >= 0.95
for s in ["A_mb_dfa.json", "A_mb_allcpu.json"]:
    p = f"{out}/{s}"
    if os.path.exists(p):
        j = json.load(open(p))
        print(s, "hit", j["hit_rate"], "check_bad", j["check_bad"], "requests", sum(l.get("mb_requests", 0) for l in j["layers"]),
              "experts", sum(l.get("mb_experts", 0) for l in j["layers"]))
open(f"{out}/A_ok", "w").write("1" if ok else "0")
print("A:", "PASS" if ok else "FAIL")
PY
python3 - $OUT/A.jsonl <<'PY'
import json, sys
for l in open(sys.argv[1]):
    r = json.loads(l); print(f"{r['tok_s']:7.1f} tok/s  {r['config'][:90]}")
PY
[ "$(cat $OUT/A_ok)" = 1 ] || { echo "correctness failed; skipping B"; exit 1; }

# ---- B. speed
run_model() {  # file corpus L E kappa "budgets (1/8 units)"
  local f=$1 corp=$J/$2 L=$3 E=$4 K=$5 FR="$6" tag=${1%.gguf}
  local cfg=""
  for q in $FR; do
    local C=$(( E*q/8 )) n=$(( L - L*q/8 ))
    alloc $OUT/${tag}_layers_n$n.txt $L $E $n
    cfg="$cfg;slots=$C:policy=dfa:kappa=$K:stats=$OUT/${tag}_C${C}_split.json"
    cfg="$cfg;slots=$C:policy=dfa:kappa=$K:$MB:stats=$OUT/${tag}_C${C}_mb.json"
    cfg="$cfg;slots=$C:policy=dfa:kappa=$K:mailbox=1:helpers=14:tdec=1:stats=$OUT/${tag}_C${C}_mb14.json"
    cfg="$cfg;slots=1:alloc=$OUT/${tag}_layers_n$n.txt:$MB:stats=$OUT/${tag}_n${n}_mblayers.json"
  done
  R $EC_BIN -m $M/$f --corpus $corp --seqs $SEQ --host-experts --ec "${cfg#;}" -t 30 --n-prefill 128 --n-decode 128 > $OUT/${tag}_ec.jsonl
  echo "$tag rc=$?"
  for q in $FR; do
    local n=$(( L - L*q/8 ))
    R $EC_BIN -m $M/$f --corpus $corp --seqs $SEQ --ncmoe $n -t 30 --n-prefill 128 --n-decode 128 > $OUT/${tag}_static_n$n.jsonl
  done
}
run_model gpt-oss-20b-MXFP4.gguf tok_gpt-oss-20b.jsonl 24 32 1 "1 2 4"
run_model Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf tok_qwen3_30b.jsonl 48 128 2 "1 2 4"
run_model gpt-oss-120b-MXFP4.gguf tok_gpt-oss-120b.jsonl 36 128 1 "1 2"
run_model Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf tok_qwen3_30b.jsonl 48 128 2 "1 2"
python3 - "$OUT" <<'PY'
import json, sys, glob, os
out = sys.argv[1]
def rate(rows): return sum(r["n_decode"] for r in rows) / (sum(r["decode_ms"] for r in rows) / 1000)
for p in sorted(glob.glob(f"{out}/*.jsonl")):
    by = {}
    for l in open(p):
        if l.strip():
            r = json.loads(l); by.setdefault(r.get("config", ""), []).append(r)
    for k, v in by.items():
        st = k.split("stats=")[-1].split(":")[0] if "stats=" in k else ""
        s = json.load(open(st)) if st and os.path.exists(st) else {}
        print(f"{rate(v):7.1f} tok/s hit {s.get('hit_rate', 0):.3f} adm/step {s.get('admits', 0)/max(1, s.get('steps', 1)):5.1f} {os.path.basename(p)} {k.split(':stats')[0][:70]}")
PY
