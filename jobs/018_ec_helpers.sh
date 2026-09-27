#!/bin/bash
# Helpers v2: phases split into claimable chunks (a descheduled helper no longer stalls a request) and an
# AVX-512 VNNI MXFP4 kernel. A: correctness gate against the split-graph cache. B: CPU throughput of the
# helpers' building blocks on this machine. C: A/B at the 25 % budget, each configuration twice, interleaved.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/ec/setup.sh"
R() { timeout 20m "$@" 2>> $OUT/stderr_runs.txt; }
SEQ=4,8,11,14
MB="mailbox=1:tdec=1:helpers=28"

# ---- A. correctness
for spec in "gpt-oss-20b-MXFP4.gguf tok_gpt-oss-20b.jsonl 8 1" "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf tok_qwen3_30b.jsonl 32 2"; do
  set -- $spec; tag=${1%.gguf}
  cfg="slots=$3:policy=dfa:kappa=$4;slots=$3:policy=dfa:kappa=$4:$MB:pin=1:check=1:stats=$OUT/A_${tag}.json"
  R $EC_BIN -m $M/$1 --corpus $J/$2 --seqs 4 --host-experts --ec "$cfg" -t 30 --n-prefill 128 --n-decode 64 --dump $OUT/A_${tag}.bin > $OUT/A_${tag}.jsonl
  echo "A $tag rc=$?"
done
python3 - $OUT <<'PY'
import numpy as np, sys, os, json
out = sys.argv[1]; ok = True
for tag in ["gpt-oss-20b-MXFP4", "Qwen3-30B-A3B-Instruct-2507-Q4_K_M"]:
    def load(i):
        p = f"{out}/A_{tag}.bin.{i}"
        if not os.path.exists(p) or os.path.getsize(p) == 0: return None
        r = np.fromfile(p, dtype=np.int32).reshape(-1, 10); return r[:, :5], r[:, 5:].view(np.float32)
    A, B = load(0), load(1)
    if A is None or B is None: print(tag, "MISSING"); ok = False; continue
    n = min(len(A[0]), len(B[0])); agree = float((A[0][:n, 0] == B[0][:n, 0]).mean()); d = float(np.abs(A[1][:n, 0] - B[1][:n, 0]).max())
    j = json.load(open(f"{out}/A_{tag}.json"))
    print(f"{tag}: {n} steps, top-1 agreement {agree:.3f}, max top-1 logit diff {d:.4f}, check_bad {j['check_bad']}, mxk {j.get('mxk')}")
    ok = ok and n >= 60 and agree >= 0.95 and j["check_bad"] == 0
open(f"{out}/A_ok", "w").write("1" if ok else "0"); print("A:", "PASS" if ok else "FAIL")
PY
[ "$(cat $OUT/A_ok)" = 1 ] || { echo "correctness failed"; exit 1; }

# ---- B. CPU building blocks
$EC_CPUBENCH 2880 > $OUT/cpubench_2880.txt 2>&1; cat $OUT/cpubench_2880.txt
$EC_CPUBENCH 2048 > $OUT/cpubench_2048.txt 2>&1; cat $OUT/cpubench_2048.txt

# ---- C. A/B, twice, interleaved
run_ab() {  # file corpus L E kappa C
  local f=$1 corp=$J/$2 L=$3 E=$4 K=$5 C=$6 tag=${1%.gguf}
  local B="slots=$C:policy=dfa:kappa=$K:$MB"
  local cfg=""
  for rep in 1 2; do
    cfg="$cfg;$B:pin=1:stats=$OUT/${tag}_pin_mxk_r$rep.json"
    cfg="$cfg;$B:pin=1:mxk=0:stats=$OUT/${tag}_pin_nomxk_r$rep.json"
    cfg="$cfg;$B:stats=$OUT/${tag}_nopin_mxk_r$rep.json"
    cfg="$cfg;slots=$C:policy=dfa:kappa=$K:stats=$OUT/${tag}_split_r$rep.json"
  done
  R $EC_BIN -m $M/$f --corpus $corp --seqs $SEQ --host-experts --ec "${cfg#;}" -t 30 --n-prefill 128 --n-decode 128 > $OUT/${tag}_ec.jsonl
  echo "$tag rc=$?"
  for n in $(( L - L/4 )); do
    R $EC_BIN -m $M/$f --corpus $corp --seqs $SEQ --ncmoe $n -t 30 --n-prefill 128 --n-decode 128 > $OUT/${tag}_static_n$n.jsonl
  done
}
run_ab gpt-oss-20b-MXFP4.gguf tok_gpt-oss-20b.jsonl 24 32 1 8
run_ab Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf tok_qwen3_30b.jsonl 48 128 2 32
run_ab gpt-oss-120b-MXFP4.gguf tok_gpt-oss-120b.jsonl 36 128 1 32
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
        busy = [round(b / c, 1) if c else 0 for b, c in zip(s.get("mb_busy_us", []), s.get("mb_count", []))]
        print(f"{rate(v):7.1f} tok/s hit {s.get('hit_rate', 0):.3f} busy_us/m {busy} {os.path.basename(p)} {k.split(':stats')[0][:60]} {os.path.basename(st)}")
PY
