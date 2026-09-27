#!/bin/bash
# (a) Pre-registered phase-2 protocol (prereg/a10_ec) on gpt-oss-20b and Qwen3-30B-A3B Q4_K_M: 16 test
#     sequences x 192 decode steps, budgets 12.5/25/50 %, configurations exactly as pre-registered
#     (static hot set, LRU, DFA, DFA serial; batched copies) against llama.cpp static layers.
# (b) Overhead decomposition and admission control (diagnosed in 012): all-resident cache with and without
#     overlap/check; DFA with the r* admission margin (kappa ~ r*), with a per-step budget; LRU with a queue cap.
# (c) CPU-bandwidth sensitivity: the same at 8 threads (STREAM ~90 GB/s, desktop-class) vs 30 (~152 GB/s).
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/ec/setup.sh"
R() { timeout 30m "$@" 2>> $OUT/stderr_runs.txt; }
PROF=0/1/2/3/5/6/9/17
TEST=4,7,8,10,11,12,14,15,16,18,19,20,25,26,27,31
TESTQ=4,7,8,10,11,12,14,15,16,18,19,20,25,26,27,32
prereg() {  # file corpus L E test "ns" allgpu
  local f=$1 corp=$J/$2 L=$3 E=$4 TS=$5 NS="$6" tag=${1%.gguf}
  local cfg="slots=1:policy=static:stats=$OUT/${tag}_profile.json:seqs=$PROF"
  for n in $NS; do
    local C=$(( (E*(L-n) + L/2) / L ))
    for v in "static:init=$OUT/${tag}_profile.json" "lru" "dfa" "dfa:overlap=0"; do
      local nm=$(echo $v | cut -d: -f1)$(echo $v | grep -q overlap=0 && echo _serial)
      cfg="$cfg;slots=$C:paced=0:policy=$v:stats=$OUT/${tag}_C${C}_${nm}.json"
    done
  done
  R $EC_BIN -m $M/$f --corpus $corp --seqs $TS --host-experts --ec "$cfg" -t 30 --n-prefill 128 --n-decode 192 > $OUT/${tag}_ec.jsonl
  for n in $NS $L 0; do
    R $EC_BIN -m $M/$f --corpus $corp --seqs $TS --ncmoe $n -t 30 --n-prefill 128 --n-decode 192 > $OUT/${tag}_static_n$n.jsonl
  done
}
prereg gpt-oss-20b-MXFP4.gguf tok_gpt-oss-20b.jsonl 24 32 $TEST "21 18 12"
prereg Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf tok_qwen3_30b.jsonl 48 128 $TESTQ "42 36 24"

# (b)+(c): 25 % budget, 8 test sequences x 128 steps
SEQ=4,8,11,14,16,19,26,31
for T in 30 8; do
  for spec in "gpt-oss-20b-MXFP4.gguf tok_gpt-oss-20b.jsonl 24 32 18 8" "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf tok_qwen3_30b.jsonl 48 128 36 9"; do
    set -- $spec; f=$1; corp=$J/$2; L=$3; E=$4; n=$5; KAPPA=$6; tag=${f%.gguf}_t$T; C=$(( E*(L-n)/L ))
    python3 -c "
for il in range($L): print(il, *range($E))" > $OUT/allres_$E.txt
    cfg="slots=1:policy=static:stats=$OUT/${tag}_profile.json:seqs=$PROF"
    cfg="$cfg;slots=$E:policy=static:init=$OUT/allres_$E.txt:stats=$OUT/${tag}_allres.json"
    cfg="$cfg;slots=$E:policy=static:init=$OUT/allres_$E.txt:overlap=0:stats=$OUT/${tag}_allres_serial.json"
    cfg="$cfg;slots=$C:policy=static:init=$OUT/${tag}_profile.json:stats=$OUT/${tag}_static.json"
    cfg="$cfg;slots=$C:policy=dfa:paced=0:stats=$OUT/${tag}_dfa.json"
    cfg="$cfg;slots=$C:policy=dfa:paced=0:kappa=$KAPPA:stats=$OUT/${tag}_dfa_rstar.json"
    cfg="$cfg;slots=$C:policy=dfa:paced=0:max_admit=2:stats=$OUT/${tag}_dfa_ma2.json"
    cfg="$cfg;slots=$C:policy=dfa:paced=0:kappa=$KAPPA:overlap=0:stats=$OUT/${tag}_dfa_rstar_serial.json"
    cfg="$cfg;slots=$C:policy=lru:paced=1:max_pending=4:stats=$OUT/${tag}_lru_mp4.json"
    R $EC_BIN -m $M/$f --corpus $corp --seqs $SEQ --host-experts --ec "$cfg" -t $T --n-prefill 128 --n-decode 128 > $OUT/${tag}_ec.jsonl
    for nn in $n $L 0; do
      R $EC_BIN -m $M/$f --corpus $corp --seqs $SEQ --ncmoe $nn -t $T --n-prefill 128 --n-decode 128 > $OUT/${tag}_static_n$nn.jsonl
    done
  done
done
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
