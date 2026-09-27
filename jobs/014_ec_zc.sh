#!/bin/bash
# Zero-copy admissions (slot copies by a kernel reading pinned host memory, so they never queue in the copy
# engine ahead of the step's own transfers) vs cudaMemcpyAsync admissions; overlap on/off with no misses;
# budgets 12.5/25/50 %. gpt-oss-20b and Qwen3-30B-A3B Q4_K_M, 4 test sequences x 128 steps.
# Correctness: top-5 logits with zero-copy and with memcpy admissions must be bit-identical.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/ec/setup.sh"
R() { timeout 25m "$@" 2>> $OUT/stderr_runs.txt; }
PROF=0/1/2/3/5/6/9/17
SEQ=4,8,11,14
for spec in "gpt-oss-20b-MXFP4.gguf tok_gpt-oss-20b.jsonl 24 32" "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf tok_qwen3_30b.jsonl 48 128"; do
  set -- $spec; f=$1; corp=$J/$2; L=$3; E=$4; tag=${f%.gguf}
  C1=$(( E/8 )); C2=$(( E/4 )); C3=$(( E/2 ))
  python3 -c "
for il in range($L): print(il, *range($E))" > $OUT/allres_$E.txt
  cfg="slots=1:policy=static:stats=$OUT/${tag}_profile.json:seqs=$PROF"
  cfg="$cfg;slots=$E:policy=static:init=$OUT/allres_$E.txt:stats=$OUT/${tag}_allres.json"
  cfg="$cfg;slots=$E:policy=static:init=$OUT/allres_$E.txt:overlap=0:stats=$OUT/${tag}_allres_serial.json"
  cfg="$cfg;slots=$C2:policy=static:init=$OUT/${tag}_profile.json:stats=$OUT/${tag}_static_C$C2.json"
  cfg="$cfg;slots=$C2:policy=static:init=$OUT/${tag}_profile.json:overlap=0:stats=$OUT/${tag}_static_serial_C$C2.json"
  cfg="$cfg;slots=$C2:policy=dfa:zc=1:stats=$OUT/${tag}_dfa_zc_C$C2.json"
  cfg="$cfg;slots=$C2:policy=dfa:zc=0:stats=$OUT/${tag}_dfa_dma_C$C2.json"
  cfg="$cfg;slots=$C2:policy=dfa:zc=1:overlap=0:stats=$OUT/${tag}_dfa_zc_serial_C$C2.json"
  cfg="$cfg;slots=$C2:policy=lru:zc=1:stats=$OUT/${tag}_lru_zc_C$C2.json"
  cfg="$cfg;slots=$C1:policy=dfa:zc=1:stats=$OUT/${tag}_dfa_zc_C$C1.json"
  cfg="$cfg;slots=$C3:policy=dfa:zc=1:stats=$OUT/${tag}_dfa_zc_C$C3.json"
  R $EC_BIN -m $M/$f --corpus $corp --seqs $SEQ --host-experts --ec "$cfg" -t 30 --n-prefill 128 --n-decode 128 --dump $OUT/${tag}_ec.bin > $OUT/${tag}_ec.jsonl
  for n in $(( L*7/8 )) $(( L*3/4 )) $(( L/2 )) $L 0; do
    R $EC_BIN -m $M/$f --corpus $corp --seqs $SEQ --ncmoe $n -t 30 --n-prefill 128 --n-decode 128 > $OUT/${tag}_static_n$n.jsonl
  done
  cmp $OUT/${tag}_ec.bin.5 $OUT/${tag}_ec.bin.6 && echo "$tag: zero-copy and memcpy admissions give identical logits" || echo "$tag: LOGITS DIFFER between zero-copy and memcpy admissions"
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
        print(f"{rate(v):7.1f} tok/s hit {s.get('hit_rate', 0):.3f} adm/step {s.get('admits', 0)/max(1, s.get('steps', 1)):5.1f} zc {s.get('zc')} {os.path.basename(p)} {k.split(':stats')[0][:60]}")
PY
