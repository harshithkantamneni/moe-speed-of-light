#!/bin/bash
# Diagnosis of the phase-2 slowdown (009: the cache ran at all-CPU speed although 68% of experts hit).
# gpt-oss-20b, 25% budget (8 of 32 slots), 4 test sequences x 128 decode steps, ids checked on-device.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/ec/setup.sh"
F=$M/gpt-oss-20b-MXFP4.gguf; C=$J/tok_gpt-oss-20b.jsonl; SEQ=4,10,16,26; T=30
COMMON="--corpus $C -t $T --n-prefill 128 --n-decode 128"
python3 -c "
for il in range(24): print(il, *range(32))" > $OUT/all_resident.txt
R() { timeout 25m "$@" 2>> $OUT/stderr_runs.txt; }
cfg="slots=1:policy=static:stats=$OUT/profile.json:seqs=0/1/2/3/5/6/9/17"
cfg="$cfg;slots=32:policy=static:init=$OUT/all_resident.txt:check=1:stats=$OUT/s_allres.json"
cfg="$cfg;slots=8:policy=static:init=$OUT/profile.json:check=1:stats=$OUT/s_static.json"
cfg="$cfg;slots=8:policy=dfa:paced=0:check=1:stats=$OUT/s_dfa_batched.json"
cfg="$cfg;slots=8:policy=dfa:paced=0:overlap=0:check=1:stats=$OUT/s_dfa_batched_serial.json"
cfg="$cfg;slots=8:policy=dfa:paced=1:check=1:stats=$OUT/s_dfa_paced.json"
cfg="$cfg;slots=8:policy=dfa:paced=1:overlap=0:check=1:stats=$OUT/s_dfa_paced_serial.json"
cfg="$cfg;slots=8:policy=dfa:paced=1:max_admit=2:check=1:stats=$OUT/s_dfa_paced_ma2.json"
cfg="$cfg;slots=8:policy=dfa:paced=1:chunk=4194304:check=1:stats=$OUT/s_dfa_paced_c4m.json"
cfg="$cfg;slots=8:policy=lru:paced=1:check=1:stats=$OUT/s_lru_paced.json"
cfg="$cfg;slots=8:policy=lru:paced=1:max_pending=4:check=1:stats=$OUT/s_lru_paced_mp4.json"
R $EC_BIN -m $F --seqs $SEQ --host-experts --ec "$cfg" $COMMON > $OUT/ec.jsonl
for n in 0 18 24; do R $EC_BIN -m $F --seqs $SEQ --ncmoe $n $COMMON > $OUT/static_n$n.jsonl; done
python3 - "$OUT" <<'PY'
import json, sys, glob, os
out = sys.argv[1]
def rate(rows): return sum(r["n_decode"] for r in rows) / (sum(r["decode_ms"] for r in rows) / 1000)
by = {}
for l in open(f"{out}/ec.jsonl"):
    r = json.loads(l); by.setdefault(r["config"], []).append(r)
for k, v in by.items():
    st = k.split("stats=")[-1].split(":")[0]
    s = json.load(open(st)) if os.path.exists(st) else {}
    print(f"{rate(v):7.1f} tok/s  hit {s.get('hit_rate', 0):.3f}  admits/step {s.get('admits', 0)/max(1, s.get('steps', 1)):5.1f}  bad {s.get('check_bad')}  gpu {s.get('check_gpu_frac', 0):.3f}  {k.split(':stats')[0]}")
for p in sorted(glob.glob(f"{out}/static_n*.jsonl")):
    print(f"{rate([json.loads(l) for l in open(p)]):7.1f} tok/s  {os.path.basename(p)}")
PY
# timelines (32 steps): batched vs paced DFA, and the all-resident cache (pure overhead)
NSYS=$(command -v nsys || ls /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1)
if [ -z "$NSYS" ]; then
  sudo apt-get install -y -qq cuda-nsight-systems-12-8 > /dev/null 2>&1; NSYS=$(ls /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1)
fi
for spec in "batched slots=8:policy=dfa:paced=0" "paced slots=8:policy=dfa:paced=1" "allres slots=32:policy=static:init=$OUT/all_resident.txt"; do
  set -- $spec; tag=$1
  timeout 10m $NSYS profile -t cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none -o $WORK/nsys_ec_$tag --force-overwrite true \
    $EC_BIN -m $F --seqs 4 --host-experts --ec "$2" --corpus $C -t $T --n-prefill 128 --n-decode 32 > /dev/null 2>> $OUT/stderr_runs.txt
  timeout 5m $NSYS stats -r cuda_gpu_trace -f csv -o $WORK/nsys_ec_$tag $WORK/nsys_ec_$tag.nsys-rep > /dev/null 2>&1
  gzip -c $WORK/nsys_ec_${tag}_cuda_gpu_trace.csv > $OUT/nsys_ec_${tag}_gpu_trace.csv.gz
done
ls -la $OUT
