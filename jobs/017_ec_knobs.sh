#!/bin/bash
# Exploratory: mailbox-mode knobs at the 25 % budget (helper count and pinning, admission-kernel blocks, DFA
# half-life, per-step admission cap), and anchors for the step-time model: all experts on the GPU, all on the
# helpers, and the static-layer layout on the helpers. Helper service times by number of CPU experts are in
# the stats files (mb_busy_us / mb_count), per-layer miss histograms in miss_hist.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/ec/setup.sh"
R() { timeout 20m "$@" 2>> $OUT/stderr_runs.txt; }
SEQ=4,8,11,14
alloc() { python3 -c "
L,E,n=$2,$3,$4
print('\n'.join(f'{il} {0 if il < n else E}' for il in range(L)))" > $1; }
run_model() {  # file corpus L E kappa C
  local f=$1 corp=$J/$2 L=$3 E=$4 K=$5 C=$6 tag=${1%.gguf}
  alloc $OUT/${tag}_allcpu.txt $L $E $L
  alloc $OUT/${tag}_half.txt $L $E $(( L/2 ))
  local B="slots=$C:policy=dfa:kappa=$K:mailbox=1:tdec=1"
  local cfg="$B:helpers=28:stats=$OUT/${tag}_h28.json"
  cfg="$cfg;$B:helpers=28:pin=1:stats=$OUT/${tag}_h28p.json"
  cfg="$cfg;$B:helpers=29:pin=1:stats=$OUT/${tag}_h29p.json"
  cfg="$cfg;$B:helpers=24:pin=1:stats=$OUT/${tag}_h24p.json"
  cfg="$cfg;$B:helpers=28:zc_blocks=8:stats=$OUT/${tag}_zc8.json"
  cfg="$cfg;$B:helpers=28:zc_blocks=32:stats=$OUT/${tag}_zc32.json"
  cfg="$cfg;$B:helpers=28:half_life=8:stats=$OUT/${tag}_hl8.json"
  cfg="$cfg;$B:helpers=28:max_admit=2:stats=$OUT/${tag}_ma2.json"
  cfg="$cfg;$B:helpers=28:max_admit=6:stats=$OUT/${tag}_ma6.json"
  cfg="$cfg;slots=1:alloc=$OUT/${tag}_allcpu.txt:mailbox=1:tdec=1:helpers=28:stats=$OUT/${tag}_mb_allcpu.json"
  cfg="$cfg;slots=1:alloc=$OUT/${tag}_half.txt:mailbox=1:tdec=1:helpers=28:stats=$OUT/${tag}_mb_half.json"
  R $EC_BIN -m $M/$f --corpus $corp --seqs $SEQ --host-experts --ec "$cfg" -t 30 --n-prefill 128 --n-decode 128 > $OUT/${tag}_ec.jsonl
  echo "$tag rc=$?"
  for n in 0 $(( L/2 )) $L; do
    R $EC_BIN -m $M/$f --corpus $corp --seqs $SEQ --ncmoe $n -t 30 --n-prefill 128 --n-decode 128 > $OUT/${tag}_static_n$n.jsonl
  done
}
run_model gpt-oss-20b-MXFP4.gguf tok_gpt-oss-20b.jsonl 24 32 1 8
run_model Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf tok_qwen3_30b.jsonl 48 128 2 32
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
        print(f"{rate(v):7.1f} tok/s hit {s.get('hit_rate', 0):.3f} adm/step {s.get('admits', 0)/max(1, s.get('steps', 1)):5.1f} busy_us/m {busy} {os.path.basename(p)} {k.split(':stats')[0][:70]}")
PY
