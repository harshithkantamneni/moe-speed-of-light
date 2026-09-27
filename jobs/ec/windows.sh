# Shared by jobs 025/026: phase-3 systems at equal expert VRAM with correct decode windows (source after setup.sh).
R() { timeout 40m "$@" 2>> $OUT/stderr_runs.txt; }
MB="mailbox=1:tdec=1:helpers=28"
NP=640   # prefill the whole prompt (the longest test prompt is 612 tokens, Qwen3 seq 32)
alloc() { python3 -c "
L,E,n=$2,$3,$4
print('\n'.join(f'{il} {0 if il < n else E}' for il in range(L)))" > $1; }
run_model() {  # file corpus L E kappa "budgets (1/8)" seqs allgpu(0/1) arm
  local f=$1 corp=$2 L=$3 E=$4 K=$5 FR="$6" SEQ=$7 AG=$8 arm=$9 tag=${1%.gguf}_$9
  local cfg=""
  for q in $FR; do
    local C=$(( E*q/8 )) n=$(( L - L*q/8 ))
    alloc $OUT/${tag}_n$n.txt $L $E $n
    cfg="$cfg;slots=$C:policy=dfa:kappa=$K:$MB:stats=$OUT/${tag}_mb_C$C.json"
    cfg="$cfg;slots=1:alloc=$OUT/${tag}_n$n.txt:$MB:stats=$OUT/${tag}_mb_n$n.json"
  done
  R $EC_BIN -m $M/$f --corpus $corp --seqs $SEQ --host-experts --ec "${cfg#;}" -t 30 --n-prefill $NP --n-decode 192 --dump $OUT/${tag}_ec.bin > $OUT/${tag}_ec.jsonl
  echo "$tag ec rc=$?"
  for q in $FR; do
    local n=$(( L - L*q/8 ))
    R $EC_BIN -m $M/$f --corpus $corp --seqs $SEQ --ncmoe $n -t 30 --n-prefill $NP --n-decode 192 --dump $OUT/${tag}_static_n$n.bin > $OUT/${tag}_static_n$n.jsonl
    echo "$tag static n=$n rc=$?"
  done
  if [ "$AG" = 1 ]; then
    R $EC_BIN -m $M/$f --corpus $corp --seqs $SEQ --ncmoe 0 -t 30 --n-prefill $NP --n-decode 192 --dump $OUT/${tag}_static_n0.bin > $OUT/${tag}_static_n0.jsonl
    echo "$tag all-GPU rc=$?"
  fi
}
summary() {
python3 - "$OUT" <<'PY'
import json, sys, glob, os
out = sys.argv[1]
def rate(rows): return sum(r["n_decode"] for r in rows) / (sum(r["decode_ms"] for r in rows) / 1000)
for p in sorted(glob.glob(f"{out}/*_ec.jsonl") + glob.glob(f"{out}/*_static_n*.jsonl")):
    by = {}
    for l in open(p):
        if l.strip():
            r = json.loads(l); by.setdefault(r.get("config", ""), []).append(r)
    for k, v in by.items():
        st = k.split("stats=")[-1].split(":")[0] if "stats=" in k else ""
        s = json.load(open(st)) if st and os.path.exists(st) else {}
        nll = sum(r["nll_sum"] for r in v) / max(1, sum(r["nll_n"] for r in v))
        print(f"{rate(v):7.1f} tok/s nll {nll:.4f} hit {s.get('hit_rate', 0):.3f} n_seq {len(v)} steps {sum(r['n_decode'] for r in v)} {os.path.basename(p)} {k.split(':stats')[0][:50]}")
PY
}
