#!/bin/bash
# Checkpoint 2 on a home-PC-class host (RTX 5090 + Ryzen 9 9950X/9950X3D, dual-channel DDR5, >= 126 GB, whole machine;
# Vast container, clocks not locked): where the expert cache (jobs/ec2 patch on llama.cpp 4da6337) stands against
# llama.cpp on gpt-oss-120b MXFP4, own text (12 test prompts, whole prompt prefilled, 128 tokens decoded), at
# -ncmoe 32 / 27 / 20 (cache: 14 / 32 / 56 of 128 experts per layer), plus the calibration that
# research_notes/.../admission_simulation.md section 8 asks for (concurrent DRAM + PCIe reads, fetch latency, helper
# service times). llama.cpp is measured twice: stock llama-bench (tg128 at depth 512, 5 repetitions, two thread
# counts) and stock --ncmoe inside ec-bench on the same prompts as the cache.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
platform
CORES=$(lscpu -p=CORE | grep -v '^#' | sort -u | wc -l); echo "physical cores $CORES" | tee $OUT/cores.txt
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && timeout 15m $WORK/concur > $OUT/concur.txt 2>&1
build_all
# checkpoint 1 op tests that job 058 could not reach (its multi-token cases aborted the run): id -1 on every type
timeout 20m $EC_TESTS -b CUDA0 -o MUL_MAT_ID_EC > $OUT/ops_mmid_ec.txt 2>&1; echo "ops MUL_MAT_ID_EC rc=$?"
timeout 30m compute-sanitizer --tool memcheck --error-exitcode 9 $EC_TESTS -b CUDA0 -o MUL_MAT_ID_EC > $OUT/memcheck_mmid_ec.txt 2>&1; echo "memcheck rc=$?"
getmodel ggml-org/gpt-oss-120b-GGUF gpt-oss-120b-MXFP4.gguf
F=$M/gpt-oss-120b-MXFP4.gguf
R() { local t=$1; shift; timeout "$t" "$@"; }
# the cache first (the page cache is warm from the download), warm-up configuration discarded
H=$(( CORES > 4 ? CORES - 2 : 2 )); MB="mailbox=1:tdec=1:helpers=$H"
COMMON="--corpus $J/S_gpt-oss-120b.jsonl --seqs 0,1,2,3,4,5,6,7,8,9,10,11 -t $CORES --n-prefill 640 --n-decode 128 --no-mmap"
CFG="slots=14:policy=dfa:kappa=1:$MB"
for C in 14 32 56; do CFG="$CFG;slots=$C:policy=dfa:kappa=1:$MB:stats=$OUT/ec_C$C.json"; done
R 60m $EC_BIN -m $F $COMMON --host-experts --ec "$CFG" --dump $OUT/ec.bin > $OUT/ec.jsonl 2> $OUT/ec.err; echo "ec rc=$?"
for n in 32 27 20; do
  R 40m $EC_BIN -m $F $COMMON --ncmoe $n --dump $OUT/static_n$n.bin > $OUT/static_n$n.jsonl 2> $OUT/static_n$n.err; echo "static n=$n rc=$?"
done
for n in 32 27 20; do for t in $CORES $(( CORES / 2 )); do
  R 30m $STOCK_BENCH -m $F -ngl 99 -ncmoe $n -fa on -lm none -t $t -p 0 -n 128 -d 512 -r 5 -o jsonl >> $OUT/llamabench.jsonl 2>> $OUT/llamabench.err
  echo "llama-bench n=$n t=$t rc=$?"
done; done
# the same with FETCH (a layer's misses split between fetching into slots and the helpers; table from the
# trace simulation at estimated constants, research_notes/.../admission_simulation.md section 7)
CFGF="slots=14:policy=dfa:kappa=1:$MB:fetch=0,1,1,2,3"
for C in 14 32 56; do CFGF="$CFGF;slots=$C:policy=dfa:kappa=1:$MB:fetch=0,1,1,2,3:stats=$OUT/ecf_C$C.json"; done
R 60m $EC_BIN -m $F $COMMON --host-experts --ec "$CFGF" --dump $OUT/ecf.bin > $OUT/ecf.jsonl 2> $OUT/ecf.err; echo "ec fetch rc=$?"
python3 - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, sys, os, glob
import numpy as np
out = sys.argv[1]
def runs(p):
    by = {}
    for l in open(p):
        if l.strip():
            r = json.loads(l); by.setdefault(r.get("config", ""), []).append(r)
    return by
def rate(v): return sum(r["n_decode"] for r in v) / (sum(r["decode_ms"] for r in v) / 1000)
def nll(v): return sum(r["nll_sum"] for r in v) / max(1, sum(r["nll_n"] for r in v))
import re
for f in sorted(glob.glob(f"{out}/ops_*.txt") + glob.glob(f"{out}/memcheck_*.txt")):
    t = open(f, errors="replace").read()
    err = re.findall(r"ERROR SUMMARY: (\d+) error", t)
    print(f"{os.path.basename(f)}: OK {t.count('OK')} FAIL {t.count('FAIL')}" + (f" memcheck errors {err[-1]}" if err else "") +
          " | " + " ".join(l.strip() for l in t.splitlines() if "tests passed" in l)[:120])
def top1(p): return np.fromfile(p, dtype=np.int32).reshape(-1, 10)[:, 0] if os.path.exists(p) and os.path.getsize(p) else None
ours, oursf, stat, lb = {}, {}, {}, {}
for tag, dd in (("ec", ours), ("ecf", oursf)):
    if os.path.exists(f"{out}/{tag}.jsonl") and os.path.getsize(f"{out}/{tag}.jsonl"):
        by = runs(f"{out}/{tag}.jsonl"); keys = list(by)
        for i, k in enumerate(keys[1:], start=1):
            C = int(k.split("slots=")[1].split(":")[0]); st = f"{out}/{tag}_C{C}.json"
            dd[C] = (rate(by[k]), nll(by[k]), top1(f"{out}/{tag}.bin.{i}"), json.load(open(st)) if os.path.exists(st) else {})
for n in (32, 27, 20):
    p = f"{out}/static_n{n}.jsonl"
    if os.path.exists(p) and os.path.getsize(p):
        v = list(runs(p).values())[0]; stat[n] = (rate(v), nll(v), top1(f"{out}/static_n{n}.bin"))
if os.path.exists(f"{out}/llamabench.jsonl"):
    for l in open(f"{out}/llamabench.jsonl"):
        if l.strip().startswith("{"):
            r = json.loads(l); s = np.array(r.get("samples_ns", []), float)
            hm = r["n_gen"] * len(s) / (s.sum() / 1e9) if len(s) else r.get("avg_ts", 0)
            n = r.get("n_cpu_moe"); lb.setdefault(n, []).append((hm, r.get("n_threads"), r.get("avg_ts"), r.get("stddev_ts")))
for n, C in ((32, 14), (27, 32), (20, 56)):
    o = ours.get(C); s = stat.get(n); b = max(lb.get(n, [(0, None, None, None)]))
    line = f"ncmoe {n} / C {C}: "
    if o: line += f"ours {o[0]:6.1f} tok/s nll {o[1]:.4f} hit {o[3].get('hit_rate')} | "
    if s: line += f"static(ec-bench) {s[0]:6.1f} nll {s[1]:.4f} | "
    line += f"llama-bench best {b[0]:6.1f} (t={b[1]}) | "
    best_ll = max(s[0] if s else 0, b[0])
    if o and best_ll: line += f"R_ll {o[0]/best_ll:.3f}"
    if o and s and o[2] is not None and s[2] is not None and len(o[2]) == len(s[2]): line += f" top1 agree {float((o[2]==s[2]).mean()):.4f} dNLL {100*(o[1]-s[1])/s[1]:+.2f}%"
    print(line)
    of = oursf.get(C)
    if of:
        line = f"   + fetch: {of[0]:6.1f} tok/s nll {of[1]:.4f} hit {of[3].get('hit_rate')} fetches {of[3].get('fetches')}"
        if o: line += f" | vs cache {of[0]/o[0]:.3f}x"
        if best_ll: line += f" | R_ll {of[0]/best_ll:.3f}"
        if s and of[2] is not None and s[2] is not None and len(of[2]) == len(s[2]): line += f" top1 agree {float((of[2]==s[2]).mean()):.4f} dNLL {100*(of[1]-s[1])/s[1]:+.2f}%"
        print(line)
PY
