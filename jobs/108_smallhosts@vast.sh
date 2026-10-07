#!/bin/bash
# Job 108: the time relation (eq. sum, with its overlap term, as registered in jobs 106 and 107) registered on machines
# never rented before, restricted in advance to the class it was found on: one NUMA node and at most 32 usable
# physical cores ("desktop-class"; the line was drawn after job 107, from jobs 093-107, and is fixed here before any
# machine starts). gpt-oss-120b only (Qwen3 would double the download), the deployed cache only (base: decayed
# count, kappa 1, the host's fetch table, admissions copied in the background), at C = 14 (11%) and C = 32 (25%),
# two rounds each (separate processes), 20 AIME-25 problems x 256 teacher-forced steps. Engine: patch oracle5, as
# jobs 106-107. Before any timed run, Nsight Systems profiles base at each budget (one problem, 24 decode tokens):
# G = the GPU's own kernel time per token (every kernel but the helper wait and the expert copies). The relation:
#   T = G + (M + A (1 - G/T)) S / B_host,  solved for T,
# M the misses and A the background admissions per token (the run's counters, mean over rounds), S = 13,253,760 bytes,
# B_host the probe's highest host rate; T is compared with base's time per token, mean over the two rounds.
# VALIDITY, registered here:
#   V0 (before anything is installed or downloaded): the GPU's UUID is in no earlier launch (jobs/ec2/known_gpu_uuids.txt,
#      which now also lists job 107's), at most 48 GB of host memory in use;
#   V0c (after the tools are installed, before any download): one NUMA node (lscpu) and at most 32 usable physical
#      cores (setup.sh usable_cores); device reads >= 1500 GB/s; disk >= 100 GB;
#   V1: the deployed cache's loss per token (mean over problems) within 2% of 0.190 in every round (checked after the
#      first round; a failing host stops);
#   V2: base's time per token at C = 14 differs by at most 2% between its two rounds (max/min - 1; a failing host stops).
# Hosts are rented from the offer list below, in order, until three machines are valid or the credit falls below
# $0.60; a host that fails V0, V0c, V1 or V2 is replaced by the next offer. Every host that starts is reported. With
# fewer than three valid machines the test is inconclusive.
# Predictions, committed before launch, over the valid machines:
#   1. G_prof 4.0-5.0 ms at both budgets;
#   2. the relation within 8% of base's time at every cell (machine x budget), at most one cell beyond 6%, and the
#      median |error| over all cells at most 4%. (The 8% band is the spread of the launches it was found on: on the 37
#      desktop-class launches of jobs 093-106 that ran stably, 35 are within 8% and 32 within 6% at gpt-oss 11%, and 36
#      within both at 25%. Job 107's new desktop, a Ryzen 9 5900XT, was off by 7.1% and 2.3% at these two cells.)
#   3. the read rate the deployed cache implies, (M + A (1 - G/T)) S / (T - G), is 0.85-1.25 of B_host at every cell
#      (found on those 37 launches: 0.88-1.16 at 11%, 0.94-1.22 at 25%).
# Reported without a prediction: the plain form (without (1 - G/T)).
# Offers, in order (none in the rental ledger; verified first, then by expected cost, rate plus 65 GB of traffic):
# 43575030 (EPYC 7543), 32984223 (EPYC 7K62), 53424353 (Ryzen 7 5700X3D), 54156078 (Ryzen 9 9950X3D), 45485624
# (EPYC 9354), 54573924 (Ryzen 9 5950X), 51325952 (Core i9-13900KF), 47251469 (Ryzen 9 7950X).
# Budget: the job stops starting steps 75 minutes after launch.
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
# --- V0, before anything is installed or downloaded
JD="$(cd "$(dirname "$0")" && pwd)"
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1 | tr -d ' ')
free -g > $OUT/free.txt; USEDGB=$(awk '/Mem:/{print $3}' $OUT/free.txt)
echo "GPU UUID $UUID; host memory in use ${USEDGB} GB" | tee $OUT/v0.txt
if grep -qx "$UUID" "$JD/ec2/known_gpu_uuids.txt"; then echo "V0 FAIL: GPU rented before ($UUID), stopping" | tee -a $OUT/v0.txt; exit 8; fi
[ "${USEDGB:-999}" -le 48 ] || { echo "V0 FAIL: ${USEDGB} GB of host memory in use (> 48), stopping" | tee -a $OUT/v0.txt; exit 7; }
echo "V0 identity and memory ok" | tee -a $OUT/v0.txt
. "$JD/ec2/setup.sh"
. "$J/prof.sh"
export BASE; export -f build_tree platform getmodel
export PROF_KEEP_TAIL="G14 G32"
PATCH=$J/llama.cpp-expert-cache-4da6337-oracle5.patch
T0=$(date +%s); BUDGET=$(( 75 * 60 )); RESERVE=300
ORDER_SEED=$(( $(date +%s) % 100000 )); echo "order seed base $ORDER_SEED" | tee $OUT/order_seed.txt
NSEQ=20
secs() { case $1 in *h) echo $(( ${1%h} * 3600 ));; *m) echo $(( ${1%m} * 60 ));; *s) echo ${1%s};; *) echo $1;; esac; }
left() { echo $(( T0 + BUDGET - RESERVE - $(date +%s) )); }
R() {  # timeout cmd...: bounded by the step's timeout and by the job's deadline; past the deadline the step is skipped
  local t=$(secs $1); shift; local l=$(left)
  if [ "$l" -lt 120 ]; then echo "SKIPPED (deadline): $*" | tee -a $OUT/skipped.txt; return 124; fi
  [ "$t" -gt "$l" ] && t=$l
  timeout -k 30 "$t" "$@"
}
# --- V0c: the class, the card, the disk (before any download)
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket|NUMA node\(s\)" | tee $OUT/cpu.txt
NUMA=$(lscpu | sed -n 's/^NUMA node(s): *\([0-9]*\).*/\1/p'); CORES=$(usable_cores)
echo "usable physical cores $CORES (nproc $(nproc)); NUMA nodes ${NUMA:-?}" | tee $OUT/cores.txt
[ "${NUMA:-9}" -eq 1 ] && [ "${CORES:-99}" -le 32 ] || { echo "V0c FAIL: ${NUMA} NUMA nodes, ${CORES} usable cores: not desktop-class, stopping" | tee -a $OUT/v0.txt; exit 9; }
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current,driver_version,power.limit --format=csv > $OUT/gpu.csv 2>&1
MEMCLK=$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "memory clock max ${MEMCLK} MHz" | tee $OUT/gate.txt
nvcc -O3 -arch=sm_$SM "$J/bw.cu" -o $WORK/bw0 && $WORK/bw0 > $OUT/bw_gate.txt 2>&1
DR=$(sed -n 's/device read 1 GiB: *\([0-9]*\).*/\1/p' $OUT/bw_gate.txt)
echo "device read ${DR} GB/s" | tee -a $OUT/gate.txt
[ "${DR:-0}" -ge 1500 ] || { echo "V0c FAIL: GPU reads below 1500 GB/s: throttled card, stopping" | tee -a $OUT/v0.txt; exit 5; }
RAMGB=$(awk '/Mem:/{print $2}' $OUT/free.txt); echo "host RAM ${RAMGB} GB, in use ${USEDGB} GB" | tee -a $OUT/gate.txt
DISKGB=$(df -BG --output=avail $WORK | tail -1 | tr -dc 0-9); echo "disk free ${DISKGB} GB" | tee -a $OUT/gate.txt
[ "${DISKGB:-0}" -ge 100 ] || { echo "V0c FAIL: not enough disk, stopping" | tee -a $OUT/v0.txt; exit 6; }
echo "V0c class, card and disk ok" | tee -a $OUT/v0.txt
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip >> $OUT/apt.txt 2>&1
python3 -m pip install -q --break-system-packages huggingface_hub hf_xet numpy > $OUT/pip.txt 2>&1
# --- the download first; the build and the probe run meanwhile
mkdir -p $M
( t0=$(date +%s)
  python3 -c "import sys; from huggingface_hub import hf_hub_download as d; d('ggml-org/gpt-oss-120b-GGUF', 'gpt-oss-120b-MXFP4.gguf', local_dir=sys.argv[1])" "$M"
  echo "gguf hf download rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $M/gpt-oss-120b-MXFP4.gguf
  if [ ! -s $M/gpt-oss-120b-MXFP4.gguf ]; then
    rm -f $M/gpt-oss-120b-MXFP4.gguf
    for i in 1 2 3; do getmodel ggml-org/gpt-oss-120b-GGUF gpt-oss-120b-MXFP4.gguf; [ -f $M/gpt-oss-120b-MXFP4.gguf ] && break; done
  fi
) > $OUT/dl_gguf.txt 2>&1 &
DLG=$!
install_nsys &
NSP=$!
build_tree $WORK/lc-ec "$PATCH" llama-ec-bench
wait $NSP; install_nsys
EC_BIN=$WORK/lc-ec/build/bin/llama-ec-bench; ls -la $EC_BIN || { echo "no ec-bench binary"; exit 3; }
sha256sum $PATCH $J/known_gpu_uuids.txt | tee $OUT/patch_sha.txt; echo "base $BASE" >> $OUT/patch_sha.txt
# --- the host (idle): platform, our probe, the fetch table
platform
H=$(( CORES > 4 ? CORES - 2 : 2 ))
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
grep -q "concurrent t=" $OUT/concur.txt || { echo "concur incomplete, retrying"; R 15m $WORK/concur > $OUT/concur.txt 2>&1; }
LAWG=$(python3 $J/fetch_table.py $OUT/concur.txt $H 13253760 4 $OUT/fetch_table_law_gptoss.json 37)
echo "law table ($(date -u +%FT%TZ)): gpt-oss $LAWG" | tee $OUT/tables.txt
[ -n "$LAWG" ] || { echo "no law table"; exit 3; }
echo "build and probe done, waiting for the gpt-oss download ($(( $(date +%s) - T0 )) s since launch)"
wait $DLG
cat $OUT/dl_gguf.txt
F=$M/gpt-oss-120b-MXFP4.gguf
[ -s "$F" ] || { echo "GGUF MISSING"; exit 3; }
MB="mailbox=1:tdec=1:helpers=$H"
run_round() {  # C round: one process, the deployed cache only
  local C=$1 rd=$2
  local common="--corpus $CG -t $CORES --n-seq $NSEQ --n-prefill 1024 --n-decode 256 --no-mmap --host-experts"
  local cfg="slots=$C:policy=dfa:$MB:fetch=$LAWG:paced=0:overlap=1:oracle_w=0:kappa=1:stats=$OUT/st_g_C${C}_r${rd}_base.json"
  echo "cell g C$C round $rd configs: $cfg" >> $OUT/configs.txt
  local t0=$(date +%s)
  R 30m $EC_BIN -m $F $common --order-seed $(( ORDER_SEED + 100 * rd + C )) --ec "$cfg" > $OUT/ec_g_C${C}_r${rd}.jsonl 2> $OUT/ec_g_C${C}_r${rd}.err
  local rc=$?
  echo "cell g C$C round $rd rc=$rc in $(( $(date +%s) - t0 )) s"; echo "ec_g_C${C}_r${rd} $rc $(( $(date +%s) - t0 ))" >> $OUT/runs.txt
  tail -n 30 $OUT/ec_g_C${C}_r${rd}.err > $OUT/ec_g_C${C}_r${rd}.err.tail; rm -f $OUT/ec_g_C${C}_r${rd}.err
}
vcheck() {  # V1: the deployed cache's loss in every round so far; V2: its spread over the C = 14 rounds
  python3 - $OUT "$@" <<'PY'
import glob, json, os, sys
out, what = sys.argv[1], sys.argv[2]
def rows(f):
    return [json.loads(l) for l in open(f) if l.strip()]
msg, bad = [], False
if what == "V1":
    for f in sorted(glob.glob(f"{out}/ec_g_C*_r*.jsonl")):
        b = rows(f)
        if not b:
            msg.append(f"V1 {os.path.basename(f)}: no rows"); bad = True; continue
        nll = sum(r["nll_sum"] / r["nll_n"] for r in b) / len(b)
        ok = abs(nll / 0.190 - 1) <= 0.02
        bad |= not ok
        msg.append(f"V1 {os.path.basename(f)} loss {nll:.4f} vs 0.190: {'ok' if ok else 'FAIL'}")
else:
    t = []
    for rd in (1, 2):
        b = rows(f"{out}/ec_g_C14_r{rd}.jsonl")
        t.append(sum(r["decode_ms"] / r["n_decode"] for r in b) / len(b) if b else float("nan"))
    v = max(t) / min(t) - 1
    bad = not (v <= 0.02)
    msg.append(f"V2 base ms/token {t[0]:.3f} {t[1]:.3f}: spread {100 * v:.2f}%: {'FAIL' if bad else 'ok'}")
print("\n".join(msg))
sys.exit(9 if bad else 0)
PY
}
CG=$J/aime25_owntext_gptoss_084.jsonl; wc -l $CG
for C in 14 32; do
  prof_run G$C $EC_BIN -m $F --corpus $CG -t $CORES --n-prefill 1024 --n-decode 24 --no-mmap --host-experts --seqs 1,0 \
    --ec "slots=$C:policy=dfa:$MB:kappa=1:fetch=$LAWG:paced=0:overlap=1:oracle_w=0:seqs=1/0"
done
python3 - $OUT G14 G32 > $OUT/g_prof.json <<'PY'
import json, sys
out = sys.argv[1]; r = {}
for lab in sys.argv[2:]:
    try:
        d = json.load(open(f"{out}/prof_{lab}.json"))["median_ms"]
        r[lab] = dict(G_prof_ms=sum(v for k, v in d.items() if k.startswith("cat_") and k not in ("cat_ec_wait", "cat_ec_copy")),
                      wall_ms=d["wall"], wait_ms=d.get("cat_ec_wait"), copy_ms=d.get("cat_ec_copy"))
    except Exception as e:
        r[lab] = dict(error=str(e))
print(json.dumps(r))
PY
echo "G_prof written $(date -u +%FT%TZ), before any timed run:"; cat $OUT/g_prof.json
FAILED=""
run_round 14 1
vcheck V1 | tee -a $OUT/validity.txt; [ ${PIPESTATUS[0]} -eq 0 ] || { echo "V1 failed: stopping" | tee -a $OUT/validity.txt; FAILED=V1; }
if [ -z "$FAILED" ]; then
  run_round 14 2
  vcheck V2 | tee -a $OUT/validity.txt; [ ${PIPESTATUS[0]} -eq 0 ] || { echo "V2 failed: stopping" | tee -a $OUT/validity.txt; FAILED=V2; }
fi
if [ -z "$FAILED" ]; then
  run_round 32 1; run_round 32 2
  vcheck V1 | tee -a $OUT/validity.txt
fi
rm -f $F
python3 - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, os, sys
out = sys.argv[1]; line = []
for f in sorted(os.listdir(out)):
    if f.startswith("ec_") and f.endswith(".jsonl"):
        rows = [json.loads(l) for l in open(f"{out}/{f}") if l.strip()]
        if rows:
            ms = sum(r["decode_ms"] / r["n_decode"] for r in rows) / len(rows)
            s = json.load(open(f"{out}/{f.replace('ec_', 'st_').replace('.jsonl', '_base.json')}"))
            n = max(1, s.get("steps", 1))
            print(f"{f:22s} {1000 / ms:7.2f} tok/s {ms:6.2f} ms misses {s['misses'] / n:.2f} admits {s['admits'] / n:.2f}")
            line.append(f"{f[3:-6]} {1000 / ms:.1f}")
open(f"{out}/oneline.txt", "w").write("108 " + " ".join(line) + "\n")
PY
ONE=$(cat $OUT/oneline.txt)
python3 $J/manifest.py "$OUT" 108_smallhosts@vast "$T0" "$ONE" law_gptoss=$LAWG device_read_gbs=$DR memory_clock_max_mhz=$MEMCLK
cat $OUT/validity.txt 2>/dev/null
echo "SUMMARY: $ONE"
