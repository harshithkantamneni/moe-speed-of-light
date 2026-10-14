#!/bin/bash
# Job 119: the fetches-off test again, out of sample (reviews 35a/b and 36a/b). Jobs 114 and 115 measured the 2 x 2
# (which experts are cached x how each admission is read) with the engine's in-step fetches switched off on five valid
# RTX 5090 machines of two kinds (two Ryzen 9 5950Xs from one provider, three Intel machines), and job 115's pooled
# threshold was set after job 114's three machines had run; job 118 (a third CPU family) got no valid machine. This job
# repeats job 115's protocol (jobs/115_moremachines@vast.sh; engine patch oracle5, gpt-oss-120b MXFP4, the job 084
# own-text corpus, 20 AIME-25 problems x 256 teacher-forced steps, configurations base base0 foa foa0 bypass bypass0
# fetch both3p in one process per round in a seeded shuffled order) on new machines of any consumer CPU family, with
# every threshold below fixed before any of its machines starts, and adds:
#   (a) rounds: C = 14 three times (job 115: twice) and C = 32 twice (once), so that every configuration's variation
#       between rounds is measured on every machine and enters its intervals (rule below);
#   (b) a manipulation of the link-to-CPU ratio on the same machine and the same PCIe link: the four fetches-off arms
#       (base0 bypass0 foa0 fetch) once more at C = 14 with half the CPU helper threads (H2 = H / 2, at least 2), which
#       lowers the CPU's read rate B_c and so raises B_p / B_c. Process "h"; its ratio is the first probe's B_p over its
#       CPU reading at H2 threads (jobs/ec2/fetch_table.py at H2);
#   (c) a second model: Qwen3-30B-A3B BF16 (converted on the machine from Qwen/Qwen3-30B-A3B, as job 099) at C = 16
#       (12.5%), the same eight configurations, two rounds, its own routing (lookahead, untimed) and its own fetch table;
#   (d) Nsight profiles of the deployed cache before any timed run (gpt-oss C = 14 and 32, Qwen3 C = 16): this
#       machine's own T_GPU (jobs/ec2/prof.sh, as jobs 110-112).
# Both models download, and Qwen3 is converted, before any timed run (no download or conversion runs while anything is
# timed). The first probe runs while the models download; three more probe runs follow on the idle machine (job 115).
# Definitions as jobs 114/115 (scripts/job114.py on main): X/Y is Y's mean time per token over the problems / X's (a
# speed ratio), geometric mean over rounds; the gap is base - Eq. (1) at the first probe's rate (R* of the first 20
# problems); a share is the time a change removes over the gap; MIN's set alone is base0 - bypass0; the interaction is
# bypass0 + foa0 - base0 - fetch (ms per token); with half the helpers the same with base0h etc. and the fetches-off gap
# base0h - Eq. (1). Earlier data (jobs 114/115's five machines, 11%): bypass0/base0 1.09-1.16; bypass0/bypass 1.10-1.18
# where the fetch table is not all zero (1.00 where it is); MIN's set alone 13-23% of the gap; the interaction 21% on
# the two 5950Xs (ratio 0.78-0.79, PCIe 4.0 at 26 GB/s), -3% to +3% on the three Intel machines (0.54-0.57, 34-49 GB/s);
# job 114's invalid host e (Ryzen 9 9950X3D, ratio 0.87, PCIe 5.0) -2%. Qwen3 12.5% with the fetches on: job 096's O4
# read bypass/base 0.995.
# Predictions, committed before launch, on every valid machine unless stated (11%: gpt-oss C = 14; 25%: C = 32):
#   K1. counters: fetch's misses and in-step fetches per token within 1% of jobs 109-118's (39.7 and 19.8 at 11%, 16.4
#       and 9.7 at 25%); base0 and bypass0 make no in-step fetch, and their misses and admissions per token are within
#       1% of job 114's at 11% (base0 62.1 and 8.2, bypass0 43.5 and 20.4);
#   K2. where the first probe's fetch table is not all zero: bypass0/bypass at least 1.05 at 11% (the in-step fetches
#       evict MIN's set);
#   K3. bypass0/base0 at least 1.07 at 11% and at least 1.20 at 25% (MIN's admissions pay with the fetches off);
#   K4. foa0/base0 within [0.98, 1.10] at 11%;
#   K5. pooled over this job's valid machines only (none of jobs 114/115's): MIN's set alone closes a mean share of the
#       gap at 11% of at least 0.10, with its 95% t-interval over machines above zero (inconclusive with fewer than three
#       valid machines);
#   K6. the link (job 118's H5, never tested): on machines whose first probe reads the link at B_p >= 40 GB/s, the
#       interaction's share of the gap at 11% (full helpers) lies within [-0.05, 0.10], whatever the ratio;
#   K7. the manipulation: with half the helpers the probe's ratio is at least 1.3x the full-helper ratio, and the
#       interaction's share of the fetches-off gap rises by at least 0.05 (I0h - I0 >= 0.05, I0 = interaction / (base0 -
#       Eq. (1)), I0h the same at H2) on every valid machine: slower CPU reads make reading once worth more;
#   K8. Qwen3 12.5% (C = 16): bypass0/base0 at least 1.05; where Qwen3's fetch table is not all zero, bypass/base at most
#       1.03 and bypass0/bypass at least 1.03 (the same eviction on a second model);
#   K9. variation between rounds: at 11%, over this job's valid machines and the seven configurations other than base,
#       the spread of a configuration's ratio over base between its three rounds (largest / smallest - 1) is at most
#       3% for at least 80% of the machine-configurations.
# Rule for intervals (all clauses but K1 and K9): a machine's ratio is the geometric mean over its rounds; its 95%
# interval combines the problems' bootstrap interval with the configuration's own variation between rounds on that
# machine (half-width sqrt(hw_problems^2 + (t_{0.975, n-1} s_rounds / sqrt(n))^2), n rounds; one round: problems only).
# A clause held when that interval lies on the predicted side (pooled clauses: the t-interval over machines), held
# (point) for counters, failed otherwise. Analysis: scripts/job119.py on main, written after the runs to these
# definitions.
# VALIDITY: V0 the GPU's UUID in no earlier launch (jobs/ec2/known_gpu_uuids.txt, now with job 118's), at most 48 GB of
# host memory in use; V0c one NUMA node, at most 32 usable physical cores, "5090" in the name, device reads >= 1500
# GB/s, at least 90 GB of host memory, at least 210 GB of free disk; VR the first probe's ratio B_p / B_c at least 0.5;
# V1 base's mean loss within 2% of 0.190 (gpt-oss) and of 0.0855 (Qwen3, job 099's three RTX 5090 hosts); V2 base's
# three rounds at 11% within 2% of each other (largest / smallest - 1); V3 every configuration's loss within 1% of base's
# in its process (reported; a configuration failing it is left out there). A host is valid if it passes V0, V0c, VR, V1
# (gpt-oss) and V2; its Qwen3 clauses need Qwen3's V1 too. Every host that starts is reported.
# Hosts: single-GPU, whole-machine (gpu_frac 1) on-demand RTX 5090 offers with at least 96 GB of host memory and 240 GB
# of disk, at most $1.25/h, at least 300 Mb/s down, whose listed CPU is a consumer desktop part (Ryzen 3/5/7/9 that is
# not a Threadripper, Core i3/i5/i7/i9, Core Ultra), not in the rental ledger and not from Vast host 716820, taken in
# order of expected cost, $/h x (3.0 h + the hours to download 124 GB at the listed speed), from a listing committed
# and pushed before each launch (jobs/ec2/list119.py). At most two valid machines per Vast host (provider). Up to three
# run at once. A host that fails a gate or V1/V2 is reported and replaced by the next. The job stops starting hosts when
# eight are valid, fourteen have started, its spend in the rental ledger reaches $35, or 72 hours after this header was
# pushed, whichever comes first. Budget per host: 270 minutes (the Qwen3 part is skipped if the deadline nears; recorded).
# Amendment (committed before host b of either job): the listing rule's host memory is at least 90 GB, the machine's own
# gate (96 GB listings report 95-96 GB); nothing else changes.
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
SMOKE=${SMOKE:-0}
CARD=5090
MREPO=ggml-org/gpt-oss-120b-GGUF; MFILE=gpt-oss-120b-MXFP4.gguf; CA=14; CB=32; CQ16=16; NSEQ=20; NDEC=256
BMIN=270; NA="base base0 foa foa0 bypass bypass0 fetch both3p"; NH="base0 bypass0 foa0 fetch"; PROF=1
V1REF=0.190; V1QREF=0.0855; DRMIN=1500; VRMIN=0.5; RAMMIN=90; DISKMIN=210
KA=3; KB=2
echo "CARD=$CARD SMOKE=$SMOKE model $MFILE budgets C=$CA,$CB Qwen3 C=$CQ16 problems $NSEQ steps $NDEC A=[$NA] H=[$NH]" | tee $OUT/mode.txt
JD="$(cd "$(dirname "$0")" && pwd)"
gate_fail() {  # exit-code message: a failed gate stops the host
  local code=$1; shift
  echo "$*" | tee -a $OUT/v0.txt
  if [ "$SMOKE" = 1 ]; then echo "(smoke run: recorded, not enforced)" | tee -a $OUT/v0.txt; return 0; fi
  exit $code
}
# --- V0, before anything is installed or downloaded
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1 | tr -d ' ')
free -g > $OUT/free.txt; USEDGB=$(awk '/Mem:/{print $3}' $OUT/free.txt)
echo "GPU UUID $UUID; host memory in use ${USEDGB} GB" | tee $OUT/v0.txt
if grep -qx "$UUID" "$JD/ec2/known_gpu_uuids.txt"; then
  gate_fail 8 "V0 FAIL: GPU rented before ($UUID), stopping"
else
  echo "V0: a GPU in no earlier launch" | tee -a $OUT/v0.txt
fi
[ "${USEDGB:-999}" -le 48 ] || gate_fail 7 "V0 FAIL: ${USEDGB} GB of host memory in use (> 48), stopping"
echo "V0 checked" | tee -a $OUT/v0.txt
. "$JD/ec2/setup.sh"
. "$J/prof.sh"
export BASE; export -f build_tree platform getmodel
export PROF_KEEP_TAIL="G$CA G$CB Q$CQ16"
PATCH=$J/llama.cpp-expert-cache-4da6337-oracle5.patch
T0=$(date +%s); BUDGET=$(( BMIN * 60 )); RESERVE=300
ORDER_SEED=$(( $(date +%s) % 100000 )); echo "order seed base $ORDER_SEED" | tee $OUT/order_seed.txt
secs() { case $1 in *h) echo $(( ${1%h} * 3600 ));; *m) echo $(( ${1%m} * 60 ));; *s) echo ${1%s};; *) echo $1;; esac; }
left() { echo $(( T0 + BUDGET - RESERVE - $(date +%s) )); }
R() {  # timeout cmd...: bounded by the step's timeout and by the job's deadline; past the deadline the step is skipped
  local t=$(secs $1); shift; local l=$(left)
  if [ "$l" -lt 120 ]; then echo "SKIPPED (deadline): $*" | tee -a $OUT/skipped.txt; return 124; fi
  [ "$t" -gt "$l" ] && t=$l
  timeout -k 30 "$t" "$@"
}
# --- V0c: the class, the card, the memory, the disk (before any download)
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket|NUMA node\(s\)" | tee $OUT/cpu.txt
NUMA=$(lscpu | sed -n 's/^NUMA node(s): *\([0-9]*\).*/\1/p'); CORES=$(usable_cores)
echo "usable physical cores $CORES (nproc $(nproc)); NUMA nodes ${NUMA:-?}" | tee $OUT/cores.txt
[ "${NUMA:-9}" -eq 1 ] && [ "${CORES:-99}" -le 32 ] || gate_fail 9 "V0c FAIL: ${NUMA} NUMA nodes, ${CORES} usable cores: not desktop-class, stopping"
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current,driver_version,power.limit --format=csv > $OUT/gpu.csv 2>&1
MEMCLK=$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "memory clock max ${MEMCLK} MHz" | tee $OUT/gate.txt
nvcc -O3 -arch=sm_$SM "$J/bw.cu" -o $WORK/bw0 && $WORK/bw0 > $OUT/bw_gate.txt 2>&1
DR=$(sed -n 's/device read 1 GiB: *\([0-9]*\).*/\1/p' $OUT/bw_gate.txt)
echo "device read ${DR} GB/s" | tee -a $OUT/gate.txt
GNAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); GMEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "card ${GNAME}, ${GMEM} MiB" | tee -a $OUT/gate.txt
case "$GNAME" in *$CARD*) ;; *) gate_fail 4 "V0c FAIL: not an RTX $CARD (${GNAME}), stopping";; esac
[ "${DR:-0}" -ge $DRMIN ] || gate_fail 5 "V0c FAIL: GPU reads below $DRMIN GB/s: throttled card, stopping"
RAMGB=$(awk '/Mem:/{print $2}' $OUT/free.txt); echo "host RAM ${RAMGB} GB, in use ${USEDGB} GB" | tee -a $OUT/gate.txt
[ "${RAMGB:-0}" -ge $RAMMIN ] || gate_fail 6 "V0c FAIL: ${RAMGB} GB of host memory (< $RAMMIN), stopping"
DISKGB=$(df -BG --output=avail $WORK | tail -1 | tr -dc 0-9); echo "disk free ${DISKGB} GB" | tee -a $OUT/gate.txt
[ "${DISKGB:-0}" -ge $DISKMIN ] || gate_fail 6 "V0c FAIL: ${DISKGB} GB of free disk (< $DISKMIN), stopping"
echo "V0c checked" | tee -a $OUT/v0.txt
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip python3-venv >> $OUT/apt.txt 2>&1
python3 -m pip install -q --break-system-packages uv huggingface_hub hf_xet numpy scipy > $OUT/pip.txt 2>&1
# --- both downloads first, and the converter's environment; the build and the probe run meanwhile
mkdir -p $M
( t0=$(date +%s)
  python3 -c "import sys; from huggingface_hub import hf_hub_download as d; d(sys.argv[2], sys.argv[3], local_dir=sys.argv[1])" "$M" "$MREPO" "$MFILE"
  echo "gguf hf download rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $M/$MFILE
  if [ ! -s $M/$MFILE ]; then
    rm -f $M/$MFILE
    for i in 1 2 3; do getmodel $MREPO $MFILE; [ -f $M/$MFILE ] && break; done
  fi
) > $OUT/dl_gguf.txt 2>&1 &
DLG=$!
HFQ=$M/Qwen3-30B-A3B
( t0=$(date +%s); python3 -c "import sys; from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-30B-A3B', local_dir=sys.argv[1])" "$HFQ"
  echo "qwen3 hf download rc=$? in $(( $(date +%s) - t0 )) s"; du -sh "$HFQ" ) > $OUT/dl_qwen3.txt 2>&1 &
DLQ=$!
VE=$WORK/venv
( uv venv -q $VE && . $VE/bin/activate && uv pip install -q torch --index-url https://download.pytorch.org/whl/cpu &&
  uv pip install -q numpy transformers sentencepiece safetensors protobuf ) > $OUT/venv_install.txt 2>&1 &
VEP=$!
if [ "$PROF" = 1 ]; then install_nsys & NSP=$!; fi
build_tree $WORK/lc-ec "$PATCH" llama-ec-bench
if [ "$PROF" = 1 ]; then wait $NSP; install_nsys; fi
EC_BIN=$WORK/lc-ec/build/bin/llama-ec-bench; ls -la $EC_BIN || { echo "no ec-bench binary"; exit 3; }
sha256sum $PATCH $J/known_gpu_uuids.txt $J/fetch_table.py | tee $OUT/patch_sha.txt; echo "base $BASE" >> $OUT/patch_sha.txt
# --- the host (downloading, otherwise idle): platform, our probe, the fetch tables, the ratios
platform
H=$(( CORES > 4 ? CORES - 2 : 2 )); H2=$(( H / 2 )); [ $H2 -lt 2 ] && H2=2
echo "helpers H=$H H2=$H2" | tee -a $OUT/gate.txt
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && R 15m $WORK/concur > $OUT/concur.txt 2>&1
grep -q "concurrent t=" $OUT/concur.txt || { echo "concur incomplete, retrying"; R 15m $WORK/concur > $OUT/concur.txt 2>&1; }
echo "probe 1 ended $(date -u +%FT%TZ); downloads running: $(kill -0 $DLG 2>/dev/null && echo yes || echo no) $(kill -0 $DLQ 2>/dev/null && echo yes || echo no)" | tee $OUT/probe1_when.txt
LAWG=$(python3 $J/fetch_table.py $OUT/concur.txt $H 13253760 4 $OUT/fetch_table_law_gptoss.json 37)
LAWG2=$(python3 $J/fetch_table.py $OUT/concur.txt $H2 13253760 4 $OUT/fetch_table_law_gptoss_h2.json 37)
LAWQ=$(python3 $J/fetch_table.py $OUT/concur.txt $H 9437184 8 $OUT/fetch_table_law_qwen3.json 48)
echo "law tables ($(date -u +%FT%TZ)): gpt-oss $LAWG; gpt-oss at H2 $LAWG2; qwen3 $LAWQ" | tee $OUT/tables.txt
[ -n "$LAWG" ] && [ -n "$LAWG2" ] && [ -n "$LAWQ" ] || { echo "no law table"; exit 3; }
ratio_of() { python3 -c "import json, sys; d = json.load(open(sys.argv[1])); print(f'{d[\"B_p\"] / d[\"B_c\"]:.4f}')" "$1"; }
RATIO=$(ratio_of $OUT/fetch_table_law_gptoss.json); RATIO2=$(ratio_of $OUT/fetch_table_law_gptoss_h2.json)
echo "VR: B_p / B_c = ${RATIO:-?} (H2: ${RATIO2:-?})" | tee -a $OUT/gate.txt
if ! python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)" "${RATIO:-0}" $VRMIN; then
  if [ "$SMOKE" != 1 ]; then pkill -P $DLG; kill $DLG; pkill -P $DLQ; kill $DLQ; fi
  gate_fail 10 "VR FAIL: link/CPU ratio ${RATIO} below $VRMIN, stopping"
fi
echo "VR checked" | tee -a $OUT/v0.txt
echo "build and probe done, waiting for the downloads ($(( $(date +%s) - T0 )) s since launch)"
wait $DLG
cat $OUT/dl_gguf.txt
F=$M/$MFILE
[ -s "$F" ] || { echo "GGUF MISSING"; exit 3; }
# --- Qwen3: wait for its download and the converter, convert to BF16 GGUF before anything is timed
wait $DLQ $VEP; cat $OUT/dl_qwen3.txt; tail -2 $OUT/venv_install.txt
GQ=$M/Qwen3-30B-A3B-BF16.gguf
( t0=$(date +%s); PYTHONPATH=$WORK/lc-ec/gguf-py timeout 40m $VE/bin/python $WORK/lc-ec/convert_hf_to_gguf.py "$HFQ" --outtype bf16 --outfile $GQ
  echo "convert rc=$? in $(( $(date +%s) - t0 )) s"; ls -la $GQ ) > $OUT/convert_qwen3.txt 2>&1
tail -2 $OUT/convert_qwen3.txt; rm -rf "$HFQ"
df -h $WORK | tail -1 > $OUT/df.txt
# --- PROBE: three more runs on the idle machine (recorded; the fetch tables and the gate stay the first probe's)
for i in 1 2 3; do
  t0=$(date +%s)
  R 15m $WORK/concur > $OUT/concur_idle$i.txt 2>&1
  echo "idle probe $i rc=$? in $(( $(date +%s) - t0 )) s ($(date -u +%FT%TZ))" | tee -a $OUT/probe_idle.txt
done
python3 - $OUT > $OUT/probe_compare.txt <<'PY'
import re, sys
out = sys.argv[1]
def best(f):
    t = open(f).read()
    v = [float(x) for x in re.findall(r"cpu_read_gbs t=\d+ ([\d.]+)", t)]
    v += [float(x) for x in re.findall(r"pcie_\w+_gbs [^\n]*? ([\d.]+)\n", t)]
    v += [float(x) for x in re.findall(r"sum ([\d.]+)", t)]
    return max(v) if v else float("nan")
b1 = best(f"{out}/concur.txt")
bi = [best(f"{out}/concur_idle{i}.txt") for i in (1, 2, 3)]
print(f"first probe highest {b1:.2f} GB/s; idle runs {' '.join(f'{x:.2f}' for x in bi)}; idle best / first {max(bi) / b1:.4f}; idle spread {max(bi) / min(bi):.4f}")
PY
cat $OUT/probe_compare.txt
MB="mailbox=1:tdec=1:helpers=$H"; MBH="mailbox=1:tdec=1:helpers=$H2"
PACED="paced=1:chunk=16777216:max_pending=64"
CG=$J/aime25_owntext_gptoss_084.jsonl; CQF=$J/aime25_owntext_qwen3_084b.jsonl; wc -l $CG $CQF
lookahead() {  # tag gguf corpus ncmoe: this machine's routing of the teacher-forced text (untimed)
  local tag=$1 g=$2 corpus=$3 n=$4 t0=$(date +%s)
  R 30m $EC_BIN -m $g --corpus $corpus -t $CORES --n-seq $NSEQ --n-prefill 1024 --n-decode $NDEC --ncmoe $n --lookahead $OUT/la_$tag.bin > $OUT/la_$tag.jsonl 2> $OUT/la_$tag.err
  echo "lookahead $tag rc=$? in $(( $(date +%s) - t0 )) s: $(ls -la $OUT/la_$tag.bin | awk '{print $5}') bytes"; cat $OUT/la_$tag.bin.json
  tail -n 5 $OUT/la_$tag.err > $OUT/la_$tag.err.tail; rm -f $OUT/la_$tag.err
}
profiles() {  # model-tag gguf corpus law C...: Nsight profiles of the deployed cache, one problem, 24 decode tokens
  local m=$1 g=$2 corpus=$3 law=$4; shift 4
  [ "$PROF" = 1 ] || return 0
  local labs="" C
  for C in "$@"; do
    prof_run $m$C $EC_BIN -m $g --corpus $corpus -t $CORES --n-prefill 1024 --n-decode 24 --no-mmap --host-experts --seqs 1,0 \
      --ec "slots=$C:policy=dfa:$MB:kappa=1:fetch=$law:paced=0:overlap=1:oracle_w=0:seqs=1/0"
    labs="$labs $m$C"
  done
  python3 - $OUT $labs >> $OUT/g_prof.jsonl <<'PY'
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
  echo "profiles written $(date -u +%FT%TZ), before any timed run of this model:"; tail -1 $OUT/g_prof.jsonl
}
run_round() {  # model-tag C round process "names": one process, the named configurations in a shuffled order
  local m=$1 C=$2 rd=$3 pr=$4 names=$5
  local g corpus la law mb
  case $m in
    g) g=$F; corpus=$CG; la=$OUT/la_g.bin; law=$LAWG; mb=$MB ;;
    h) g=$F; corpus=$CG; la=$OUT/la_g.bin; law=$LAWG2; mb=$MBH ;;
    q) g=$GQ; corpus=$CQF; la=$OUT/la_q.bin; law=$LAWQ; mb=$MB ;;
  esac
  local common="--corpus $corpus -t $CORES --n-seq $NSEQ --n-prefill 1024 --n-decode $NDEC --no-mmap --host-experts"
  local base="slots=$C:policy=dfa:$mb:fetch=$law:paced=0:overlap=1:oracle_w=0"
  local orc="slots=$C:policy=oracle:oracle=$la:$mb:kappa=1:fetch=$law:overlap=1:oracle_w=0"
  local base0="slots=$C:policy=dfa:$mb:fetch=0:paced=0:overlap=1:oracle_w=0"
  local orc0="slots=$C:policy=oracle:oracle=$la:$mb:kappa=1:fetch=0:overlap=1:oracle_w=0"
  local tag="${m}_C${C}_r${rd}${pr}"
  local cfg="" n c
  for n in $names; do
    case $n in
      base)    c="$base:kappa=1" ;;
      base0)   c="$base0:kappa=1" ;;
      foa0)    c="$base0:kappa=1:fetch_on_admit=1" ;;
      bypass0) c="$orc0:oracle_bypass=1:paced=0" ;;
      foa)     c="$base:kappa=1:fetch_on_admit=1" ;;
      bypass)  c="$orc:oracle_bypass=1:paced=0" ;;
      fetch)   c="$orc:oracle_bypass=1:oracle_fetch=1:paced=0" ;;
      both3p)  c="$orc:oracle_bypass=0:oracle_lead=3:oracle_fetch=1:$PACED" ;;
      *)       echo "unknown configuration $n"; return 2 ;;
    esac
    cfg="${cfg:+$cfg;}$c:stats=$OUT/st_${tag}_$n.json"
  done
  local off=0; case $m in h) off=7 ;; q) off=13 ;; esac
  local seed=$(( ORDER_SEED + 100 * rd + C + off ))
  echo "cell $tag order-seed $seed configs: $cfg" >> $OUT/configs.txt
  local t0=$(date +%s)
  R 60m $EC_BIN -m $g $common --order-seed $seed --ec "$cfg" > $OUT/ec_${tag}.jsonl 2> $OUT/ec_${tag}.err
  local rc=$?
  echo "cell $tag rc=$rc in $(( $(date +%s) - t0 )) s"; echo "ec_${tag} $rc $(( $(date +%s) - t0 ))" >> $OUT/runs.txt
  grep "\[ec-bench\] config [0-9]*/" $OUT/ec_${tag}.err | sed 's/:stats=.*st_/ -> st_/' > $OUT/ec_${tag}.order
  grep -i "oracle\|plan\|fetch table\|fetch-on-admit\|prefetch\|shuffled\|kappa\|helpers" $OUT/ec_${tag}.err | head -60 > $OUT/ec_${tag}.err.oracle
  tail -n 30 $OUT/ec_${tag}.err > $OUT/ec_${tag}.err.tail; rm -f $OUT/ec_${tag}.err
}
vcheck() {  # what model-tag reference [C rounds...]: V1 / V2 / V3 on the rounds run so far
  python3 - $OUT $SMOKE "$@" <<'PY'
import glob, json, os, sys
out, smoke, what, m, ref = sys.argv[1], sys.argv[2] == "1", sys.argv[3], sys.argv[4], float(sys.argv[5])
def by_config(f):
    by = {}
    for l in open(f):
        if l.strip():
            r = json.loads(l)
            lab = os.path.basename(r["config"].rsplit("stats=", 1)[-1]).replace(".json", "").rsplit("_", 1)[-1]
            by.setdefault(lab, []).append(r)
    return by
msg, bad = [], False
files = sorted(glob.glob(f"{out}/ec_{m}_C*_r*.jsonl"))
if what == "V1":
    for f in files:
        b = by_config(f).get("base", []) or by_config(f).get("base0", [])
        if not b:
            msg.append(f"V1 {os.path.basename(f)}: no base rows"); bad = True; continue
        nll = sum(r["nll_sum"] / r["nll_n"] for r in b) / len(b)
        ok = smoke or abs(nll / ref - 1) <= 0.02
        bad |= not ok
        msg.append(f"V1 {os.path.basename(f)} base loss {nll:.4f} vs {ref}: {'ok' if ok else 'FAIL'}" + (" (smoke: not checked)" if smoke else ""))
elif what == "V2":
    C, rounds = sys.argv[6], sys.argv[7:]
    t = []
    for rd in rounds:
        p = f"{out}/ec_{m}_C{C}_r{rd}A.jsonl"
        b = by_config(p).get("base", []) if os.path.exists(p) else []
        t.append(sum(r["decode_ms"] / r["n_decode"] for r in b) / len(b) if b else float("nan"))
    v = max(t) / min(t) - 1
    bad = not (v <= 0.02)
    msg.append(f"V2 {m} C{C} base ms/token {' '.join(f'{x:.3f}' for x in t)}: spread {100 * v:.2f}%: {'FAIL' if bad else 'ok'}")
else:  # V3: every configuration computes the same model as base in its process; reported, never stops the host
    for f in files:
        by = by_config(f)
        k0 = "base" if "base" in by else "base0"
        if k0 not in by:
            continue
        b = sum(r["nll_sum"] / r["nll_n"] for r in by[k0]) / len(by[k0])
        for k, v in sorted(by.items()):
            nll = sum(r["nll_sum"] / r["nll_n"] for r in v) / len(v)
            ok = abs(nll / b - 1) <= 0.01
            msg.append(f"V3 {os.path.basename(f)} {k}: {len(v)} problems, loss {nll:.4f} vs {k0} {b:.4f} ({100 * (nll / b - 1):+.2f}%): {'ok' if ok else 'FAIL'}")
print("\n".join(msg))
sys.exit(9 if bad else 0)
PY
}
# ---- gpt-oss-120b: routing, profiles, then the timed rounds
lookahead g $F $CG 24
[ -s $OUT/la_g.bin ] || { echo "no lookahead file: oracle runs impossible"; exit 3; }
profiles G $F $CG $LAWG $CA $CB
FAILED=""
run_round g $CA 1 A "$NA"
vcheck V1 g $V1REF | tee -a $OUT/validity.txt; [ ${PIPESTATUS[0]} -eq 0 ] || { echo "V1 failed: stopping" | tee -a $OUT/validity.txt; FAILED=V1; }
vcheck V3 g $V1REF | tee -a $OUT/validity.txt
if [ -z "$FAILED" ]; then
  run_round g $CA 2 A "$NA"
  vcheck V2 g $V1REF $CA 1 2 | tee -a $OUT/validity.txt; [ ${PIPESTATUS[0]} -eq 0 ] || { echo "V2 failed (two rounds): stopping" | tee -a $OUT/validity.txt; [ "$SMOKE" = 1 ] || FAILED=V2; }
fi
if [ -z "$FAILED" ]; then
  run_round g $CA 3 A "$NA"
  vcheck V2 g $V1REF $CA 1 2 3 | tee -a $OUT/validity.txt; [ ${PIPESTATUS[0]} -eq 0 ] || { echo "V2 failed (three rounds): stopping" | tee -a $OUT/validity.txt; [ "$SMOKE" = 1 ] || FAILED=V2; }
fi
if [ -z "$FAILED" ]; then
  run_round g $CB 1 A "$NA"
  run_round g $CB 2 A "$NA"
  vcheck V1 g $V1REF | tee -a $OUT/validity.txt
  vcheck V3 g $V1REF | tee -a $OUT/validity.txt
  run_round h $CA 1 A "$NH"
  vcheck V3 h $V1REF | tee -a $OUT/validity.txt
fi
rm -f $F $OUT/la_g.bin
# ---- Qwen3-30B-A3B BF16 at C = 16 (a valid host only)
if [ -z "$FAILED" ] && [ -s "$GQ" ]; then
  lookahead q $GQ $CQF 36
  if [ -s $OUT/la_q.bin ]; then
    profiles Q $GQ $CQF $LAWQ $CQ16
    run_round q $CQ16 1 A "$NA"
    vcheck V1 q $V1QREF | tee -a $OUT/validity.txt
    run_round q $CQ16 2 A "$NA"
    vcheck V2 q $V1QREF $CQ16 1 2 | tee -a $OUT/validity.txt
    vcheck V3 q $V1QREF | tee -a $OUT/validity.txt
  else
    echo "no Qwen3 lookahead file: Qwen3 runs skipped" | tee -a $OUT/skipped.txt
  fi
else
  echo "Qwen3 part skipped (FAILED=${FAILED:-none}; GGUF $(ls -la $GQ 2>/dev/null | awk '{print $5}'))" | tee -a $OUT/skipped.txt
fi
rm -f $GQ $OUT/la_q.bin
# ---- summary
python3 - "$OUT" <<'PY' | tee $OUT/summary.txt
import json, os, sys, statistics as st
out = sys.argv[1]; line = []
for f in sorted(os.listdir(out)):
    if not (f.startswith("ec_") and f.endswith(".jsonl")):
        continue
    by = {}
    for l in open(f"{out}/{f}"):
        if l.strip():
            r = json.loads(l)
            lab = os.path.basename(r["config"].rsplit("stats=", 1)[-1]).replace(".json", "")
            by.setdefault(lab, {})[r["seq"]] = (r["decode_ms"] / r["n_decode"], r["nll_sum"] / r["nll_n"])
    base = next((v for k, v in by.items() if k.endswith("_base")), None) or next((v for k, v in by.items() if k.endswith("_base0")), None)
    for lab, v in sorted(by.items()):
        ms = st.mean(x[0] for x in v.values()); nll = st.mean(x[1] for x in v.values())
        b = st.mean(base[s][0] for s in v if s in base) if base else float("nan")
        bn = st.mean(base[s][1] for s in v if s in base) if base else float("nan")
        stf = f"{out}/{lab}.json"
        s = json.load(open(stf)) if os.path.exists(stf) else {}
        n = max(1, s.get("steps", 1))
        print(f"{lab:26s} {1000 / ms:7.2f} tok/s {ms:6.2f} ms n={len(v)} vs base {b / ms:.3f} loss {100 * (nll / bn - 1):+.2f}% "
              f"misses {s.get('misses', 0) / n:.2f} admits {s.get('admits', 0) / n:.2f} prefetches {s.get('prefetches', 0) / n:.2f} "
              f"fetches {s.get('fetches', 0) / n:.2f} reads/tok {(s.get('misses', 0) + s.get('admits', 0) + s.get('prefetches', 0)) / n:.2f}")
        if "_r1" in lab and not lab.endswith(("_base", "_base0")):
            line.append(f"{lab.replace('st_', '')} {b / ms:.3f}")
open(f"{out}/oneline.txt", "w").write("119 " + " ".join(line) + "\n")
PY
ONE=$(cat $OUT/oneline.txt)
python3 $J/manifest.py "$OUT" "119_replication@vast" "$T0" "$ONE" card=$CARD law_gptoss=$LAWG law_gptoss_h2=$LAWG2 law_qwen3=$LAWQ device_read_gbs=$DR memory_clock_max_mhz=$MEMCLK ratio=$RATIO ratio_h2=$RATIO2 helpers=$H helpers_h2=$H2 failed=${FAILED:-none} smoke=$SMOKE
cat $OUT/skipped.txt 2>/dev/null
cat $OUT/validity.txt 2>/dev/null
echo "SUMMARY: $ONE"
