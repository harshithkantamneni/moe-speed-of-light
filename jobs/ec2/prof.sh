# Nsight Systems helpers for the profiling jobs (source after setup.sh). Every profiled run is one ec-bench process on
# one sequence; the GPU kernel trace (CUDA graphs traced per node) and the CUDA API trace are exported as CSV and
# summarised per decode token on the machine (prof_summary.py), since full exports exceed the log channel.
install_nsys() {
  NSYS=$(ls /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | tail -1)
  if [ -z "$NSYS" ]; then
    local pkg
    pkg=$(apt-cache search --names-only '^nsight-systems-20[0-9][0-9]' 2>/dev/null | awk '{print $1}' | sort -V | tail -1)
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ${pkg:-nsight-systems} > $OUT/nsys_install.txt 2>&1
    NSYS=$(ls /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | tail -1)
  fi
  [ -z "$NSYS" ] && NSYS=$(command -v nsys)
  echo "nsys: ${NSYS:-MISSING} ($($NSYS --version 2>/dev/null))" | tee -a $OUT/nsys_install.txt
  export NSYS
}
prof_run() {  # label, then the command (env assignments via env ...)
  local label=$1; shift
  [ -z "$NSYS" ] && { echo "prof $label: no nsys"; return 1; }
  timeout 30m $NSYS profile -o $WORK/prof_$label --force-overwrite true --trace=cuda --cuda-graph-trace=node \
    --sample=none --cpuctxsw=none "$@" > $OUT/prof_$label.stdout 2> $OUT/prof_$label.err
  echo "prof $label rc=$?"
  # one stats call for all reports (a second call refuses to reuse the SQLite export without --force-export)
  $NSYS stats --force-export=true --report cuda_gpu_trace,cuda_api_trace,cuda_gpu_kern_sum --format csv \
    --output $WORK/prof_$label $WORK/prof_$label.nsys-rep > /dev/null 2>> $OUT/prof_$label.err
  # summarise on the machine (full exports are too large for the log channel); keep the decode-tail trace only for
  # the labels listed in PROF_KEEP_TAIL, and the full kernel summary for all
  local keep=""; case " $PROF_KEEP_TAIL " in *" $label "*) keep=--keep-tail;; esac
  python3 $J/prof_summary.py --work $WORK --label $label --out $OUT $keep >> $OUT/prof_summary.txt 2>> $OUT/prof_$label.err
  [ -f $WORK/prof_${label}_cuda_gpu_kern_sum.csv ] && gzip -c $WORK/prof_${label}_cuda_gpu_kern_sum.csv > $OUT/prof_${label}_kern_sum.csv.gz
  rm -f $WORK/prof_${label}_*.csv $WORK/prof_${label}.sqlite
  ls -la $OUT/prof_${label}* 2>/dev/null
}
# Interactive collection for servers (job 079+): the server runs under `nsys launch`, and the client starts and stops
# collection around one measured request (bs1_client --wrap / --before-cmd / --after-cmd). Which switches go to
# `launch` and which to `start` differs between nsys versions, so nsys_selftest tries the variants on a small torch
# program with a CUDA graph and exports NSYS_LOPTS / NSYS_SOPTS for the first variant whose report has kernels.
nsys_selftest() {  # python with torch
  local py=$1 v
  cat > $WORK/nsys_selftest.py <<'PY'
import time, torch
x = torch.randn(1024, 1024, device="cuda"); y = x @ x; torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): y = x @ x
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g): y = torch.relu(x @ x) + 1
t0 = time.time()
while time.time() - t0 < 25:
    g.replay(); torch.cuda.synchronize(); time.sleep(0.001)
PY
  local i=0
  for v in "--trace=cuda --cuda-graph-trace=node --trace-fork-before-exec=true|--sample=none --cpuctxsw=none" \
           "--trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --trace-fork-before-exec=true|" \
           "--trace=cuda --cuda-graph-trace=node|" "--trace=cuda|"; do
    i=$((i+1)); local lo=${v%%|*} so=${v#*|}
    rm -f $WORK/nsys_st_$i.nsys-rep
    $NSYS launch --session-new=st$i $lo $py $WORK/nsys_selftest.py > $OUT/nsys_selftest_$i.txt 2>&1 &
    local pid=$!
    sleep 8
    $NSYS start --session=st$i $so -o $WORK/nsys_st_$i --force-overwrite=true >> $OUT/nsys_selftest_$i.txt 2>&1
    sleep 4
    $NSYS stop --session=st$i >> $OUT/nsys_selftest_$i.txt 2>&1
    wait $pid
    local nk=0
    if [ -f $WORK/nsys_st_$i.nsys-rep ]; then
      $NSYS stats --force-export=true --report cuda_gpu_trace --format csv --output $WORK/nsys_st_$i $WORK/nsys_st_$i.nsys-rep > /dev/null 2>> $OUT/nsys_selftest_$i.txt
      nk=$(cat $WORK/nsys_st_${i}_cuda_gpu_trace.csv 2>/dev/null | wc -l)
    fi
    echo "nsys selftest variant $i [launch: $lo] [start: $so]: report rows $nk" | tee -a $OUT/nsys_selftest.txt
    if [ "$nk" -gt 50 ]; then export NSYS_LOPTS="$lo" NSYS_SOPTS="$so"; return 0; fi
  done
  return 1
}
prof_post() {  # label decode_steps: export and summarise $WORK/prof_<label>.nsys-rep
  local label=$1 steps=$2
  [ -f $WORK/prof_$label.nsys-rep ] || { echo "prof $label: no report"; return 1; }
  $NSYS stats --force-export=true --report cuda_gpu_trace,cuda_api_trace,cuda_gpu_kern_sum --format csv \
    --output $WORK/prof_$label $WORK/prof_$label.nsys-rep > /dev/null 2>> $OUT/prof_$label.err
  python3 $J/prof_summary.py --work $WORK --label $label --out $OUT --tokens ${PROF_TOKENS:-200} --decode-steps $steps \
    >> $OUT/prof_summary.txt 2>> $OUT/prof_$label.err
  # a 3-token kernel timeline for the labels in PROF_KEEP_TAIL (small enough for the archive)
  case " $PROF_KEEP_TAIL " in *" $label "*)
    python3 $J/prof_summary.py --work $WORK --label $label --out $OUT/tail3 --tokens 3 --decode-steps $steps --keep-tail \
      > /dev/null 2>> $OUT/prof_$label.err;; esac
  [ -f $WORK/prof_${label}_cuda_gpu_kern_sum.csv ] && gzip -c $WORK/prof_${label}_cuda_gpu_kern_sum.csv > $OUT/prof_${label}_kern_sum.csv.gz
  rm -f $WORK/prof_${label}_*.csv $WORK/prof_${label}.sqlite
  tail -1 $OUT/prof_summary.txt
}
