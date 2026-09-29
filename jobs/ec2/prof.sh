# Nsight Systems helpers for the profiling jobs (source after setup.sh). Every profiled run is one ec-bench process on
# one sequence; the GPU kernel trace (CUDA graphs traced per node) and the CUDA API trace are exported as gzipped CSV,
# with a per-kernel summary, so the per-token breakdown can be computed off the machine.
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
  for f in $WORK/prof_${label}_*.csv; do gzip -c "$f" > $OUT/$(basename "$f").gz; done
  ls -la $OUT/prof_${label}_* 2>/dev/null
}
