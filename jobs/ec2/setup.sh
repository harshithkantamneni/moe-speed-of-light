# Environment for the expert-cache jobs on rented containers (Vast; source it). llama.cpp 4da6337 stock and with
# jobs/ec2/llama.cpp-expert-cache-4da6337.patch, built for the local GPU; models fetched on demand from Hugging Face.
# Exports J M SM STOCK_BIN EC_BIN EC_TESTS STOCK_TESTS. No credentials are used anywhere.
J=${J:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
export J M=$WORK/models GGML_NO_BACKTRACE=1 PATH=/usr/local/cuda/bin:$PATH
BASE=4da6337767f973e2b4d0797e5b323d77d8565e4a
if ! command -v cmake >/dev/null || ! command -v ninja >/dev/null || ! command -v python3 >/dev/null || ! command -v numactl >/dev/null; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cmake ninja-build build-essential git curl \
    python3 python3-numpy numactl pciutils > $OUT/apt.txt 2>&1
fi
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
export SM NPROC=$(nproc)
platform() {
  nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current,driver_version,power.limit --format=csv > $OUT/gpu.csv 2>&1
  nvidia-smi -q > $OUT/nvidia-smi-q.txt 2>&1
  lscpu > $OUT/lscpu.txt; free -g > $OUT/free.txt; numactl -H > $OUT/numa.txt 2>&1; nproc > $OUT/nproc.txt
  cat /sys/fs/cgroup/cpu.max > $OUT/cgroup_cpu_max.txt 2>/dev/null; df -h / $WORK > $OUT/df.txt
  { nvidia-smi -pm 1; nvidia-smi -lgc $(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1); } > $OUT/clock-lock.txt 2>&1
  mkdir -p $WORK/bench && cd $WORK/bench
  curl -fsSL https://raw.githubusercontent.com/jeffhammond/STREAM/master/stream.c -o stream.c
  gcc -O3 -march=native -fopenmp -DSTREAM_ARRAY_SIZE=200000000 -DNTIMES=8 -mcmodel=medium stream.c -o stream
  for t in 1 2 4 8 $(( NPROC/2 )) $NPROC; do
    echo "== OMP_NUM_THREADS=$t"; OMP_NUM_THREADS=$t OMP_PROC_BIND=spread ./stream | grep -E "Copy|Scale|Add|Triad"
  done > $OUT/stream.txt 2>&1
  nvcc -O3 -arch=sm_$SM "$J/bw.cu" -o bw && ./bw > $OUT/bw.txt 2>&1
  cd $WORK
}
build_tree() {  # dir patch|"" targets...
  local d=$1 p=$2; shift 2
  [ -f $d/.built ] && return 0
  ( cd $WORK && { [ -d lc ] || { git init -q lc && git -C lc remote add origin https://github.com/ggml-org/llama.cpp; }; } &&
    git -C lc fetch -q --depth 1 origin $BASE && rm -rf $d && git -C lc worktree add -q -f $d FETCH_HEAD && cd $d &&
    { [ -z "$p" ] || git apply "$p"; } &&
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$SM -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_TESTS=ON > $OUT/cmake_$(basename $d).txt 2>&1 &&
    cmake --build build -j$NPROC --target "$@" > $OUT/build_$(basename $d).txt 2>&1 && touch .built )
  echo "build $(basename $d): $([ -f $d/.built ] && echo ok || echo FAILED)"; grep -E "error" $OUT/build_$(basename $d).txt | head -20
}
build_all() {
  local t0=$(date +%s)
  build_tree $WORK/lc-ec "$J/llama.cpp-expert-cache-4da6337.patch" llama-ec-bench llama-ec-cpubench test-backend-ops llama-bench
  build_tree $WORK/lc-stock "" test-backend-ops llama-bench llama-server
  echo "builds took $(( $(date +%s) - t0 )) s"
}
export EC_BIN=$WORK/lc-ec/build/bin/llama-ec-bench EC_TESTS=$WORK/lc-ec/build/bin/test-backend-ops \
       EC_CPUBENCH=$WORK/lc-ec/build/bin/llama-ec-cpubench STOCK_TESTS=$WORK/lc-stock/build/bin/test-backend-ops \
       STOCK_BENCH=$WORK/lc-stock/build/bin/llama-bench STOCK_SERVER=$WORK/lc-stock/build/bin/llama-server
getmodel() {  # repo file
  mkdir -p $M; [ -f $M/$2 ] && return 0
  local t0=$(date +%s)
  curl -fL --retry 5 --retry-delay 5 -C - -s -o $M/$2.part "https://huggingface.co/$1/resolve/main/$2" && mv $M/$2.part $M/$2
  echo "download $2: $(stat -c %s $M/$2 2>/dev/null) bytes in $(( $(date +%s) - t0 )) s"
}
