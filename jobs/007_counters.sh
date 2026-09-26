#!/bin/bash
# Rerun of 004 after the 005 fix. Hardware counters and timelines (pre-registered: prereg/PROTOCOL.md on main).
#  (a) Nsight Compute: GPU DRAM bytes read per decode token, by differencing a
#      -n 8 and a -n 4 run (cancels load, warm-up and prompt).
#  (b) Nsight Systems: GPU busy vs idle per decode token, to split the measured
#      step time into the GPU part and the CPU-expert + hand-off part.
set -x
exec 2>&1
B=$WORK/llama.cpp/build/bin/llama-bench
M=$WORK/models
declare -A F=([gpt-oss-20b-mxfp4]=gpt-oss-20b-MXFP4.gguf [qwen3-30b-a3b-q4_k_m]=Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf)
T=$(sed -n 's/threads=//p' "$(dirname "$OUT")/006_sweep/threads.txt" 2>/dev/null || true)
[ -n "$T" ] || T=$(( $(nproc) / 2 ))
NCU=$(command -v ncu || ls /usr/local/cuda*/bin/ncu 2>/dev/null | head -1)
NSYS=$(command -v nsys || ls /usr/local/cuda*/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1)
echo "ncu=$NCU nsys=$NSYS threads=$T"

for spec in "gpt-oss-20b-mxfp4 0" "gpt-oss-20b-mxfp4 12" "qwen3-30b-a3b-q4_k_m 0" "qwen3-30b-a3b-q4_k_m 24"; do
  set -- $spec; model=$1; n=$2
  for tokens in 4 8; do
    tag=${model}_ncmoe${n}_n${tokens}
    timeout 30m sudo $NCU --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum \
      --cache-control all --clock-control none --csv --page raw --log-file "$OUT/ncu_$tag.csv" \
      $B -m $M/${F[$model]} -ngl 99 -ncmoe $n -fa 1 -p 0 -n $tokens -r 1 --no-warmup -t $T -o jsonl > "$OUT/ncu_$tag.bench.jsonl"
    echo "ncu $tag rc=$?"
  done
done
python3 - "$OUT" <<'PY'
import csv, glob, json, os, re, sys
out = sys.argv[1]; res = {}
for f in sorted(glob.glob(os.path.join(out, "ncu_*_n*.csv"))):
    tag = os.path.basename(f)[4:-4]
    rows = [r for r in csv.reader(open(f)) if r]
    hdr_i = next(i for i, r in enumerate(rows) if "dram__bytes_read.sum" in r)
    hdr, units = rows[hdr_i], rows[hdr_i + 1]
    ci = hdr.index("dram__bytes_read.sum"); wi = hdr.index("dram__bytes_write.sum")
    scale = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}
    rd = sum(float(r[ci].replace(",", "")) * scale.get(units[ci], 1) for r in rows[hdr_i + 2:] if len(r) > ci and r[ci])
    wr = sum(float(r[wi].replace(",", "")) * scale.get(units[wi], 1) for r in rows[hdr_i + 2:] if len(r) > wi and r[wi])
    res[tag] = dict(read_bytes=rd, write_bytes=wr, kernels=len(rows) - hdr_i - 2)
per_tok = {}
for tag, v in res.items():
    if tag.endswith("_n8") and tag[:-3] + "_n4" in res:
        a = res[tag[:-3] + "_n4"]
        per_tok[tag[:-3]] = dict(read_bytes_per_token=(v["read_bytes"] - a["read_bytes"]) / 4,
                                 write_bytes_per_token=(v["write_bytes"] - a["write_bytes"]) / 4,
                                 kernels_per_token=(v["kernels"] - a["kernels"]) / 4)
json.dump(dict(raw=res, per_token=per_tok), open(os.path.join(out, "ncu_summary.json"), "w"), indent=1)
print(json.dumps(per_tok, indent=1))
PY

# (b) timelines: 32 decode tokens per configuration
for spec in "gpt-oss-20b-mxfp4 0" "gpt-oss-20b-mxfp4 12" "gpt-oss-20b-mxfp4 24" "qwen3-30b-a3b-q4_k_m 0" "qwen3-30b-a3b-q4_k_m 24" "qwen3-30b-a3b-q4_k_m 48"; do
  set -- $spec; model=$1; n=$2; tag=${model}_ncmoe${n}
  timeout 15m $NSYS profile -t cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none -o "nsys_$tag" --force-overwrite true \
    $B -m $M/${F[$model]} -ngl 99 -ncmoe $n -fa 1 -p 0 -n 32 -r 1 --no-warmup -t $T -o jsonl > "$OUT/nsys_$tag.bench.jsonl"
  $NSYS stats -r cuda_gpu_trace -f csv -o "nsys_$tag" "nsys_$tag.nsys-rep" > /dev/null 2>&1
  gzip -c "nsys_${tag}_cuda_gpu_trace.csv" > "$OUT/nsys_${tag}_gpu_trace.csv.gz"
done
ls -la $OUT
