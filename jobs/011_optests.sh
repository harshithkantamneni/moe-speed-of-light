#!/bin/bash
# Fresh-instance setup (jobs/ec/setup.sh) and the op-level tests of the expert-cache kernel changes,
# CUDA against CPU. (Job 008 ran the filter while the cases sat in a disabled block: 0/0 ran.)
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/ec/setup.sh"
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,ecc.mode.current --format=csv | tee "$OUT/gpu.csv"
timeout 20m $EC_TESTS -b CUDA0 -o MUL_MAT_ID_EC,ADD_ID_EC > "$OUT/test_ec.txt" 2>&1; echo "rc=$?"
grep -E "tests passed|FAIL" "$OUT/test_ec.txt"; grep -c " OK" "$OUT/test_ec.txt"
timeout 30m $EC_TESTS -b CUDA0 -o MUL_MAT_ID,ADD_ID > "$OUT/test_regular.txt" 2>&1; echo "rc=$?"
grep -E "tests passed" "$OUT/test_regular.txt"
