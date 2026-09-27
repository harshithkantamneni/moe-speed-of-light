#!/bin/bash
# Op-level tests of the expert-cache kernel changes, CUDA vs CPU. Job 008 ran the filter while the
# new cases sat in a disabled block (0/0 run); this registers them in the eval set and runs them.
set -x
exec 2>&1
J=$(cd "$(dirname "$0")" && pwd)/ec
cd $WORK/llama.cpp-ec && git apply "$J/optests.patch" && cmake --build build -j$(nproc) --target test-backend-ops > "$OUT/build.txt" 2>&1; echo "build rc=$?"
./build/bin/test-backend-ops -b CUDA0 -o MUL_MAT_ID_EC,ADD_ID_EC > "$OUT/test_ec.txt" 2>&1; echo "rc=$?"
grep -E "tests passed|FAIL" "$OUT/test_ec.txt"; grep -c "OK" "$OUT/test_ec.txt"
./build/bin/test-backend-ops -b CUDA0 -o MUL_MAT_ID,ADD_ID > "$OUT/test_regular.txt" 2>&1; echo "rc=$?"
grep -E "tests passed" "$OUT/test_regular.txt"
