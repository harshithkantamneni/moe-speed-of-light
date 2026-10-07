#!/bin/bash
# Job 109, smoke run (offer 53860926, a Threadripper PRO 3000; not one of the job's hosts): jobs/109_onlinepolicy@vast.sh
# with SMOKE=1 (gpt-oss-20b at C = 4 and 8, 3 problems x 64 steps, gates recorded but not enforced). It checks that
# every configuration, R1 and R2 included, completes and that R1 and R2 pass V3; its numbers enter no prediction.
export SMOKE=1
exec bash "$(cd "$(dirname "$0")" && pwd)/109_onlinepolicy@vast.sh"
