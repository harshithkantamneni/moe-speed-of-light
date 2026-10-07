#!/bin/bash
# Job 109, smoke run (offer 48418440, a Threadripper 3970X rented before for job 103e, as 53860926 was no longer offered; not one of the job's hosts): jobs/109_onlinepolicy@vast.sh
# with SMOKE=1 (gpt-oss-20b at C = 4 and 8, 3 problems x 64 steps, gates recorded but not enforced). It checks that
# every configuration, R1 and R2 included, completes and that R1 and R2 pass V3; its numbers enter no prediction.
export SMOKE=1
exec bash "$(cd "$(dirname "$0")" && pwd)/109_onlinepolicy@vast.sh"
