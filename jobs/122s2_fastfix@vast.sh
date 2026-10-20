#!/bin/bash
# Job 122, second smoke run (the first, 122s, never started its container; not one of the job's hosts): jobs/122_fastfix@vast.sh with SMOKE=1 (gpt-oss-20b, 2 problems x 32
# steps, gates recorded but not enforced). It checks that the engine builds, that the four configurations complete, and
# that own makes in-step fetches with its victims in MIN's order; its numbers enter no prediction.
export SMOKE=1
exec bash "$(cd "$(dirname "$0")" && pwd)/122_fastfix@vast.sh"
