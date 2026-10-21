#!/bin/bash
# Job 123, smoke run (not one of the job's hosts): jobs/123_confirm@vast.sh with SMOKE=1 (gpt-oss-20b, 2 problems x 32
# steps, budgets 14 and 24, gates recorded but not enforced). It checks that the engine builds, that the plans are
# computed, that all ten arms complete (lru for the first time in ec-bench) and write their per-layer stats, and that
# the summary and the manifest are written; its numbers enter no prediction.
export SMOKE=1
exec bash "$(cd "$(dirname "$0")" && pwd)/123_confirm@vast.sh"
