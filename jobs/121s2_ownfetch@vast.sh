#!/bin/bash
# Job 121, second smoke run (not one of the job's hosts; the first, 121s, ran pr14 and pr32 with the cache off and an all-zero fetch table): jobs/121_ownfetch@vast.sh with SMOKE=1 (gpt-oss-20b, 2 problems x 32
# steps per text, gates recorded but not enforced). It checks that both engines build, that every configuration of
# processes g, s and p completes, that own makes in-step fetches with its victims in MIN's order, and that upstream's
# MoE cache logs its slots and counters; its numbers enter no prediction.
export SMOKE=1
exec bash "$(cd "$(dirname "$0")" && pwd)/121_ownfetch@vast.sh"
