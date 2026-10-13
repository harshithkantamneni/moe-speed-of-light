#!/bin/bash
# CPU test of engine patch oracle6 (LLAMA_EC_ORACLE_FORECAST / _FC_LAG), run before job 117's registration, on a
# tiny Qwen3-MoE GGUF (4 layers, 32 experts, top-4) and a CPU build of llama-ec-bench with the patch:
#  (1) a forecast equal to the trace (lag 0, H = 4) gives the exact window's counters (hybrid W = 4, fetch path);
#  (2) random forecasts at lags 0, 1, 2 give exactly vmap.replay_fc's reads on the forecast as the engine sees it;
#  (3) without a forecast, the hybrid configurations equal vmap.replay (exact window, random-error window, no window).
#     EC_BIN=.../llama-ec-bench GGUF=tiny.gguf CORPUS=corpus.jsonl OUT=dir bash cputest117.sh C
set -u
J="$(cd "$(dirname "$0")" && pwd)"
B=${EC_BIN:?}; G=${GGUF:?}; O=${OUT:?}; C=${1:-8}
rm -rf $O; mkdir -p $O
cp ${CORPUS:?} $O/corpus.jsonl
COMMON="--corpus $O/corpus.jsonl -t 2 --n-prefill 32 --n-decode 160"
$B -m $G $COMMON --lookahead $O/la.bin > $O/la.jsonl 2> $O/la.err || { tail $O/la.err; exit 1; }
python3 $J/cputest117_mkfc.py $O || exit 1
orc="slots=$C:policy=oracle:oracle=$O/la.bin:overlap=1:kappa=-1000000000:oracle_bypass=1:oracle_fetch=1:oracle_hybrid=1:oracle_w=4"
: > $O/configs.txt
add() { echo "$orc:$1:stats=$O/$2.json" >> $O/configs.txt; }
add "oracle_w=4" exact_w4
add "oracle_forecast=$O/fc_exact.bin" fc_exact_l0
add "oracle_forecast=$O/fc_rand.bin" fc_rand_l0
add "oracle_forecast=$O/fc_rand.bin:oracle_fc_lag=1" fc_rand_l1
add "oracle_forecast=$O/fc_rand.bin:oracle_fc_lag=2" fc_rand_l2
add "oracle_forecast=$O/fc_exact.bin:oracle_fc_lag=1" fc_exact_l1
add "oracle_w=4:oracle_recall=0.7:oracle_fill=1:oracle_seed=1" r7_w4
add "oracle_w=-1" aa
cfg=$(paste -sd';' $O/configs.txt)
$B -m $G $COMMON --order-seed 9 --ec "$cfg" > $O/ec.jsonl 2> $O/ec.err || { tail -20 $O/ec.err; exit 1; }
python3 $J/cputest117_check.py $O $C
