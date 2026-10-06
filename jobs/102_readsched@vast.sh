#!/bin/bash
# Job 102: can the bound's host term be reached? A microbenchmark (jobs/ec2/readsched.cu) replays MIN's per-layer host
# reads of gpt-oss-120b at C = 14 and 32 (11% and 25% of each layer's experts; counts from scripts/readsched.py on the
# AIME-25 routing of job 084c, whole 13.25 MB experts, cache carried across problems, the first 4,000 steps) through
# CPU helper threads (usable cores - 2, as the engine) and the PCIe copy engine. No model runs and nothing is
# downloaded. Modes: layer (per layer, the experts split whole between CPU and link by the probe's rates; the layer
# waits for both), token (the same split, one wait per token: the bound's view), layer_link (every expert over the
# link), layer_cpu (every expert on the CPU). Two repetitions each. The bound's host term on this host is
# reads per token x S / B_host, with B_host the probe's highest rate (as Eq. 1).
#
# The analytic version (scripts/readsched.py: per layer, the best whole-expert split at the probe's B_c, B_p, B_cp,
# layers in sequence, no latency) is 85-99% of the bound's host term on the 21 hosts probed so far at C = 14, 82-99% at
# C = 32.
#
# Predictions, committed before launch (per host and cell; "fraction" = the bound's host-term time / measured time):
#   1. token mode reaches a fraction >= 0.80;
#   2. layer mode reaches a fraction >= 0.50, and at least 0.03 below the analytic per-layer fraction (the cost of
#      waiting per layer that the analysis leaves out);
#   3. the fraction of layer mode is lower at C = 32 than at C = 14 (fewer experts per layer, so fixed costs weigh more);
#   4. layer_link is slower than layer by more than 10% on a host whose CPU path reads at least 1.5x its link, and
#      within 10% where the two match;
#   5. the two repetitions of each mode agree within 3%.
# Hosts: 102a = an i9-13900KF behind a slow link (Pd of job 099, offer 51748728; Pf's offer was gone at launch); 102b =
# a Ryzen 9 9950X (offer
# 52267630, listing 44.7 GB/s); 102c = a Ryzen 7 9800X3D behind a 27 GB/s link (offer 54227417).
# Budget: the job stops starting steps 45 minutes after launch.
set -x
exec 2>&1
export CUDA_VISIBLE_DEVICES=0
. "$(cd "$(dirname "$0")" && pwd)/ec2/setup.sh"
T0=$(date +%s); BUDGET=$(( 45 * 60 ))
left() { echo $(( T0 + BUDGET - $(date +%s) )); }
platform
nvcc -O3 -arch=sm_$SM "$J/bw.cu" -o $WORK/bw0 && $WORK/bw0 > $OUT/bw_gate.txt 2>&1
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket" | tee $OUT/cpu.txt
CORES=$(usable_cores); echo "usable physical cores $CORES (nproc $(nproc))" | tee $OUT/cores.txt
H=$(( CORES > 4 ? CORES - 2 : 2 ))
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/concur.cu -o $WORK/concur && timeout 15m $WORK/concur > $OUT/concur.txt 2>&1
nvcc -O3 -arch=sm_$SM -Xcompiler -march=native -std=c++17 $J/readsched.cu -o $WORK/readsched || { echo "no readsched binary"; exit 3; }
sha256sum $J/readsched.cu $J/minreads_g14.bin $J/minreads_g32.bin | tee $OUT/inputs_sha.txt
RATES=$(cd $J && python3 -c "
import sys; from fetch_table import bandwidths
bc, bp, bb = bandwidths(open('$OUT/concur.txt').read(), $H); print(f'{bc:.3f} {bp:.3f} {bb:.3f}')")
echo "rates B_c B_p B_cp: $RATES (helpers $H)" | tee $OUT/rates.txt
for C in 14 32; do
  [ "$(left)" -lt 300 ] && { echo "SKIPPED (deadline) C$C" | tee -a $OUT/skipped.txt; continue; }
  timeout $(left) $WORK/readsched $J/minreads_g$C.bin 36 $H $RATES 4000 > $OUT/readsched_C$C.txt 2>&1
  echo "C$C rc=$?"; cat $OUT/readsched_C$C.txt
done
python3 $J/manifest.py "$OUT" 102_readsched@vast "$T0" "102 $(grep -h 'mode=layer rep=0' $OUT/readsched_C*.txt | awk '{print $6}' | tr '\n' ' ')" helpers=$H
echo "SUMMARY: done"
