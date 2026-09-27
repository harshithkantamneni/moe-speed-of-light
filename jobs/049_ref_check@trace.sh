#!/bin/bash
# Reference check of the audit models' real traces: stock transformers (whole model, fp32, on the Grace CPU) against
# the layer-streamed collector's selections and NLL on the first conversations of the D packs (scripts/ref_check.py).
# Why: the collector's NLL on Mixtral and Phi-3.5-MoE differs from vLLM-FP8's by 0.1 nats (Phi-3.5: the collector's
# final LayerNorm/lm_head bias are not loaded, which affects only its NLL check, not routing).
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/prov/setup.sh"
R=$(dirname $OUT)
free -g
for spec in "phi3.5-moe 047b_audit_phi35@trace 2" "mixtral-8x7b 045b_audit_mixtral@trace 2" "qwen2-57b 046b_audit_qwen2_57b@trace 1"; do
  set -- $spec
  ( cd $P && OMP_NUM_THREADS=$(nproc) timeout 35m $WORK/tv/bin/python scripts/ref_check.py --model-dir $MD/$1 \
      --pack $R/$2/$1_D.npz --corpus $R/$2/corp_$1_D.jsonl --score $R/$2/score_$1_D.jsonl --n $3 ) > $OUT/ref_$1.log 2>&1
  echo "$1 rc=$?"; tail -4 $OUT/ref_$1.log
done
