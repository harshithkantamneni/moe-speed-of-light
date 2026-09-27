#!/bin/bash
# Phase 4, 4.1: as job 023, for gpt-oss-120b.
set -x
exec 2>&1
. "$(cd "$(dirname "$0")" && pwd)/prov/setup.sh"
G=$(dirname "$OUT")/022_prov_gen@a100
trc() {
  [ -f "$3" ] || { echo "missing $3"; return; }
  ( cd $P && timeout 40m $WORK/tv/bin/python scripts/trace_corpus.py --repo $MD/$1 --hub-id $2 --corpus "$3" --out $OUT/$4.npz --device cuda ) > $OUT/$4.log 2>&1
  echo "$4 rc=$?"; tail -1 $OUT/$4.log
}
wait_dl gpt-oss-120b || exit 1
trc gpt-oss-120b openai/gpt-oss-120b $P/data/tok_gpt-oss-120b.jsonl gpt-oss-120b_tok
for arm in D G S; do trc gpt-oss-120b openai/gpt-oss-120b "$G/corp_gpt-oss-120b_$arm.jsonl" gpt-oss-120b_$arm; done
trc gpt-oss-120b openai/gpt-oss-120b $P/data/gen020c/gpt-oss-120b-MXFP4_gen.jsonl gpt-oss-120b_gen020c_gpt-oss-120b-MXFP4
ls -la $OUT
