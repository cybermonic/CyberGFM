#!/usr/bin/env bash
# OpTC walk-length sweep: pretrain once, then CLS fine-tune (as in the paper) from the
# final and best pretraining checkpoints over several walk lengths and seeds.
#
# Select settings by validation AUC (va_auc), never by test scores.
#
# Usage: ./sweep_optc.sh DATASET DEVICE [--trw]
#   e.g. ./sweep_optc.sh optc-ts 3
#        ./sweep_optc.sh optc-ts 3 --trw
set -euo pipefail

usage() { echo "Usage: $0 DATASET DEVICE [--trw]" >&2; exit 1; }

[ $# -ge 2 ] || usage
DS=$1
DEV=$2
shift 2

TRW=""
while [ $# -gt 0 ]; do
  case "$1" in
    --trw) TRW="--trw" ;;
    *) usage ;;
  esac
  shift
done

# Pretraining walk length (OpTC has no edge features, so 1 hop = 1 token) and fine-tuning walk lengths.
# Probe on optc-ts with a 64-hop cap: static walks average 41.3 nodes, temporal 40.7, so the
# graph is dense enough that time ordering barely shortens walks. Both modes use the same grid.
PT_WL=64
WLS="1 2 4 8 16 32"
PT_TAG="_wl${PT_WL}"
SEEDS="3 4"

echo "dataset=$DS device=$DEV ${TRW:-static} pretrain_walk_len=$PT_WL fine-tune walk lengths: $WLS"

#python pretrain.py --dataset "$DS" --device "$DEV" $TRW --walk-len $PT_WL --tag "$PT_TAG"

for ckpt in final best; do
  flag=""
  [ "$ckpt" = best ] && flag="--best-pretrained"
  for wl in $WLS; do
    for seed in $SEEDS; do
      python cls_finetune.py --dataset "$DS" --device "$DEV" --walk-len "$wl" $flag $TRW \
        --pretrain-tag "$PT_TAG" --seed "$seed" --tag "_${ckpt}_s${seed}"
    done
  done
done