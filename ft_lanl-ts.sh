#!/usr/bin/env bash
set -euo pipefail
DS=lanl14argus-dirtyts
DEV=3
WLS="1 2 4 6 8 10 16 32"
SEEDS="0 1 2 3 4"

python pretrain.py --dataset $DS --device $DEV --eval-every 100   # adjust after checking tokens/epoch

for ckpt in final best; do
  flag=""
  [ "$ckpt" = best ] && flag="--best-pretrained"
  for wl in $WLS; do
    for seed in $SEEDS; do
      python lp_finetune.py --dataset $DS --device $DEV --walk-len $wl $flag \
        --seed $seed --tag "_${ckpt}_s${seed}"
    done
  done
done