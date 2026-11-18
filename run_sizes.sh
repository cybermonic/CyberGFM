#!/usr/bin/env bash
# Model-size experiment: BERT mini, med and baseline on lanl14argus, static walks, fine-tuned and
# evaluated at walk length 4 only, five pretraining seeds each, everything on the h100 queue.
# The tiny model is the main runs (results/lp-static/lanl14argus/lp_ps*_tiny_wl4_*), so it is
# not rerun here.
#
# This drives run_all_hpc.sh once per size, so it inherits its rules: one bsub job per cell,
# fine-tunes chained to their pretraining with ended(), existing checkpoints / finished CSVs
# skipped, --force to redo. Names carry the size, so nothing overlaps the tiny runs:
#   checkpoints  pretrained/static/lanl14argus/rw_bert_lanl14argus_ps1_<size>.pt
#   results      results/lp-static/lanl14argus/lp_ps1_<size>_wl4_ps1_s0.csv
#   job logs     logs/size-<size>/pt_static_lanl14argus_<size>_ps1.{sh,out}, ft_..._wl4.{sh,out}
#
#   ./run_sizes.sh --dry-run          # check the bsub commands first
#   ./run_sizes.sh                    # 3 sizes x 5 seeds = 15 pretrainings + 15 fine-tunes
#   SIZES="baseline" ./run_sizes.sh   # one size only
#   WALL_PT=49:00 ./run_sizes.sh      # the h100 maximum (RUNLIMIT 2940 min)
#
# Run from the repo root on the login node.
set -uo pipefail

SIZES=${SIZES:-"mini small med baseline"}
here=$(cd "$(dirname "$0")" && pwd)
runner=${RUNNER:-$here/run_all_hpc.sh}
[ -f "$runner" ] || runner=./run_all_hpc.sh
[ -f "$runner" ] || { echo "run_all_hpc.sh not found (set RUNNER)" >&2; exit 1; }

# Pretraining cost grows steeply with size (baseline is BERT-base: 12 layers x 768), and a job
# killed at its wall limit still leaves the last periodic checkpoint behind, which the fine-tune
# would then use without complaint. So ask for a long wall time by default: if it exceeds the
# queue's RUNLIMIT, bsub refuses the job loudly (SUBMIT FAILED) instead of truncating it quietly.
#   bqueues -l h100 | grep -A2 -i runlimit
export WALL_PT=${WALL_PT:-"48:00"}   # h100: default RUNLIMIT 2880 min, max 2940 (49:00)
export WALL_FT=${WALL_FT:-"24:00"}

for size in $SIZES; do
  case "$size" in mini|small|med|baseline|tiny) ;; *) echo "unknown size '$size'" >&2; exit 1 ;; esac
  echo "=== $size ==="
  SIZE=$size DATASETS="lanl14argus optc" WLS=4 \
  LOGDIR=logs/size-$size \
    bash "$runner" "$@" || exit $?
  echo
done