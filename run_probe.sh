#!/usr/bin/env bash
# Validation probe: which validation metric tracks test AP, and does a lower fine-tuning learning
# rate flatten the per-epoch curves? Re-runs LP fine-tuning from the EXISTING pretrained
# checkpoints with --val-probe, which computes every candidate validation metric at every epoch
# (common.validation_probe) next to the usual results. Nothing is pretrained and nothing under
# results/ is touched.
#
#   datasets  lanl14argus (random split), lanl14argus-dirtyts (7-day split), unsw
#   modes     static and temporal
#   lr        3e-4 (lp_finetune.py's default) and 3e-5 (cls_finetune.py's)
#   seeds     pretraining seeds 1 and 2 -- fix the rule on these, confirm on 3-5 afterwards:
#             PSEEDS="3 4 5" ./run_probe.sh
#   walk len  4 on LANL, 16 on UNSW (run_all_hpc.sh's WLS=match)
#
# Output, all under results-probe/:
#   results-probe/lp-<mode>/<dataset>/lp_ps1_tiny_wl4_ps1_s0_lr3e-4.csv         the usual results CSV
#   results-probe/lp-<mode>/<dataset>/probe/lp_ps1_tiny_wl4_ps1_s0_lr3e-4.csv   the probe metrics
#
#   ./run_probe.sh --dry-run
#   ./run_probe.sh                       # 3 datasets x 2 modes x 2 rates x 2 seeds = 24 fine-tunes
#   python show_probe.py                 # once they finish (text only; fine on the login node)
#   DATASETS=unsw MODES=static LRS=3e-4 ./run_probe.sh
#
# Needs the --val-probe versions of common.py and lp_finetune.py. Run from the repo root on the
# login node; every run is a bsub job.
set -uo pipefail

DATASETS=${DATASETS:-"lanl14argus lanl14argus-dirtyts unsw"}
MODES=${MODES:-"static temporal"}
LRS=${LRS:-"3e-4 3e-5"}
PSEEDS=${PSEEDS:-"1 2"}
SIZE=${SIZE:-tiny}
RESULTS=${RESULTS:-results-probe}
PRETRAINED=${PRETRAINED:-pretrained}
EXTRA_FT=${EXTRA_FT:-""}
QUEUE=${QUEUE:-normal}
GPU_RES=${GPU_RES:-"num=1:mode=exclusive_process"}
CORES=${CORES:-4}
WALL=${WALL:-"12:00"}
MEM_MB=${MEM_MB:-32000}
PRELUDE=${PRELUDE:-"source ~/miniconda3/bin/activate gfm"}
LOGDIR=${LOGDIR:-logs/probe}
REPO=${REPO:-$PWD}

DRY=0; FORCE=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --force) FORCE=1 ;;
    *) echo "unknown option: $a" >&2; exit 1 ;;
  esac
done
[ -f "$REPO/lp_finetune.py" ] || { echo "run this from the repo root" >&2; exit 1; }
grep -q -- '--val-probe' "$REPO/lp_finetune.py" && grep -q 'def validation_probe' "$REPO/common.py" \
  || { echo "lp_finetune.py / common.py have no --val-probe: copy in the probe versions first" >&2; exit 1; }
[ "$DRY" = 1 ] || command -v bsub >/dev/null || { echo "bsub not found; run this on a login node" >&2; exit 1; }
mkdir -p "$LOGDIR"

ft_walk_len() { case "$1" in unsw*) echo 16 ;; *) echo 4 ;; esac; }

n=0
for ds in $DATASETS; do
  case "$ds" in lanl*|unsw*) ;; *) echo "  $ds: the probe is for the LP datasets (lanl*, unsw*); skipped" >&2; continue ;; esac
  wl=$(ft_walk_len "$ds")
  for mode in $MODES; do
    trw=""; pre=rw_bert; [ "$mode" = temporal ] && { trw="--trw"; pre=trw_bert; }
    for s in $PSEEDS; do
      ck="$PRETRAINED/$mode/$ds/${pre}_${ds}_ps${s}_${SIZE}.pt"
      if [ ! -f "$REPO/$ck" ] && [ "$DRY" = 0 ]; then echo "  no checkpoint $ck -- skipped" >&2; continue; fi
      for lr in $LRS; do
        out_dir="$RESULTS/lp-$mode/$ds"
        stem="lp_ps${s}_${SIZE}_wl${wl}_ps${s}_s0_lr${lr}"
        if [ "$FORCE" = 0 ] && grep -q '^# BEST' "$REPO/$out_dir/$stem.csv" 2>/dev/null && [ -f "$REPO/$out_dir/probe/$stem.csv" ]; then
          echo "  have      $out_dir/$stem.csv"; continue
        fi
        name="probe_${mode}_${ds}_ps${s}_lr${lr}"
        jobsh="$LOGDIR/$name.sh"; out="$LOGDIR/$name.out"
        cmd="python lp_finetune.py --dataset $ds $trw --size $SIZE --device 0 --walk-len $wl --seed 0 --lr $lr --pretrain-tag _ps${s} --tag _ps${s}_s0_lr${lr} --out-dir $out_dir --val-probe $EXTRA_FT"
        {
          echo "#!/usr/bin/env bash"
          echo "set -e"
          printf 'cd %q\n' "$REPO"
          [ -n "$PRELUDE" ] && echo "$PRELUDE"
          echo 'echo "host: $(hostname)  gpus: ${CUDA_VISIBLE_DEVICES:-unset}  started: $(date -Is)"'
          echo "$cmd"
        } > "$jobsh"
        args=(-J "$name" -q "$QUEUE" -gpu "$GPU_RES" -n "$CORES" -W "$WALL" -R "rusage[mem=$MEM_MB]" -o "$out" -e "$out")
        if [ "$DRY" = 1 ]; then
          echo "  bsub ${args[*]} < $jobsh"; echo "      $cmd"; n=$((n + 1)); continue
        fi
        rm -f "$out"
        res=$(bsub "${args[@]}" < "$jobsh" 2>&1)
        id=$(printf '%s\n' "$res" | sed -n 's/^Job <\([0-9]*\)>.*/\1/p')
        if [ -z "$id" ]; then echo "  SUBMIT FAILED $name -- bsub said:" >&2; printf '%s\n' "$res" | sed 's/^/    /' >&2; continue; fi
        printf '  %-52s %s\n' "$name" "$id"; n=$((n + 1))
      done
    done
  done
done
echo
echo "$n job(s) $([ "$DRY" = 1 ] && echo listed || echo submitted); logs in $LOGDIR/   (kill all: bkill -J 'probe_*' 0)"
echo "then: python show_probe.py --root $RESULTS"
