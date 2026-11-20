#!/usr/bin/env bash
# Pretraining-length sweep on lanl14argus: pretrain with --n-tokens 1 3 5 7 9 (x 1e8 tokens), five
# trials each (pretraining seeds 1-5, the same seeds as the main runs), all
# submitted at once so they run in parallel, each followed by a fine-tune chained to its own
# pretraining (-w "ended(...)"; the fine-tune exits cleanly if no checkpoint was produced).
#
# Everything carries the token budget in its name:
#   checkpoints  pretrained/<mode>/lanl14argus/{t}rw_bert_lanl14argus_1e8_ps1_tiny.pt   (--tag)
#   pretrain logs (train/eval CSVs) next to the checkpoint, same stem
#   fine-tune CSVs  results-ntokens/lp-<mode>/lanl14argus/lp_1e8_ps1_tiny_wl4_s0.csv
#   LSF logs     logs/ntokens/pt_<mode>_lanl14argus_1e8_ps1.{sh,out}, ft_..._1e8_ps1_wl4.{sh,out}
#
#   ./run_ntokens.sh --dry-run                 # print the bsub commands, submit nothing
#   ./run_ntokens.sh                           # 5 budgets x 5 seeds: 25 pretrainings + 25 fine-tunes
#   PSEEDS="1" MODES="static temporal" ./run_ntokens.sh
#   NTOKENS="1 3 5 7 9 11" ./run_ntokens.sh
#   ./run_ntokens.sh --pretrain-only
#   ./run_ntokens.sh --force                   # delete and redo existing checkpoints / CSVs
#
# Run from the repo root on the login node; every model run is a bsub job.
set -uo pipefail

DATASET=lanl14argus
NTOKENS=${NTOKENS:-"1 2 3 4 5 6 7 8 9"}          # units of 1e8 tokens (pretrain.py --n-tokens)
PSEEDS=${PSEEDS:-"1 2 3 4 5"}      # one trial per pretraining seed
MODES=${MODES:-"static"}                 # static and/or temporal
SIZE=${SIZE:-tiny}
WL=${WL:-4}                              # fine-tune walk length
EXTRA_PRETRAIN=${EXTRA_PRETRAIN:-""}
EXTRA_FT=${EXTRA_FT:-""}
RESULTS=${RESULTS:-results-ntokens}
PRETRAINED=${PRETRAINED:-pretrained}
LOGDIR=${LOGDIR:-logs/ntokens}

QUEUE=${QUEUE:-normal}
QUEUE_PT=${QUEUE_PT:-$QUEUE}
QUEUE_FT=${QUEUE_FT:-$QUEUE}
GPU_RES=${GPU_RES:-"num=1:mode=exclusive_process"}
HOSTS=${HOSTS:-""}
WALL_PT=${WALL_PT:-"12:00"}
WALL_FT=${WALL_FT:-"08:00"}
MEM_MB=${MEM_MB:-"32000"}
CORES=${CORES:-4}
PRELUDE=${PRELUDE:-"source ~/miniconda3/bin/activate gfm"}
REPO=${REPO:-$PWD}

DRY=0; FORCE=0; PT_ONLY=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --force) FORCE=1 ;;
    --pretrain-only) PT_ONLY=1 ;;
    *) echo "unknown option: $a" >&2; exit 1 ;;
  esac
done
[ -f "$REPO/pretrain.py" ] || { echo "no pretrain.py in $REPO -- run from the repo root or set REPO" >&2; exit 1; }
[ "$DRY" = 1 ] || command -v bsub >/dev/null || { echo "bsub not found; run this on a login node" >&2; exit 1; }
mkdir -p "$LOGDIR"

submit() {  # submit <jobname> <queue> <wall> <dependency-or-empty> <command...>; prints job id
  local name="$1" queue="$2" wall="$3" dep="$4"; shift 4
  local out="$LOGDIR/${name}.out" jobsh="$LOGDIR/${name}.sh"
  {
    echo "#!/usr/bin/env bash"
    echo "set -e"
    printf 'cd %q\n' "$REPO"
    [ -n "$PRELUDE" ] && echo "$PRELUDE"
    echo 'echo "host: $(hostname)  gpus: ${CUDA_VISIBLE_DEVICES:-unset}  started: $(date -Is)"'
    echo "$*"
  } > "$jobsh"
  local args=(-J "$name" -q "$queue" -gpu "$GPU_RES" -n "$CORES" -W "$wall"
              -R "rusage[mem=$MEM_MB]" -o "$out" -e "$out")
  [ -n "$dep" ] && [ "$dep" != DRYRUN ] && args+=(-w "ended($dep)")
  [ -n "$HOSTS" ] && args+=(-m "$HOSTS")
  if [ "$DRY" = 1 ]; then
    printf '  bsub %s < %s\n' "${args[*]}" "$jobsh" >&2; echo DRYRUN; return 0
  fi
  local id
  id=$(bsub "${args[@]}" < "$jobsh" 2>&1 | sed -n 's/^Job <\([0-9]*\)>.*/\1/p')
  [ -n "$id" ] || { echo "  SUBMIT FAILED: $name" >&2; return 1; }
  echo "$id"
}

n_pt=0; n_ft=0
for mode in $MODES; do
  trw=""; pre=rw_bert
  [ "$mode" = temporal ] && { trw="--trw"; pre=trw_bert; }
  for s in $PSEEDS; do
    for nt in $NTOKENS; do
      tag="_${nt}e8_ps${s}"                       # e.g. _1e8_ps1
      ck="$PRETRAINED/$mode/$DATASET/${pre}_${DATASET}${tag}_${SIZE}.pt"
      ptname="pt_${mode}_${DATASET}_${nt}e8_ps${s}"

      ptid=""
      if [ "$FORCE" = 0 ] && [ -f "$REPO/$ck" ]; then
        echo "  have      $ck"
      else
        clear=""; [ "$FORCE" = 1 ] && clear="rm -f $ck ${ck%.pt}-best.pt;"
        ptid=$(submit "$ptname" "$QUEUE_PT" "$WALL_PT" "" $clear \
          python pretrain.py --dataset "$DATASET" $trw --size "$SIZE" --device 0 \
                 --seed "$s" --n-tokens "$nt" --tag "$tag" $EXTRA_PRETRAIN) || continue
        echo "  pretrain  $ptname  ($ptid)"
        n_pt=$((n_pt + 1))
      fi
      [ "$PT_ONLY" = 1 ] && continue

      out_dir="$RESULTS/lp-$mode/$DATASET"
      csv="$out_dir/lp${tag}_${SIZE}_wl${WL}_s0.csv"
      if [ "$FORCE" = 0 ] && [ -f "$REPO/$csv" ] && grep -q '^# BEST' "$REPO/$csv" 2>/dev/null; then
        echo "  have      $csv"; continue
      fi
      ftname="ft_${mode}_${DATASET}_${nt}e8_ps${s}_wl${WL}"
      clear=""; [ "$FORCE" = 1 ] && clear="rm -f $csv;"
      guard="[ -f $ck ] || { echo 'no checkpoint $ck -- pretraining did not finish; skipping'; exit 0; };"
      ftid=$(submit "$ftname" "$QUEUE_FT" "$WALL_FT" "$ptid" $clear "$guard" \
        python lp_finetune.py --dataset "$DATASET" $trw --size "$SIZE" --device 0 \
               --walk-len "$WL" --seed 0 --pretrain-tag "$tag" --tag "_s0" \
               --out-dir "$out_dir" $EXTRA_FT) || continue
      echo "  finetune  $ftname  ($ftid${ptid:+, after $ptid})"
      n_ft=$((n_ft + 1))
    done
  done
done

echo
echo "$n_pt pretraining(s) and $n_ft fine-tune(s) submitted; LSF logs in $LOGDIR/"
[ "$DRY" = 1 ] || echo "follow one with:  bpeek -f <jobid>"