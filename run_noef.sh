#!/usr/bin/env bash
# lanl14argus WITHOUT edge features (--ignore-edge-feats everywhere): pretrain, then fine-tune
# with both heads -- link prediction (lp_finetune.py) and the [CLS] classifier (cls_finetune.py) --
# for static and temporal walks, five pretraining seeds each.
#
# Walk length is 4 for pretraining and fine-tuning, as in the with-features LANL runs, so edge
# features are the only difference. (Without features pretrain.py's LANL default would switch to
# 64-node walks, so --walk-len 4 is passed explicitly.)
#
# Everything carries _noef in its name, so nothing overlaps the main runs:
#   checkpoints  pretrained/<mode>/lanl14argus/{t}rw_bert_lanl14argus_noef_ps1_tiny.pt
#   results      results/{lp,cls}-<mode>/lanl14argus/{lp,cls}_noef_ps1_tiny_wl4_s0.csv
#   job logs     logs/noef/pt_<mode>_lanl14argus_noef_ps1.{sh,out}, ft_<lp|cls>_..._wl4.{sh,out}
# show_main_results.py lists them as lp-static_noef, cls-temporal_noef, ... next to the main rows.
#
#   ./run_noef.sh --dry-run       # print the bsub commands, submit nothing
#   ./run_noef.sh                 # 2 modes x 5 seeds: 10 pretrainings + 20 fine-tunes
#   HEADS=lp MODES=static ./run_noef.sh
#   ./run_noef.sh --force         # delete and redo this experiment's checkpoints / CSVs
#
# Run from the repo root on the login node; every model run is a bsub job.
set -uo pipefail

DATASET=unsw
PSEEDS=${PSEEDS:-"1 2 3 4 5"}
MODES=${MODES:-"static temporal"}
HEADS=${HEADS:-"lp cls"}
WL=16                                      # pretraining and fine-tuning walk length
WLS=${WLS:-$WL}
SIZE=${SIZE:-tiny}
VARIANT=_noef
EXTRA_PRETRAIN=${EXTRA_PRETRAIN:-"--n-tokens 1"}   # 1e9 tokens, as the main LANL runs
EXTRA_FT=${EXTRA_FT:-""}
RESULTS=${RESULTS:-noef}
PRETRAINED=${PRETRAINED:-pretrained}
LOGDIR=${LOGDIR:-logs/noef}

QUEUE=${QUEUE:-normal}
QUEUE_PT=${QUEUE_PT:-$QUEUE}
QUEUE_FT=${QUEUE_FT:-$QUEUE}
GPU_RES=${GPU_RES:-"num=1:mode=exclusive_process"}
HOSTS=${HOSTS:-""}
WALL_PT=${WALL_PT:-"24:00"}
WALL_FT=${WALL_FT:-"24:00"}
MEM_MB=${MEM_MB:-"32000"}
CORES=${CORES:-4}
PRELUDE=${PRELUDE:-"source ~/miniconda3/bin/activate gfm"}
REPO=${REPO:-$PWD}

DRY=0; FORCE=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --force) FORCE=1 ;;
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
    printf '  bsub %s < %s\n      %s\n' "${args[*]}" "$jobsh" "$*" >&2; echo DRYRUN; return 0
  fi
  local res id
  res=$(bsub "${args[@]}" < "$jobsh" 2>&1)
  id=$(printf '%s\n' "$res" | sed -n 's/^Job <\([0-9]*\)>.*/\1/p')
  if [ -z "$id" ]; then
    echo "  SUBMIT FAILED: $name -- bsub said:" >&2; printf '%s\n' "$res" | sed 's/^/    /' >&2
    return 1
  fi
  echo "$id"
}

n_pt=0; n_ft=0
for mode in $MODES; do
  trw=""; pre=rw_bert
  [ "$mode" = temporal ] && { trw="--trw"; pre=trw_bert; }
  for s in $PSEEDS; do
    tag="${VARIANT}_ps${s}"                                     # _noef_ps1
    ck="$PRETRAINED/$mode/$DATASET/${pre}_${DATASET}${tag}_${SIZE}.pt"
    ptname="pt_${mode}_${DATASET}${VARIANT}_ps${s}"

    ptid=""
    if [ "$FORCE" = 0 ] && [ -f "$REPO/$ck" ]; then
      echo "  have      $ck"
    else
      clear=""; [ "$FORCE" = 1 ] && clear="rm -f $ck ${ck%.pt}-best.pt;"
      ptid=$(submit "$ptname" "$QUEUE_PT" "$WALL_PT" "" $clear \
        python pretrain.py --dataset "$DATASET" $trw --ignore-edge-feats --size "$SIZE" \
               --walk-len "$WL" --device 0 --seed "$s" --tag "$tag" $EXTRA_PRETRAIN) || continue
      echo "  pretrain  $ptname  ($ptid)"
      n_pt=$((n_pt + 1))
    fi

    for head in $HEADS; do
      case "$head" in lp|cls) ;; *) echo "unknown head '$head' (lp or cls)" >&2; exit 1 ;; esac
      for wl in $WLS; do
        out_dir="$RESULTS/$head-$mode/$DATASET"
        csv="$out_dir/${head}${tag}_${SIZE}_wl${wl}_s0.csv"
        if [ "$FORCE" = 0 ] && [ -f "$REPO/$csv" ] && grep -q '^# BEST' "$REPO/$csv" 2>/dev/null; then
          echo "  have      $csv"; continue
        fi
        ftname="ft_${head}_${mode}_${DATASET}${VARIANT}_ps${s}_wl${wl}"
        clear=""; [ "$FORCE" = 1 ] && clear="rm -f $csv;"
        guard="[ -f $ck ] || { echo 'no checkpoint $ck -- pretraining did not finish; skipping'; exit 0; };"
        ftid=$(submit "$ftname" "$QUEUE_FT" "$WALL_FT" "$ptid" $clear "$guard" \
          python "${head}_finetune.py" --dataset "$DATASET" $trw --ignore-edge-feats \
                 --size "$SIZE" --device 0 --walk-len "$wl" --seed 0 --pretrain-tag "$tag" \
                 --tag "_s0" --out-dir "$out_dir" $EXTRA_FT) || continue
        echo "  finetune  $ftname  ($ftid${ptid:+, after $ptid})"
        n_ft=$((n_ft + 1))
      done
    done
  done
done

echo
echo "$n_pt pretraining(s) and $n_ft fine-tune(s) $([ "$DRY" = 1 ] && echo listed || echo submitted); LSF logs in $LOGDIR/"
echo "results:  python show_main_results.py --dataset $DATASET --filter _noef"