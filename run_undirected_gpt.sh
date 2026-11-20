#!/usr/bin/env bash
# GPT (causal) base model with undirected walks on the LANL temporal splits, for the BERT-vs-GPT
# ablation: the counterpart of run_undirected.sh's BERT runs. Every random walk -- pretraining
# walks and the context walk in front of each fine-tuned / scored edge -- is drawn on the
# undirected training graph; the edges fine-tuned on and scored stay the directed ones, so the
# test sets are the main runs' files, edge for edge.
#
# Same recipe as the directed GPT runs (run_all_hpc.sh --gpt): GPT-tiny, 1e9 pretraining tokens
# (lanl14argus-ts evaluating every 100 epochs), walk length 4, LP head, static and temporal
# walks, five pretraining seeds. Default: 3 splits x 2 modes x 5 seeds = 30 pretrainings + 30
# fine-tunes, each fine-tune waiting on its own pretraining.
#
# Everything carries _undir in its name, so nothing overlaps the directed GPT runs or the BERT ones:
#   checkpoints  pretrained/<mode>/<dataset>/{t}rw_gpt_<dataset>_undir_ps1_tiny.pt
#   results      results/lp-gpt-<mode>/<dataset>/lp_undir_ps1_tiny_wl4_s0.csv
#   job logs     logs/undirected_gpt/
# show_main_results.py should list them as lp-gpt-static_undir / lp-gpt-temporal_undir.
#
#   (needs the updated pretrain_gpt.py and lp_finetune_gpt.py, which take --undirected-walks)
#   ./run_undirected_gpt.sh --smoke        # ONE short job: tiny pretrain + 1-epoch fine-tune, both modes
#   ./run_undirected_gpt.sh --dry-run      # print the bsub commands, submit nothing
#   ./run_undirected_gpt.sh                # the 60 jobs
#   DATASETS="lanl14argus-ts" MODES=temporal PSEEDS="1 2" ./run_undirected_gpt.sh
#   EXTRA_FT="--epochs 10" ./run_undirected_gpt.sh   # if your directed GPT runs were given extra flags,
#                                                    # pass the same ones here
#   ./run_undirected_gpt.sh --force        # delete and redo this experiment's checkpoints / CSVs
#
# Run from the repo root on the login node; every model run is a bsub job.
set -uo pipefail

DATASETS=${DATASETS:-"lanl14argus-ts lanl14argus-cleants lanl14argus-dirtyts"}
PSEEDS=${PSEEDS:-"1 2 3 4 5"}
MODES=${MODES:-"static temporal"}
WL=4
SIZE=${SIZE:-tiny}
VARIANT=_undir
EXTRA_PRETRAIN=${EXTRA_PRETRAIN:-""}
EXTRA_FT=${EXTRA_FT:-""}
RESULTS=${RESULTS:-results}
PRETRAINED=${PRETRAINED:-pretrained}
LOGDIR=${LOGDIR:-logs/undirected_gpt}

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

DRY=0; FORCE=0; SMOKE=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --force) FORCE=1 ;;
    --smoke) SMOKE=1 ;;
    *) echo "unknown option: $a" >&2; exit 1 ;;
  esac
done
[ -f "$REPO/pretrain_gpt.py" ] || { echo "no pretrain_gpt.py in $REPO -- run from the repo root or set REPO" >&2; exit 1; }
for f in pretrain_gpt.py lp_finetune_gpt.py; do
  grep -q -- '--undirected-walks' "$REPO/$f" \
    || { echo "$f has no --undirected-walks: replace it with the updated version first" >&2; exit 1; }
done
[ "$DRY" = 1 ] || command -v bsub >/dev/null || { echo "bsub not found; run this on a login node" >&2; exit 1; }
mkdir -p "$LOGDIR"

# the directed runs' pretraining settings (run_all_hpc.sh pt_args)
pt_args() { case "$1" in
              lanl14argus-ts) echo "--n-tokens 10 --eval-every 100" ;;
              *)              echo "--n-tokens 10" ;;
            esac; }

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

# ---- --smoke: one short job through both GPT scripts, both modes ---------------------------------
if [ "$SMOKE" = 1 ]; then
  ds=${SMOKE_DATASET:-lanl14argus-cleants}
  body=""
  for mode in static temporal; do
    trw=""; pre=rw_gpt; [ "$mode" = temporal ] && { trw="--trw"; pre=trw_gpt; }
    stem="$PRETRAINED/$mode/$ds/${pre}_${ds}_undir_smoke_${SIZE}"
    body="$body
python pretrain_gpt.py --dataset $ds $trw --undirected-walks --size $SIZE --walk-len $WL --device 0 --seed 1 --tag _undir_smoke --n-tokens 0.02 --eval-every 0
python lp_finetune_gpt.py --dataset $ds $trw --undirected-walks --size $SIZE --device 0 --walk-len $WL --epochs 1 --seed 0 --pretrain-tag _undir_smoke --tag _s0 --out-dir results-smoke/lp-gpt-$mode/$ds
rm -f $stem.pt $stem-best.pt ${stem}_log.csv ${stem}_eval.csv"
  done
  body="$body
echo SMOKE OK"
  id=$(submit "smoke_undirected_gpt" "$QUEUE" "02:00" "" "$body") || exit 1
  echo "smoke job: $id   (tail -f $LOGDIR/smoke_undirected_gpt.out)"
  echo "it should print 'walks=undirected' four times and end with 'SMOKE OK'; results-smoke/ can be deleted"
  exit 0
fi

# ---- the experiment ----------------------------------------------------------------------------
n_pt=0; n_ft=0
for ds in $DATASETS; do
  for s in tr va te; do
    [ -f "$REPO/data/${ds}_tgraph_${s}.pt" ] || [ "$DRY" = 1 ] \
      || { echo "missing data/${ds}_tgraph_${s}.pt" >&2; exit 1; }
  done
  for mode in $MODES; do
    trw=""; pre=rw_gpt
    [ "$mode" = temporal ] && { trw="--trw"; pre=trw_gpt; }
    for s in $PSEEDS; do
      ptag="${VARIANT}_ps${s}"
      ck="$PRETRAINED/$mode/$ds/${pre}_${ds}${ptag}_${SIZE}.pt"
      ptname="gpt_pt_${mode}_${ds}${VARIANT}_ps${s}"

      ptid=""
      if [ "$FORCE" = 0 ] && [ -f "$REPO/$ck" ]; then
        echo "  have      $ck"
      else
        clear=""; [ "$FORCE" = 1 ] && clear="rm -f $ck ${ck%.pt}-best.pt;"
        ptid=$(submit "$ptname" "$QUEUE_PT" "$WALL_PT" "" $clear \
          python pretrain_gpt.py --dataset "$ds" $trw --undirected-walks --size "$SIZE" \
                 --walk-len "$WL" --device 0 --seed "$s" --tag "$ptag" $(pt_args "$ds") $EXTRA_PRETRAIN) || continue
        echo "  pretrain  $ptname  ($ptid)"
        n_pt=$((n_pt + 1))
      fi

      out_dir="$RESULTS/lp-gpt-$mode/$ds"
      csv="$out_dir/lp${ptag}_${SIZE}_wl${WL}_s0.csv"
      if [ "$FORCE" = 0 ] && [ -f "$REPO/$csv" ] && grep -q '^# BEST' "$REPO/$csv" 2>/dev/null; then
        echo "  have      $csv"; continue
      fi
      ftname="gpt_ft_lp_${mode}_${ds}${VARIANT}_ps${s}_wl${WL}"
      clear=""; [ "$FORCE" = 1 ] && clear="rm -f $csv;"
      guard="[ -f $ck ] || { echo 'no checkpoint $ck -- pretraining did not finish; skipping'; exit 0; };"
      ftid=$(submit "$ftname" "$QUEUE_FT" "$WALL_FT" "$ptid" $clear "$guard" \
        python lp_finetune_gpt.py --dataset "$ds" $trw --undirected-walks \
               --size "$SIZE" --device 0 --walk-len "$WL" --seed 0 --pretrain-tag "$ptag" \
               --tag _s0 --out-dir "$out_dir" $EXTRA_FT) || continue
      echo "  finetune  $ftname  ($ftid${ptid:+, after $ptid})"
      n_ft=$((n_ft + 1))
    done
  done
done

echo
echo "$n_pt pretraining(s) and $n_ft fine-tune(s) $([ "$DRY" = 1 ] && echo listed || echo submitted); LSF logs in $LOGDIR/"
echo "results:  python show_main_results.py --filter _undir      (kill all: bkill -J 'gpt_*_undir_*' 0)"