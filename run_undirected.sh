#!/usr/bin/env bash
# Undirected-walk experiment on the LANL splits: every random walk -- the pretraining walks and the
# context walk in front of each fine-tuned / scored edge -- is sampled on the UNDIRECTED training
# graph (each edge can be walked both ways, keeping its timestamp and features). The edges that
# are fine-tuned on and the validation / test edges that are scored are the directed ones, so the
# test sets are the same files, edge for edge, as in the main runs.
#
# Everything else is the main LANL recipe: BERT-tiny, 1e9 pretraining tokens, walk length 4 for
# pretraining and fine-tuning, LP head, static and temporal walks, five pretraining seeds.
#
# Everything carries _undir in its name, so nothing overlaps the main runs:
#   checkpoints  pretrained/<mode>/<dataset>/{t}rw_bert_<dataset>_undir_ps1_tiny.pt
#   results      results/lp-<mode>/<dataset>/lp_undir_ps1_tiny_wl4_s0.csv
#   job logs     logs/undirected/pt_<mode>_<dataset>_undir_ps1.{sh,out}, ft_lp_..._wl4.{sh,out}
# show_main_results.py lists them as lp-static_undir / lp-temporal_undir next to the main rows.
#
#   ./run_undirected.sh --smoke       # ONE short job: check_undirected.py on every dataset (graph and
#                                     # walk checks, and how much context the directed walks were
#                                     # missing), then a tiny pretrain and a one-epoch fine-tune in
#                                     # each mode on lanl14argus. Do this first.
#   ./run_undirected.sh --dry-run     # print the bsub commands, submit nothing
#   ./run_undirected.sh               # 4 datasets x 2 modes x 5 seeds: 40 pretrainings + 40 fine-tunes
#   ./run_undirected.sh --ctx-only    # no new pretraining: fine-tune the MAIN (directed-walk)
#                                     # checkpoints with undirected context -> results-undir-ctx/
#   DATASETS="lanl14argus-ts" MODES=temporal PSEEDS="1 2 3" ./run_undirected.sh
#   DATASETS="lanl14argus-dirtyts8d lanl14argus-cleants8d" ./run_undirected.sh
#   ./run_undirected.sh --force       # delete and redo this experiment's checkpoints / CSVs
#
# Needs the --undirected-walks patch (common.py, pretrain.py, lp_finetune.py, cls_finetune.py).
# Run from the repo root on the login node; every model run is a bsub job.
set -uo pipefail

DATASETS=${DATASETS:-"lanl14argus-ts lanl14argus-dirtyts lanl14argus-cleants lanl14argus"}
PSEEDS=${PSEEDS:-"1 2 3 4 5"}
MODES=${MODES:-"static temporal"}
HEADS=${HEADS:-"lp"}
WL=4                                      # pretraining walk length (pretrain.py's LANL default)
WLS=${WLS:-$WL}                           # fine-tuning walk length(s)
SIZE=${SIZE:-tiny}
VARIANT=_undir
EXTRA_PRETRAIN=${EXTRA_PRETRAIN:-""}
EXTRA_FT=${EXTRA_FT:-""}
RESULTS=${RESULTS:-results}
PRETRAINED=${PRETRAINED:-pretrained}
LOGDIR=${LOGDIR:-logs/undirected}

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

DRY=0; FORCE=0; SMOKE=0; CTX_ONLY=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --force) FORCE=1 ;;
    --smoke) SMOKE=1 ;;
    --ctx-only) CTX_ONLY=1 ;;
    *) echo "unknown option: $a" >&2; exit 1 ;;
  esac
done
[ -f "$REPO/pretrain.py" ] || { echo "no pretrain.py in $REPO -- run from the repo root or set REPO" >&2; exit 1; }
for f in pretrain.py lp_finetune.py cls_finetune.py; do
  grep -q -- '--undirected-walks' "$REPO/$f" \
    || { echo "$f has no --undirected-walks: apply undirected-walks.patch first" >&2; exit 1; }
done
[ "$DRY" = 1 ] || command -v bsub >/dev/null || { echo "bsub not found; run this on a login node" >&2; exit 1; }
# --ctx-only writes to its own results root: its files are named like the main runs' (same
# checkpoints), so they must not share a directory with them.
if [ "$CTX_ONLY" = 1 ] && [ "$RESULTS" = results ]; then RESULTS=results-undir-ctx; fi
mkdir -p "$LOGDIR"

# The main runs' pretraining settings (run_all_hpc.sh pt_args): 1e9 tokens; lanl14argus-ts
# evaluates every 100 epochs because its test set is most of the data.
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

# ---- --smoke: one short job that exercises every new code path ---------------------------------
if [ "$SMOKE" = 1 ]; then
  ds=${SMOKE_DATASET:-lanl14argus}
  [ -f "$REPO/check_undirected.py" ] || { echo "put check_undirected.py in the repo root" >&2; exit 1; }
  rm -f "$LOGDIR/smoke_undirected.out"            # LSF appends; start the log clean
  body="python check_undirected.py --datasets $DATASETS --walks --device 0"
  for mode in static temporal; do
    trw=""; pre=rw_bert; [ "$mode" = temporal ] && { trw="--trw"; pre=trw_bert; }
    ck="$PRETRAINED/$mode/$ds/${pre}_${ds}_undir_smoke_${SIZE}"
    body="$body
python pretrain.py --dataset $ds $trw --undirected-walks --size $SIZE --walk-len $WL --device 0 --seed 1 --tag _undir_smoke --n-tokens 0.02 --eval-every 0
python lp_finetune.py --dataset $ds $trw --undirected-walks --size $SIZE --device 0 --walk-len $WL --epochs 1 --seed 0 --pretrain-tag _undir_smoke --tag _s0 --out-dir results-smoke/lp-$mode/$ds
rm -f $ck.pt $ck-best.pt $PRETRAINED/$mode/$ds/${pre}_${ds}_undir_smoke_${SIZE}_log.csv $PRETRAINED/$mode/$ds/${pre}_${ds}_undir_smoke_${SIZE}_eval.csv"
  done
  body="$body
echo SMOKE OK"
  id=$(submit "smoke_undirected" "$QUEUE" "02:00" "" "$body") || exit 1
  echo "smoke job: $id   (bpeek -f $id, or tail -f $LOGDIR/smoke_undirected.out)"
  echo "it ends with 'SMOKE OK'; its two result files are under results-smoke/ and can be deleted"
  exit 0
fi

# ---- the experiment ----------------------------------------------------------------------------
n_pt=0; n_ft=0
for ds in $DATASETS; do
  case "$ds" in lanl*) ;; *) echo "  note: $ds is not a LANL split; this script uses the LANL recipe (wl $WL, 1e9 tokens)" >&2 ;; esac
  for s in tr va te; do
    [ -f "$REPO/data/${ds}_tgraph_${s}.pt" ] || [ "$DRY" = 1 ] \
      || { echo "missing data/${ds}_tgraph_${s}.pt" >&2; exit 1; }
  done
  for mode in $MODES; do
    trw=""; pre=rw_bert
    [ "$mode" = temporal ] && { trw="--trw"; pre=trw_bert; }
    for s in $PSEEDS; do
      if [ "$CTX_ONLY" = 1 ]; then
        ptag="_ps${s}"; fttag="_ps${s}_s0"; jv="_undirctx"        # the main runs' checkpoint and file names
      else
        ptag="${VARIANT}_ps${s}"; fttag="_s0"; jv="$VARIANT"      # _undir_ps1
      fi
      ck="$PRETRAINED/$mode/$ds/${pre}_${ds}${ptag}_${SIZE}.pt"
      ptname="pt_${mode}_${ds}${VARIANT}_ps${s}"

      ptid=""
      if [ "$CTX_ONLY" = 1 ]; then
        [ -f "$REPO/$ck" ] || [ "$DRY" = 1 ] || { echo "  no checkpoint $ck -- skipped" >&2; continue; }
      elif [ "$FORCE" = 0 ] && [ -f "$REPO/$ck" ]; then
        echo "  have      $ck"
      else
        clear=""; [ "$FORCE" = 1 ] && clear="rm -f $ck ${ck%.pt}-best.pt;"
        ptid=$(submit "$ptname" "$QUEUE_PT" "$WALL_PT" "" $clear \
          python pretrain.py --dataset "$ds" $trw --undirected-walks --size "$SIZE" \
                 --walk-len "$WL" --device 0 --seed "$s" --tag "$ptag" $(pt_args "$ds") $EXTRA_PRETRAIN) || continue
        echo "  pretrain  $ptname  ($ptid)"
        n_pt=$((n_pt + 1))
      fi

      for head in $HEADS; do
        case "$head" in lp|cls) ;; *) echo "unknown head '$head' (lp or cls)" >&2; exit 1 ;; esac
        for wl in $WLS; do
          out_dir="$RESULTS/$head-$mode/$ds"
          csv="$out_dir/${head}${ptag}_${SIZE}_wl${wl}${fttag}.csv"
          if [ "$FORCE" = 0 ] && [ -f "$REPO/$csv" ] && grep -q '^# BEST' "$REPO/$csv" 2>/dev/null; then
            echo "  have      $csv"; continue
          fi
          ftname="ft_${head}_${mode}_${ds}${jv}_ps${s}_wl${wl}"
          clear=""; [ "$FORCE" = 1 ] && clear="rm -f $csv;"
          guard="[ -f $ck ] || { echo 'no checkpoint $ck -- pretraining did not finish; skipping'; exit 0; };"
          ftid=$(submit "$ftname" "$QUEUE_FT" "$WALL_FT" "$ptid" $clear "$guard" \
            python "${head}_finetune.py" --dataset "$ds" $trw --undirected-walks \
                   --size "$SIZE" --device 0 --walk-len "$wl" --seed 0 --pretrain-tag "$ptag" \
                   --tag "$fttag" --out-dir "$out_dir" $EXTRA_FT) || continue
          echo "  finetune  $ftname  ($ftid${ptid:+, after $ptid})"
          n_ft=$((n_ft + 1))
        done
      done
    done
  done
done

echo
echo "$n_pt pretraining(s) and $n_ft fine-tune(s) $([ "$DRY" = 1 ] && echo listed || echo submitted); LSF logs in $LOGDIR/"
if [ "$CTX_ONLY" = 1 ]; then
  echo "results:  python show_main_results.py --root $RESULTS"
else
  echo "results:  python show_main_results.py --filter _undir      (kill all: bkill -J '*_undir_*' 0)"
fi
