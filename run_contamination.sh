#!/usr/bin/env bash
# Training-window contamination experiment (pre-review item 1): dirty vs. clean training windows
# on LANL (first 8, 9, 10, 11, 12 days) and OpTC (through the first attack day), for CyberGFM
# (static and temporal), Argus, Euler and PIKACHU, five seeds each.
#
# Two steps, because the splits must exist before anything can train on them:
#
#   ./run_contamination.sh --splits      # 1. one job builds every split (contamination_splits.py)
#   ./run_contamination.sh --dry-run     # 2. once it has finished: check, then
#   ./run_contamination.sh               #    submit CyberGFM, Argus, Euler and PIKACHU
#
# Datasets (data/<name>_tgraph_{tr,va,te}.pt; test sets identical within each dirty/clean pair):
#   lanl14argus-dirtyts{8..12}d  lanl14argus-cleants{8..12}d
#   optc-argus-dirty24h          optc-argus-clean24h
# The split job prints, for every pair, the attack edges in train / val / test and how many test
# attack edges repeat a (src, dst) attack pair from the training window (also saved to
# data/contamination_report.json) -- check that each test window still has attack edges.
#
# Each method runs exactly as in its main experiments, so the rows sit next to the 7-day ones:
#   CyberGFM  run_all_hpc.sh, static + temporal walks, pretraining-matched walk length (4 on LANL,
#             64 on OpTC), LP head on LANL and CLS on OpTC          -> results/{lp,cls}-<mode>/<dataset>/
#   Argus     argus_test_lanl_ts.py --cutoff 24*D [--scrub]  (its own per-hour validation split,
#             as at 168 h); argus_test_optc.py --dataset <name>  -> results/argus/
#   Euler, PIKACHU  prior_works/run_baselines.sh                    -> results/baselines/<model>/
# Every runner skips cells that already have results, so re-running submits only what is missing.
#
#   LANL_DAYS="8 10 12" ./run_contamination.sh
#   OPTC_HOURS="24 48" ./run_contamination.sh --splits      # more OpTC windows (then rerun step 2)
#   PARTS="cybergfm" ./run_contamination.sh                  # one family only (cybergfm argus baselines)
#   PARTS=cybergfm MODES=temporal ./run_contamination.sh     # CyberGFM temporal walks only
#
# Run from the repo root on the login node; every step is a bsub job.
set -uo pipefail

LANL_DAYS=${LANL_DAYS:-"8 9 10 11 12"}
OPTC_BASE=${OPTC_BASE:-optc-argus}
OPTC_HOURS=${OPTC_HOURS:-"24"}
PARTS=${PARTS:-"cybergfm argus baselines"}
MODES=${MODES:-"static temporal"}           # CyberGFM walk modes
BASELINES=${BASELINES:-"euler,pikachu"}
SEEDS=${SEEDS:-"1 2 3 4 5"}

QUEUE=${QUEUE:-normal}
GPU_RES=${GPU_RES:-"num=1:mode=exclusive_process"}
CORES=${CORES:-4}
PRELUDE=${PRELUDE:-"source ~/miniconda3/bin/activate gfm"}
LOGDIR=${LOGDIR:-logs/contamination}
REPO=${REPO:-$PWD}

SPLITS=0; DRY=0; PASS=()
for a in "$@"; do
  case "$a" in
    --splits) SPLITS=1 ;;
    --dry-run) DRY=1; PASS+=(--dry-run) ;;
    --force) PASS+=(--force) ;;
    *) echo "unknown option: $a" >&2; exit 1 ;;
  esac
done
[ -f "$REPO/pretrain.py" ] || { echo "run this from the repo root (no pretrain.py here)" >&2; exit 1; }
[ "$DRY" = 1 ] || command -v bsub >/dev/null || { echo "bsub not found; run this on a login node" >&2; exit 1; }
mkdir -p "$LOGDIR"

submit() {  # submit <jobname> <wall> <mem_mb> <cwd> <command>; prints the job id
  local name="$1" wall="$2" mem="$3" cwd="$4" cmd="$5"
  local jobsh="$LOGDIR/${name}.sh" out="$LOGDIR/${name}.out"
  {
    echo "#!/usr/bin/env bash"
    printf 'cd %q\n' "$cwd"
    [ -n "$PRELUDE" ] && echo "$PRELUDE"
    echo 'echo "host: $(hostname)  gpus: ${CUDA_VISIBLE_DEVICES:-unset}  started: $(date -Is)"'
    echo "$cmd"
  } > "$jobsh"
  local args=(-J "$name" -q "$QUEUE" -gpu "$GPU_RES" -n "$CORES" -W "$wall"
              -R "rusage[mem=$mem]" -o "$out" -e "$out")
  if [ "$DRY" = 1 ]; then
    echo "  bsub ${args[*]} < $jobsh"; echo "      $cmd"; return 0
  fi
  local res id
  res=$(bsub "${args[@]}" < "$jobsh" 2>&1)
  id=$(printf '%s\n' "$res" | sed -n 's/^Job <\([0-9]*\)>.*/\1/p')
  if [ -z "$id" ]; then
    echo "  SUBMIT FAILED $name -- bsub said:" >&2; printf '%s\n' "$res" | sed 's/^/    /' >&2; return 1
  fi
  printf '  %-44s %s\n' "$name" "$id"
}

lanl_sets=(); optc_sets=()
for d in $LANL_DAYS; do lanl_sets+=("lanl14argus-dirtyts${d}d" "lanl14argus-cleants${d}d"); done
for h in $OPTC_HOURS; do optc_sets+=("${OPTC_BASE}-dirty${h}h" "${OPTC_BASE}-clean${h}h"); done
all_sets=("${lanl_sets[@]}" "${optc_sets[@]}")

# ---- step 1: build the splits -------------------------------------------------------------------
if [ "$SPLITS" = 1 ]; then
  [ -f "$REPO/contamination_splits.py" ] || { echo "put contamination_splits.py in the repo root" >&2; exit 1; }
  # A GPU slot is requested only because the queue expects one; the job runs no model.
  submit "contam_splits" "02:00" 64000 "$REPO" \
    "python contamination_splits.py --lanl-days $LANL_DAYS --optc-base $OPTC_BASE --optc-hours $OPTC_HOURS" || exit 1
  [ "$DRY" = 1 ] || echo "when it finishes: tail -40 $LOGDIR/contam_splits.out, then ./run_contamination.sh"
  exit 0
fi

# ---- step 2: the experiments --------------------------------------------------------------------
missing=0
for ds in "${all_sets[@]}"; do
  for s in tr va te; do
    [ -f "$REPO/data/${ds}_tgraph_${s}.pt" ] || { echo "  missing data/${ds}_tgraph_${s}.pt" >&2; missing=1; }
  done
done
if [ "$missing" = 1 ] && [ "$DRY" = 0 ]; then
  echo "splits not built yet -- run ./run_contamination.sh --splits and wait for it to finish" >&2; exit 1
fi

for part in $PARTS; do
  case "$part" in
    cybergfm)
      echo "=== CyberGFM ($MODES, matched walk length, seeds: $SEEDS) ==="
      DATASETS="${all_sets[*]}" MODES="$MODES" WLS=match PSEEDS="$SEEDS" \
      QUEUE="$QUEUE" LOGDIR="$LOGDIR/cybergfm" PRELUDE="$PRELUDE" \
        bash "$REPO/run_all_hpc.sh" "${PASS[@]}" ;;
    argus)
      echo "=== Argus ==="
      for d in $LANL_DAYS; do
        cut=$((24 * d))
        for s in $SEEDS; do
          for arm in dirty clean; do
            scrub=""; [ "$arm" = clean ] && scrub="--scrub"
            exp="lanl14argus-${arm}ts-${cut}h"      # argus_test_lanl_ts.py's experiment name
            csv="$REPO/results/argus/${exp}.csv"
            if [ -f "$csv" ] && awk -F, -v s="$s" 'NR==1{for(i=1;i<=NF;i++)if($i=="seed")c=i;next} c&&$c==s{f=1} END{exit !f}' "$csv"; then
              echo "  have      $exp seed $s"; continue
            fi
            submit "argus_lanl_${arm}${d}d_s${s}" "24:00" 32000 "$REPO/prior_works" \
              "python argus_test_lanl_ts.py --device 0 --seed $s --cutoff $cut $scrub --out-dir ../results/argus"
          done
        done
      done
      for ds in "${optc_sets[@]}"; do
        for s in $SEEDS; do
          csv="$REPO/results/argus/${ds}.csv"
          if [ -f "$csv" ] && awk -F, -v s="$s" 'NR==1{for(i=1;i<=NF;i++)if($i=="seed")c=i;next} c&&$c==s{f=1} END{exit !f}' "$csv"; then
            echo "  have      $ds seed $s"; continue
          fi
          submit "argus_${ds}_s${s}" "24:00" 32000 "$REPO/prior_works" \
            "python argus_test_optc.py --dataset $ds --device 0 --seed $s --out-dir ../results/argus"
        done
      done ;;
    baselines)
      echo "=== $BASELINES ==="
      # run_baselines.sh takes a count and a first seed; the seeds must be consecutive
      n_seeds=$(echo $SEEDS | wc -w); seed0=$(echo $SEEDS | cut -d' ' -f1)
      (cd "$REPO/prior_works" && SEEDS=$n_seeds SEED0=$seed0 \
        QUEUE="$QUEUE" PRELUDE="$PRELUDE" \
        bash run_baselines.sh --models "$BASELINES" --datasets "$(IFS=,; echo "${all_sets[*]}")" "${PASS[@]}") ;;
    *) echo "unknown part '$part' (cybergfm, argus, baselines)" >&2; exit 1 ;;
  esac
  echo
done