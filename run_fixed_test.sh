#!/usr/bin/env bash
# Fixed-test-set training-window experiment on LANL: does more training history help when the test
# set does not change? One test set (every edge from day TEST_DAY on) and one training window per
# K in WINDOWS: the last K days before the cutoff. See fixed_test_splits.py.
#
# Two steps, because the splits must exist before anything can train on them:
#
#   ./run_fixed_test.sh                     # 1. one job builds every split (fixed_test_splits.py)
#   ./run_fixed_test.sh --train --dry-run   # 2. once it has finished: check, then
#   ./run_fixed_test.sh --train             #    submit CyberGFM on them
#
# Datasets (data/<name>_tgraph_{tr,va,te}.pt; the test file is the same edges in every one):
#   lanl14argus-fixed12-clean{1,2,4,7,12}d          with the defaults below
#   ...-dirty{K}d with ARMS="clean dirty";  ...-tail with VAL=tail
# The split job prints edges and attack edges per day, and for every split its sizes and how much
# of the test set the window has already seen (also saved to data/fixed_test_report.json).
#
# Step 2 runs CyberGFM exactly as in the main experiments, through run_all_hpc.sh: static and
# temporal walks, walk length 4, LP head, five pretraining seeds
#                                            -> results/lp-<mode>/<dataset>/
# It skips cells that already have results, so re-running submits only what is missing.
#
#   WINDOWS="2 4 8" ./run_fixed_test.sh                 # other windows (same again with --train)
#   ARMS="clean dirty" ./run_fixed_test.sh              # also the unscrubbed arm
#   VAL=tail WINDOWS=12 ./run_fixed_test.sh             # rebuild the 12-day split and check it
#   TEST_DAY=9 WINDOWS="1 2 4 9" ./run_fixed_test.sh    # a different fixed test set
#   MODES=static SEEDS="1 2 3" ./run_fixed_test.sh --train
#   QUEUE_PT=h100 ./run_fixed_test.sh --train           # run_all_hpc.sh's own settings pass through
#
# Run from the repo root on the login node; every step is a bsub job.
set -uo pipefail

TEST_DAY=${TEST_DAY:-12}
WINDOWS=${WINDOWS:-"1 2 3 4 5 6 7 8 9 10 11 12"}            # days of training history before the cutoff
ARMS=${ARMS:-"dirty"}                       # clean, dirty, or both
VAL=${VAL:-random}                          # random | tail (see fixed_test_splits.py)
MODES=${MODES:-"static temporal"}           # CyberGFM walk modes
SEEDS=${SEEDS:-"1 2 3 4 5"}

QUEUE=${QUEUE:-normal}
GPU_RES=${GPU_RES-"num=1:mode=exclusive_process"}   # GPU_RES="" submits the split job without a GPU
CORES=${CORES:-4}
WALL=${WALL:-"02:00"}                       # split job only
MEM_MB=${MEM_MB:-32000}                     # split job only
PRELUDE=${PRELUDE:-"source ~/miniconda3/bin/activate gfm"}
LOGDIR=${LOGDIR:-logs/fixed_test}
REPO=${REPO:-$PWD}

TRAIN=0; DRY=0; PASS=()
for a in "$@"; do
  case "$a" in
    --train) TRAIN=1 ;;
    --dry-run) DRY=1; PASS+=(--dry-run) ;;
    --force) PASS+=(--force) ;;
    *) echo "unknown option: $a" >&2; exit 1 ;;
  esac
done
[ -f "$REPO/pretrain.py" ] || { echo "run this from the repo root (no pretrain.py here)" >&2; exit 1; }
[ "$DRY" = 1 ] || command -v bsub >/dev/null || { echo "bsub not found; run this on a login node" >&2; exit 1; }
case "$VAL" in random|tail) ;; *) echo "VAL must be random or tail, not '$VAL'" >&2; exit 1 ;; esac
for arm in $ARMS; do
  case "$arm" in clean|dirty) ;; *) echo "ARMS takes clean and/or dirty, not '$arm'" >&2; exit 1 ;; esac
done
for k in $WINDOWS; do
  case "$k" in ''|*[!0-9]*) echo "WINDOWS takes whole days, not '$k'" >&2; exit 1 ;; esac
  [ "$k" -ge 1 ] && [ "$k" -le "$TEST_DAY" ] || { echo "window $k is outside 1..$TEST_DAY days" >&2; exit 1; }
done
mkdir -p "$LOGDIR"

suffix=""; [ "$VAL" = tail ] && suffix="-tail"
sets=()
for arm in $ARMS; do
  for k in $WINDOWS; do sets+=("lanl14argus-fixed${TEST_DAY}-${arm}${k}d${suffix}"); done
done
echo "datasets: ${sets[*]}"

submit() {  # submit <jobname> <wall> <mem_mb> <cwd> <command>; prints the job id
  local name="$1" wall="$2" mem="$3" cwd="$4" cmd="$5"
  local jobsh="$LOGDIR/${name}.sh" out="$LOGDIR/${name}.out"
  {
    echo "#!/usr/bin/env bash"
    printf 'cd %q\n' "$cwd"
    [ -n "$PRELUDE" ] && echo "$PRELUDE"
    echo 'echo "host: $(hostname)  started: $(date -Is)"'
    echo "$cmd"
  } > "$jobsh"
  local args=(-J "$name" -q "$QUEUE" -n "$CORES" -W "$wall" -R "rusage[mem=$mem]" -o "$out" -e "$out")
  [ -n "$GPU_RES" ] && args+=(-gpu "$GPU_RES")
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

# ---- step 1: build the splits -------------------------------------------------------------------
if [ "$TRAIN" = 0 ]; then
  for f in fixed_test_splits.py contamination_splits.py data/lanl14argus_tgraph_csr.pt; do
    [ -f "$REPO/$f" ] || { echo "missing $f (fixed_test_splits.py goes in the repo root)" >&2; exit 1; }
  done
  name="fixed${TEST_DAY}_splits${suffix}"
  # A GPU slot is requested only because the queue expects one; the job runs no model.
  submit "$name" "$WALL" "$MEM_MB" "$REPO" \
    "python fixed_test_splits.py --test-day $TEST_DAY --windows $WINDOWS --arms $ARMS --val $VAL" || exit 1
  [ "$DRY" = 1 ] || echo "when it finishes: cat $LOGDIR/${name}.out, then ./run_fixed_test.sh --train --dry-run"
  exit 0
fi

# ---- step 2: CyberGFM on the splits -------------------------------------------------------------
missing=0
for ds in "${sets[@]}"; do
  for s in tr va te; do
    [ -f "$REPO/data/${ds}_tgraph_${s}.pt" ] || { echo "  missing data/${ds}_tgraph_${s}.pt" >&2; missing=1; }
  done
done
if [ "$missing" = 1 ] && [ "$DRY" = 0 ]; then
  echo "splits not built yet -- run ./run_fixed_test.sh and wait for it to finish" >&2; exit 1
fi

echo "=== CyberGFM ($MODES, walk length 4, seeds: $SEEDS) ==="
DATASETS="${sets[*]}" MODES="$MODES" WLS=match PSEEDS="$SEEDS" \
QUEUE="$QUEUE" LOGDIR="$LOGDIR/cybergfm" PRELUDE="$PRELUDE" \
  bash "$REPO/run_all_hpc.sh" ${PASS[@]+"${PASS[@]}"}