#!/usr/bin/env bash
# Baselines for the fixed-test-set training-window experiment on LANL (run_fixed_test.sh is the
# CyberGFM side): Argus, Euler and PIKACHU on the same training windows and the same test set
# (every edge from day TEST_DAY on), five seeds each.
#
#   ./run_fixed_test_baselines.sh --dry-run     # check, then
#   ./run_fixed_test_baselines.sh               # submit everything
#
#   Argus           prior_works/argus_test_lanl_fixed.py --test-day T --window K. It reads the raw
#                   CSR, so it does not need the split files, and keeps its own per-hour
#                   validation holdout, as at every other cutoff
#                                        -> results/argus/lanl14argus-fixed{T}-{arm}{K}d.csv
#   Euler, PIKACHU  prior_works/run_baselines.sh on the split files run_fixed_test.sh builds
#                   (data/lanl14argus-fixed{T}-{arm}{K}d_tgraph_{tr,va,te}.pt), so build those
#                   first                -> results/baselines/<model>/<dataset>.csv
# Every part skips seeds that already have a row, so re-running submits only what is missing.
#
#   WINDOWS="2 4 8" ./run_fixed_test_baselines.sh            # other windows
#   ARMS="clean dirty" ./run_fixed_test_baselines.sh         # also the scrubbed arm
#   PARTS=argus ./run_fixed_test_baselines.sh                # one family only (argus baselines)
#   BASELINES=euler ./run_fixed_test_baselines.sh            # Euler without PIKACHU
#   SEEDS="1 2 3" QUEUE=long ./run_fixed_test_baselines.sh
#
# Keep TEST_DAY, WINDOWS, ARMS and VAL the same as for run_fixed_test.sh. VAL only names the
# split files Euler and PIKACHU read (the -tail suffix); Argus is unaffected by it.
# Run from the repo root on the login node; every job is a bsub job.
set -uo pipefail

TEST_DAY=${TEST_DAY:-12}
WINDOWS=${WINDOWS:-"1 2 3 4 5 6 7 8 9 10 11 12"}            # days of training history before the cutoff
ARMS=${ARMS:-"dirty"}                       # dirty, clean, or both
VAL=${VAL:-random}                          # random | tail, as the split files were built
PARTS=${PARTS:-"argus baselines"}
BASELINES=${BASELINES:-"euler,pikachu"}
SEEDS=${SEEDS:-"1 2 3 4 5"}

QUEUE=${QUEUE:-normal}
GPU_RES=${GPU_RES:-"num=1:mode=exclusive_process"}
CORES=${CORES:-4}
WALL_ARGUS=${WALL_ARGUS:-"24:00"}
MEM_ARGUS=${MEM_ARGUS:-32000}
PRELUDE=${PRELUDE:-"source ~/miniconda3/bin/activate gfm"}
LOGDIR=${LOGDIR:-logs/fixed_test}
REPO=${REPO:-$PWD}

DRY=0; FORCE=0; PASS=()
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1; PASS+=(--dry-run) ;;
    --force) FORCE=1; PASS+=(--force) ;;
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
for part in $PARTS; do
  case "$part" in argus|baselines) ;; *) echo "unknown part '$part' (argus, baselines)" >&2; exit 1 ;; esac
done
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

has_seed() {  # has_seed <csv> <seed>: is there a row for this seed already? (awk, no python)
  [ -f "$1" ] && awk -F, -v s="$2" 'NR==1{for(i=1;i<=NF;i++)if($i=="seed")c=i;next} c&&$c==s{f=1} END{exit !f}' "$1"
}

suffix=""; [ "$VAL" = tail ] && suffix="-tail"
sets=()
for arm in $ARMS; do
  for k in $WINDOWS; do sets+=("lanl14argus-fixed${TEST_DAY}-${arm}${k}d${suffix}"); done
done

# Check every part before submitting any of it, so a missing file never leaves half the jobs
# queued (a second run would then submit those again: the skip below only sees finished rows).
ok=1; need_splits=0
for part in $PARTS; do
  case "$part" in
    argus)
      for f in prior_works/argus_test_lanl_fixed.py data/lanl14argus_tgraph_raw_csr.pt; do
        [ -f "$REPO/$f" ] || { echo "  missing $f" >&2; ok=0; }
      done ;;
    baselines)
      [ -f "$REPO/prior_works/run_baselines.sh" ] || { echo "  missing prior_works/run_baselines.sh" >&2; ok=0; }
      for ds in "${sets[@]}"; do
        for s in tr va te; do
          [ -f "$REPO/data/${ds}_tgraph_${s}.pt" ] || { echo "  missing data/${ds}_tgraph_${s}.pt" >&2; ok=0; need_splits=1; }
        done
      done ;;
  esac
done
if [ "$ok" = 0 ] && [ "$DRY" = 0 ]; then
  [ "$need_splits" = 1 ] && echo "splits not built yet -- run ARMS=\"$ARMS\" ./run_fixed_test.sh and wait for it to finish" >&2
  echo "nothing submitted" >&2; exit 1
fi

for part in $PARTS; do
  case "$part" in
    argus)
      echo "=== Argus (seeds: $SEEDS) ==="
      n_sub=0; n_have=0
      for arm in $ARMS; do
        scrub=""; [ "$arm" = clean ] && scrub=" --scrub"
        for k in $WINDOWS; do
          exp="lanl14argus-fixed${TEST_DAY}-${arm}${k}d"      # argus_test_lanl_fixed.py's experiment name
          for s in $SEEDS; do
            if [ "$FORCE" = 0 ] && has_seed "$REPO/results/argus/${exp}.csv" "$s"; then
              n_have=$((n_have + 1)); continue
            fi
            submit "argus_fixed${TEST_DAY}_${arm}${k}d_s${s}" "$WALL_ARGUS" "$MEM_ARGUS" "$REPO/prior_works" \
              "python argus_test_lanl_fixed.py --device 0 --seed $s --test-day $TEST_DAY --window $k$scrub --out-dir ../results/argus" \
              && n_sub=$((n_sub + 1))
          done
        done
      done
      did=submitted; [ "$DRY" = 1 ] && did="would be submitted"
      echo "Argus: $n_sub $did, $n_have already have a row in results/argus/" ;;
    baselines)
      echo "=== $BASELINES (seeds: $SEEDS) ==="
      # run_baselines.sh takes a count and a first seed; the seeds must be consecutive
      n_seeds=$(echo $SEEDS | wc -w); seed0=$(echo $SEEDS | cut -d' ' -f1)
      (cd "$REPO/prior_works" && SEEDS=$n_seeds SEED0=$seed0 REPO="$REPO/prior_works" \
        QUEUE="$QUEUE" PRELUDE="$PRELUDE" \
        bash run_baselines.sh --models "$BASELINES" --datasets "$(IFS=,; echo "${sets[*]}")" ${PASS[@]+"${PASS[@]}"}) ;;
  esac
  echo
done