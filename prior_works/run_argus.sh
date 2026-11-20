#!/usr/bin/env bash
# Submit every Argus baseline to LSF, one job per (experiment, seed), and return.
#
#   lanl            argus_test_lanl.py                           random LANL split
#   lanl_ts         argus_test_lanl_ts.py                        temporal, --cutoff 41
#   lanl_ts_dirty   argus_test_lanl_ts.py --cutoff $CUTOFF        attack edges IN training
#   lanl_ts_clean   argus_test_lanl_ts.py --cutoff $CUTOFF --scrub the same window, scrubbed
#   optc            argus_test_optc.py --dataset optc            random OpTC split
#   optc_argus      argus_test_optc.py --dataset optc-argus      Argus's temporal OpTC split
#   unsw            argus_test_unsw.py                           random UNSW split
#
# Each job is ONE training run at its own seed. All seeds of an experiment append to ONE file,
# <OUTDIR>/<experiment><TAG>.csv, one row per seed; the job logs go to <OUTDIR>/logs/. Nothing
# else is written. A (experiment, seed) whose row is already in that file is skipped.
#
# CUTOFF (default 168h = temporal_split.py's tr_days=7) lines the poisoning arms up with
# CyberGFM's dirtyts/cleants splits. Both arms score the identical test set.
#
#   ./run_argus.sh                                   # 7 experiments x 5 seeds = 35 jobs
#   ./run_argus.sh --list                            # what would be submitted
#   ./run_argus.sh --only optc,optc_argus
#   OUTDIR=../results/argus-paper-defaults ./run_argus.sh
#   EXTRA="--beta 0" TAG=_b0 ./run_argus.sh          # a variant, in its own files
#   PROFILE=code OUTDIR=../results/argus-code ./run_argus.sh
#                                                    # their released training recipe (below)
#
# PROFILE=code appends the flags that make every script train the way their classification.py
# does: one GRU layer, no L_DEC term, 100 epochs, early stopping at patience 3. The scripts'
# own defaults follow the PAPER where paper and code disagree (two GRU layers, beta > 0) and
# train to convergence (1000 epochs, patience 100).
#
# CUTOFF is the hour the poisoning arms train up to. It MUST match the CyberGFM dirtyts/cleants
# files or the two models are scored on different test sets: temporal_split.py builds them with
# tr_days=7, i.e. 168h (scrub_lanl.py reports the window it inferred). Set CUTOFF=172 only if
# those files are rebuilt at 7.1667 days.
#
#   python show_argus_results.py ../results/argus    # once the jobs finish
set -uo pipefail

SEEDS=${SEEDS:-5}
SEED0=${SEED0:-1}                # seeds SEED0 .. SEED0+SEEDS-1
OUTDIR=${OUTDIR:-results/argus}      # relative to prior_works/
TAG=${TAG:-""}                   # variant suffix on the results file, e.g. _b0
EXTRA=${EXTRA:-""}               # flags appended to every job
PROFILE=${PROFILE:-""}           # "code": their released recipe, see the header
CUTOFF=${CUTOFF:-168}            # poisoning-arm cutoff in hours; must match the CyberGFM files
case "$PROFILE" in
  code) EXTRA="--gru-layers 1 --beta 0 --epochs 100 --patience 3 $EXTRA" ;;
  "")   ;;
  *) echo "unknown PROFILE '$PROFILE' (use: code)" >&2; exit 1 ;;
esac

QUEUE=${QUEUE:-normal}
GPU_RES=${GPU_RES:-"num=1:mode=exclusive_process"}
WALL=${WALL:-"24:00"}
MEM_MB=${MEM_MB:-32000}
CORES=${CORES:-4}
PRELUDE=${PRELUDE:-"source ~/miniconda3/bin/activate gfm"}
REPO=${REPO:-$PWD}

DRY=0; LIST=0; FORCE=0; ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --list|-n) LIST=1 ;;
    --force)   FORCE=1 ;;
    --only)    ONLY="$2"; shift ;;
    --only=*)  ONLY="${1#*=}" ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

[ -f "$REPO/argus.py" ] || { echo "run this from prior_works/ (no argus.py here)" >&2; exit 1; }

NAMES=(lanl lanl_ts lanl_ts_dirty lanl_ts_clean optc optc_argus unsw)

spec() {  # name -> "script|flags|results file stem (what the script names its output)"
  case "$1" in
    lanl)          echo "argus_test_lanl.py|--bs 24|lanl14argus" ;;   # --bs 0 OOMs a 44 GB card
    lanl_ts)       echo "argus_test_lanl_ts.py||lanl14argus-dirtyts-41h" ;;
    lanl_ts_dirty) echo "argus_test_lanl_ts.py|--cutoff $CUTOFF|lanl14argus-dirtyts-${CUTOFF}h" ;;
    lanl_ts_clean) echo "argus_test_lanl_ts.py|--cutoff $CUTOFF --scrub|lanl14argus-cleants-${CUTOFF}h" ;;
    optc)          echo "argus_test_optc.py|--dataset optc|optc" ;;
    optc_argus)    echo "argus_test_optc.py|--dataset optc-argus|optc-argus" ;;
    unsw)          echo "argus_test_unsw.py||unsw" ;;
  esac
}

selected() { [ -z "$ONLY" ] || case ",$ONLY," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

done_seeds() {  # seeds already recorded in a results file, space separated
  [ -f "$1" ] || return 0
  python3 - "$1" <<'PY'
import csv, sys
with open(sys.argv[1], newline='') as f:
    print(' '.join(sorted({r['seed'] for r in csv.DictReader(f) if r.get('seed')})))
PY
}

[ "$LIST" = 1 ] || [ "$DRY" = 1 ] || command -v bsub >/dev/null \
  || { echo "bsub not found; run this on a login node" >&2; exit 1; }
mkdir -p "$OUTDIR/logs"

submitted=0; skipped=0; failed=0
for n in "${NAMES[@]}"; do
  selected "$n" || continue
  IFS='|' read -r script flags stem <<< "$(spec "$n")"
  csv="$OUTDIR/${stem}${TAG}.csv"
  have=" $(done_seeds "$csv") "

  for s in $(seq "$SEED0" $((SEED0 + SEEDS - 1))); do
    if [ "$FORCE" = 0 ] && [[ "$have" == *" $s "* ]]; then
      skipped=$((skipped + 1)); continue
    fi
    cmd="python $script --runs 1 --device 0 --seed $s --out-dir '$OUTDIR' --tag '$TAG' $flags $EXTRA"
    if [ "$LIST" = 1 ]; then echo "  $n s$s: $cmd"; continue; fi

    jname="argus_${n}${TAG}_s${s}"
    log="$OUTDIR/logs/${jname}.out"
    # The job body goes to bsub on stdin; no script file is left behind.
    body=$(cat <<EOF
#!/usr/bin/env bash
cd $(printf %q "$REPO")
$PRELUDE
echo "host: \$(hostname)  gpus: \${CUDA_VISIBLE_DEVICES:-unset}  started: \$(date -Is)"
start=\$(date +%s)
$cmd
rc=\$?
el=\$(( \$(date +%s) - start ))
printf '\n=== %s: exit %d after %d:%02d:%02d ===\n' '$jname' "\$rc" \$((el/3600)) \$(((el%3600)/60)) \$((el%60))
exit \$rc
EOF
)
    args=(-J "$jname" -q "$QUEUE" -gpu "$GPU_RES" -n "$CORES" -W "$WALL"
          -R "rusage[mem=$MEM_MB]" -o "$log" -e "$log")
    if [ "$DRY" = 1 ]; then
      echo "  bsub ${args[*]}  <<  $cmd"; submitted=$((submitted + 1)); continue
    fi
    id=$(bsub "${args[@]}" <<< "$body" 2>&1 | sed -n 's/^Job <\([0-9]*\)>.*/\1/p')
    if [ -n "$id" ]; then printf '  %-30s %s\n' "$jname" "$id"; submitted=$((submitted + 1))
    else echo "  SUBMIT FAILED $jname" >&2; failed=$((failed + 1)); fi
  done
done

[ "$LIST" = 1 ] && exit 0
echo
echo "submitted $submitted, skipped $skipped (seed already in its results file), failed $failed"
echo "results : $OUTDIR/<experiment>${TAG}.csv     logs: $OUTDIR/logs/"
echo "show    : python show_argus_results.py $OUTDIR"
[ "$failed" = 0 ]