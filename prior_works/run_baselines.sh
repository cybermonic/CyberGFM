#!/usr/bin/env bash
# Submit every non-Argus baseline on every dataset to LSF: one job per (model, dataset, seed).
#
#   models    sage gin euler pikachu n2v          (gnn_test.py [--gin], euler_test.py, ...)
#   datasets  lanl14argus lanl14argus-ts lanl14argus-dirtyts lanl14argus-cleants
#             optc optc-argus unsw                (CyberGFM's own split files in ../data)
#
# NOTHING RUNS ON THE LOGIN NODE. This script only writes job scripts and calls bsub; every
# Python process -- the smoke test included -- runs inside a scheduler job. (The one thing it
# reads locally is the seed column of existing results CSVs, with awk, to skip finished seeds.)
#
# All seeds of a (model, dataset) append to ONE file, <OUTDIR>/<model>/<dataset><TAG>.csv, one
# row per seed. Each job's script and log go to <OUTDIR>/logs/<jobname>.{sh,out}; a single cell
# can be re-submitted by hand with `bsub < <OUTDIR>/logs/<jobname>.sh` plus the same options.
#
#   ./run_baselines.sh --smoke                 # ONE short bsub job: every model on two toy splits
#   ./run_baselines.sh --list                  # what would be submitted
#   ./run_baselines.sh --dry-run               # the bsub commands, nothing submitted
#   ./run_baselines.sh                         # 5 models x 7 datasets x 5 seeds = 175 jobs
#   ./run_baselines.sh --models euler,pikachu --datasets optc,optc-argus
#   ./run_baselines.sh --force                 # resubmit seeds that already have a row
#   EXTRA_euler="--undirected" TAG=_und ./run_baselines.sh --models euler   # a variant
#
# Run from prior_works/ on the submit host.
set -uo pipefail

SEEDS=${SEEDS:-5}
SEED0=${SEED0:-1}
OUTDIR=${OUTDIR:-../results/baselines}
TAG=${TAG:-""}
EXTRA=${EXTRA:-""}               # appended to every job; EXTRA_<model> to that model's jobs
DATA=${DATA:-../data}
QUEUE=${QUEUE:-normal}
SMOKE_QUEUE=${SMOKE_QUEUE:-$QUEUE}
SMOKE_WALL=${SMOKE_WALL:-"01:00"}
GPU_RES=${GPU_RES:-"num=1:mode=exclusive_process"}
MEM_MB=${MEM_MB:-48000}          # PIKACHU holds T x N x 128 skip-gram embeddings in host RAM
CORES=${CORES:-4}
PRELUDE=${PRELUDE:-"source ~/miniconda3/bin/activate gfm"}
REPO=${REPO:-$PWD}

MODELS="sage gin euler pikachu n2v"
DATASETS="lanl14argus lanl14argus-ts lanl14argus-dirtyts lanl14argus-cleants optc optc-argus unsw"

DRY=0; LIST=0; FORCE=0; SMOKE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --list|-n) LIST=1 ;;
    --force)   FORCE=1 ;;
    --smoke)   SMOKE=1 ;;
    --models)  MODELS="${2//,/ }"; shift ;;
    --models=*) MODELS="${1#*=}"; MODELS="${MODELS//,/ }" ;;
    --datasets)  DATASETS="${2//,/ }"; shift ;;
    --datasets=*) DATASETS="${1#*=}"; DATASETS="${DATASETS//,/ }" ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

[ -f "$REPO/baseline_common.py" ] || { echo "run this from prior_works/ (no baseline_common.py here)" >&2; exit 1; }
[ "$LIST" = 1 ] || [ "$DRY" = 1 ] || command -v bsub >/dev/null \
  || { echo "bsub not found; run this on a login node" >&2; exit 1; }

script() { case "$1" in sage|gin) echo gnn_test.py ;; *) echo "${1}_test.py" ;; esac; }
flags()  { case "$1" in gin) echo "--gin" ;; *) echo "" ;; esac; }
wall()   { case "$1" in pikachu) echo "48:00" ;; euler) echo "24:00" ;; *) echo "12:00" ;; esac; }
smoke_flags() { case "$1" in
  sage|gin) echo "--epochs 3" ;;
  euler)    echo "--epochs 2 --bs 8" ;;
  pikachu)  echo "--w2v-epochs 1 --walk-len 20 --ae-epochs 2 --anom-epochs 2" ;;
  n2v)      echo "--n2v-epochs 1 --walk-len 20 --epochs 2" ;;
esac; }
extra_for() { local v="EXTRA_$1"; echo "${!v-}"; }

for m in $MODELS; do
  [ -f "$REPO/$(script "$m")" ] || { echo "unknown model '$m' (no $(script "$m"))" >&2; exit 1; }
done
mkdir -p "$OUTDIR/logs"

# submit <jobname> <queue> <wall> <mem_mb> <body-file>; prints the job id
submit() {
  local name="$1" queue="$2" wall="$3" mem="$4" body="$5"
  local log="$OUTDIR/logs/${name}.out"
  local args=(-J "$name" -q "$queue" -gpu "$GPU_RES" -n "$CORES" -W "$wall"
              -R "rusage[mem=$mem]" -o "$log" -e "$log")
  if [ "$DRY" = 1 ]; then
    echo "  bsub ${args[*]} < $body" >&2; echo DRYRUN; return 0
  fi
  local id
  id=$(bsub "${args[@]}" < "$body" 2>&1 | sed -n 's/^Job <\([0-9]*\)>.*/\1/p')
  [ -n "$id" ] || { echo "  SUBMIT FAILED $name" >&2; return 1; }
  echo "$id"
}

# job_header <jobname>: the common top of every job script
job_header() {
  echo "#!/usr/bin/env bash"
  printf 'cd %q\n' "$REPO"
  [ -n "$PRELUDE" ] && echo "$PRELUDE"
  echo 'echo "host: $(hostname)  gpus: ${CUDA_VISIBLE_DEVICES:-unset}  started: $(date -Is)"'
  echo 'start=$(date +%s)'
}
job_footer() {  # job_footer <jobname> <exit-status expression>
  echo 'el=$(( $(date +%s) - start ))'
  printf "printf '\\\\n=== %%s: exit %%d after %%d:%%02d:%%02d ===\\\\n' %q \"%s\" \$((el/3600)) \$(((el%%3600)/60)) \$((el%%60))\n" "$1" "$2"
  echo "exit $2"
}

# ---- smoke test: ONE short job that runs every model on a random and a temporal toy split ----
if [ "$SMOKE" = 1 ]; then
  name="bl_smoke${TAG}"
  body="$OUTDIR/logs/${name}.sh"
  {
    job_header
    echo 'rm -rf tmp/smoke tmp/toydata'
    echo 'python make_toy_data.py --out tmp/toydata || exit 1'
    echo 'fails=0; summary=""'
    echo 'run() {  # run <model> <dataset> <command...>'
    echo '  local m=$1 ds=$2; shift 2'
    echo '  echo; echo "===== $m on $ds: $*"'
    echo '  if "$@"; then summary+="  ok      $m $ds"$'"'"'\n'"'"
    echo '  else summary+="  FAILED  $m $ds"$'"'"'\n'"'"'; fails=$((fails + 1)); fi'
    echo '}'
    for m in $MODELS; do
      for ds in toy toy-ts; do
        echo "run $m $ds python $(script "$m") $(flags "$m") --dataset $ds --data tmp/toydata" \
             "--device 0 --seed 1 --out-dir tmp/smoke $(smoke_flags "$m")"
      done
    done
    echo 'echo; echo "===== smoke summary"; printf "%s" "$summary"'
    echo 'echo "$fails failure(s); result rows in tmp/smoke/<model>/"'
    echo 'rc=$fails'
    job_footer "$name" '$rc'
  } > "$body"
  [ "$LIST" = 1 ] && { echo "  $name: $body"; exit 0; }
  id=$(submit "$name" "$SMOKE_QUEUE" "$SMOKE_WALL" 16000 "$body") || exit 1
  [ "$DRY" = 1 ] && exit 0
  echo "smoke job $id submitted. Follow it with:  bpeek -f $id"
  echo "the summary is at the end of $OUTDIR/logs/${name}.out"
  exit 0
fi

# ---- the matrix -------------------------------------------------------------------------------
done_seeds() {  # seeds already in a results file's seed column, space separated (awk, no python)
  [ -f "$1" ] || return 0
  awk -F, 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == "seed") c = i; next }
           c && $c != "" { print $c }' "$1" | sort -u | tr '\n' ' '
}

submitted=0; skipped=0; failed=0
for m in $MODELS; do
  for ds in $DATASETS; do
    [ -f "$DATA/${ds}_tgraph_tr.pt" ] || [ "$LIST" = 1 ] || [ "$DRY" = 1 ] \
      || { echo "  skipping $m/$ds: no $DATA/${ds}_tgraph_tr.pt" >&2; continue; }
    csv="$OUTDIR/$m/${ds}${TAG}.csv"
    have=" $(done_seeds "$csv") "
    for s in $(seq "$SEED0" $((SEED0 + SEEDS - 1))); do
      if [ "$FORCE" = 0 ] && [[ "$have" == *" $s "* ]]; then
        skipped=$((skipped + 1)); continue
      fi
      name="bl_${m}_${ds}${TAG}_s${s}"
      cmd="python $(script "$m") $(flags "$m") --dataset $ds --data '$DATA' --device 0 --seed $s --out-dir '$OUTDIR' --tag '$TAG' $EXTRA $(extra_for "$m")"
      if [ "$LIST" = 1 ]; then echo "  $name: $cmd"; submitted=$((submitted + 1)); continue; fi

      body="$OUTDIR/logs/${name}.sh"
      { job_header; echo "$cmd"; echo 'rc=$?'; job_footer "$name" '$rc'; } > "$body"
      if id=$(submit "$name" "$QUEUE" "$(wall "$m")" "$MEM_MB" "$body"); then
        [ "$DRY" = 1 ] || printf '  %-40s %s\n' "$name" "$id"
        submitted=$((submitted + 1))
      else
        failed=$((failed + 1))
      fi
    done
  done
done

echo
[ "$LIST" = 1 ] && { echo "$submitted job(s) outstanding, $skipped already done"; exit 0; }
echo "submitted $submitted, skipped $skipped (seed already in its results file), failed $failed"
echo "results : $OUTDIR/<model>/<dataset>${TAG}.csv     logs: $OUTDIR/logs/"
echo "show    : python show_baseline_results.py $OUTDIR    (in a job, or anywhere -- it only reads CSVs)"