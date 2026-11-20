#!/usr/bin/env bash
# Fine-tuning speed test on the three random splits (ft_speedtest.py): ONE job that runs one
# epoch of fine-tuning per dataset and batch-size setting, one after another on the same GPU so
# the timings are comparable, and ends by printing the report (time per training epoch, per
# evaluation and per scoring pass, and the extrapolated full 10-epoch run).
#
#   ./run_ft_speedtest.sh --dry-run     # show the job, submit nothing
#   ./run_ft_speedtest.sh               # submit it
#   python ft_speedtest.py --report     # afterwards; it only reads speedtest/ft_speedtest.csv
#
# GRID is a list of mini-bs:eval-bs pairs; "-" keeps the fine-tuning script's own value. The first
# pair should stay "-:-": it is the baseline the others are compared with. Training batches above
# 1024 change the number of updates per epoch (a different recipe, marked * in the report); the
# evaluation batch size does not affect training.
#
#   DATASETS="lanl14argus" GRID="-:- -:4096 -:16384 -:65536" ./run_ft_speedtest.sh   # scoring only
#   MODES="static temporal" ./run_ft_speedtest.sh
#   STAGES=0 ./run_ft_speedtest.sh      # skip the extra run that splits the epoch by stage
#
# A setting that fails (e.g. out of GPU memory) is reported in the log and the job carries on.
# Results CSVs of these runs go to speedtest/results/, never to results/.
# Run from the repo root on the login node.
set -uo pipefail

DATASETS=${DATASETS:-"unsw optc lanl14argus"}
MODES=${MODES:-"static"}
GRID=${GRID:-"-:- -:8192 2048:8192 4096:16384"}
STAGES=${STAGES:-1}                         # 1: one more run per dataset timing sampling/forward/backward/step
PSEED=${PSEED:-1}                           # which pretrained checkpoint (_ps<N>) to fine-tune

QUEUE=${QUEUE:-normal}
GPU_RES=${GPU_RES:-"num=1:mode=exclusive_process"}
CORES=${CORES:-4}
WALL=${WALL:-"04:00"}
MEM_MB=${MEM_MB:-32000}
PRELUDE=${PRELUDE:-"source ~/miniconda3/bin/activate gfm"}
LOGDIR=${LOGDIR:-logs/speedtest}
REPO=${REPO:-$PWD}

DRY=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    *) echo "unknown option: $a" >&2; exit 1 ;;
  esac
done
[ -f "$REPO/lp_finetune.py" ] || { echo "run this from the repo root (no lp_finetune.py here)" >&2; exit 1; }
[ -f "$REPO/ft_speedtest.py" ] || { echo "missing ft_speedtest.py (it goes in the repo root)" >&2; exit 1; }
[ "$DRY" = 1 ] || command -v bsub >/dev/null || { echo "bsub not found; run this on a login node" >&2; exit 1; }
for pair in $GRID; do
  case "$pair" in
    *:*) ;;
    *) echo "GRID takes mini-bs:eval-bs pairs (\"-\" for the default), not '$pair'" >&2; exit 1 ;;
  esac
  for v in "${pair%%:*}" "${pair##*:}"; do
    case "$v" in -) ;; ''|*[!0-9]*) echo "bad batch size '$v' in GRID pair '$pair'" >&2; exit 1 ;; esac
  done
done

# every checkpoint and split must be there before anything is submitted
ok=1
for ds in $DATASETS; do
  for s in tr va te; do
    [ -f "$REPO/data/${ds}_tgraph_${s}.pt" ] || { echo "  missing data/${ds}_tgraph_${s}.pt" >&2; ok=0; }
  done
  for mode in $MODES; do
    pre=rw; [ "$mode" = temporal ] && pre=trw
    ck="pretrained/$mode/$ds/${pre}_bert_${ds}_ps${PSEED}_tiny.pt"
    [ -f "$REPO/$ck" ] || { echo "  missing $ck" >&2; ok=0; }
  done
done
[ "$ok" = 1 ] || [ "$DRY" = 1 ] || { echo "nothing submitted" >&2; exit 1; }

mkdir -p "$LOGDIR"
name="ft_speedtest"
jobsh="$LOGDIR/${name}.sh"; out="$LOGDIR/${name}.out"
{
  echo "#!/usr/bin/env bash"
  printf 'cd %q\n' "$REPO"
  [ -n "$PRELUDE" ] && echo "$PRELUDE"
  echo 'echo "host: $(hostname)  gpus: ${CUDA_VISIBLE_DEVICES:-unset}  started: $(date -Is)"'
  echo 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null'
  for ds in $DATASETS; do
    for mode in $MODES; do
      trw=""; [ "$mode" = temporal ] && trw=" --trw"
      base="python ft_speedtest.py --dataset $ds$trw --pretrain-tag _ps$PSEED"
      for pair in $GRID; do
        m="${pair%%:*}"; e="${pair##*:}"; extra=""
        [ "$m" != "-" ] && extra="$extra --mini-bs $m"
        [ "$e" != "-" ] && extra="$extra --eval-bs $e"
        echo "$base$extra || echo \"SPEEDTEST FAILED: $ds $mode $pair\""
      done
      if [ "$STAGES" = 1 ]; then
        echo "$base --stages || echo \"SPEEDTEST FAILED: $ds $mode --stages\""
      fi
    done
  done
  echo 'echo; echo "==================== report ===================="'
  echo "python ft_speedtest.py --report"
} > "$jobsh"
n_runs=$(grep -c '^python ft_speedtest.py --dataset' "$jobsh")

args=(-J "$name" -q "$QUEUE" -gpu "$GPU_RES" -n "$CORES" -W "$WALL"
      -R "rusage[mem=$MEM_MB]" -o "$out" -e "$out")
echo "$n_runs one-epoch runs in one job ($DATASETS; $MODES; grid: $GRID)"
if [ "$DRY" = 1 ]; then
  echo "  bsub ${args[*]} < $jobsh"; sed 's/^/      /' "$jobsh"; exit 0
fi
res=$(bsub "${args[@]}" < "$jobsh" 2>&1)
id=$(printf '%s\n' "$res" | sed -n 's/^Job <\([0-9]*\)>.*/\1/p')
if [ -z "$id" ]; then
  echo "SUBMIT FAILED -- bsub said:" >&2; printf '%s\n' "$res" | sed 's/^/    /' >&2; exit 1
fi
echo "submitted job $id; when it finishes the report is at the end of $out"
echo "(or: python ft_speedtest.py --report)"