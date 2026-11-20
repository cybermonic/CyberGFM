#!/usr/bin/env bash
# Out-of-vocabulary breakdown on LANL-41h (pre-review item 3): re-run the LANL-41h fine-tunes and
# Argus with --save-scores so every test edge's score is kept, then oov_breakdown.py splits AP by
# whether each edge's endpoints were seen before the cutoff.
#
#   CyberGFM  lp_finetune.py on the existing lanl14argus-ts checkpoints (no new pretraining),
#             static + temporal, wl4, pretraining seeds 1-5, fine-tune seed 0 -- the same command
#             run_all_hpc.sh runs, plus --save-scores
#   Argus     argus_test_lanl_ts.py (41 h cutoff), seeds 1-5, plus --save-scores
#
# Everything goes to results-oov/, so the main results are untouched:
#   results-oov/lp-<mode>/lanl14argus-ts/lp_ps1_tiny_wl4_ps1_s0.csv  (+ _scores.npz)
#   results-oov/argus/lanl14argus-dirtyts-41h.csv, results-oov/argus/scores/..._s1.pt
#
#   ./run_oov.sh --dry-run
#   ./run_oov.sh
#   python oov_breakdown.py          # once they finish (reads saved scores; fine on the login node)
#
# Needs the --save-scores patch (common.py, lp_finetune.py, prior_works/argus_common.py).
# Run from the repo root on the login node.
set -uo pipefail

DATASET=lanl14argus-ts
PSEEDS=${PSEEDS:-"1 2 3 4 5"}
MODES=${MODES:-"static temporal"}
WL=${WL:-4}
SIZE=${SIZE:-tiny}
RESULTS=${RESULTS:-results-oov}
PARTS=${PARTS:-"cybergfm argus"}
QUEUE=${QUEUE:-normal}
GPU_RES=${GPU_RES:-"num=1:mode=exclusive_process"}
CORES=${CORES:-4}
PRELUDE=${PRELUDE:-"source ~/miniconda3/bin/activate gfm"}
LOGDIR=${LOGDIR:-logs/oov}
REPO=${REPO:-$PWD}

DRY=0
for a in "$@"; do case "$a" in --dry-run) DRY=1 ;; *) echo "unknown option: $a" >&2; exit 1 ;; esac; done
[ -f "$REPO/lp_finetune.py" ] || { echo "run this from the repo root" >&2; exit 1; }
grep -q -- '--save-scores' "$REPO/lp_finetune.py" && grep -q -- '--save-scores' "$REPO/prior_works/argus_common.py" \
  || { echo "lp_finetune.py / argus_common.py have no --save-scores: apply the patch first" >&2; exit 1; }
[ "$DRY" = 1 ] || command -v bsub >/dev/null || { echo "bsub not found; run this on a login node" >&2; exit 1; }
mkdir -p "$LOGDIR"

submit() {  # submit <jobname> <wall> <cwd> <command>
  local name="$1" wall="$2" cwd="$3" cmd="$4"
  local jobsh="$LOGDIR/${name}.sh" out="$LOGDIR/${name}.out"
  {
    echo "#!/usr/bin/env bash"
    printf 'cd %q\n' "$cwd"
    [ -n "$PRELUDE" ] && echo "$PRELUDE"
    echo 'echo "host: $(hostname)  gpus: ${CUDA_VISIBLE_DEVICES:-unset}  started: $(date -Is)"'
    echo "$cmd"
  } > "$jobsh"
  local args=(-J "$name" -q "$QUEUE" -gpu "$GPU_RES" -n "$CORES" -W "$wall"
              -R "rusage[mem=32000]" -o "$out" -e "$out")
  if [ "$DRY" = 1 ]; then echo "  bsub ${args[*]} < $jobsh"; echo "      $cmd"; return 0; fi
  local res id
  res=$(bsub "${args[@]}" < "$jobsh" 2>&1)
  id=$(printf '%s\n' "$res" | sed -n 's/^Job <\([0-9]*\)>.*/\1/p')
  if [ -z "$id" ]; then echo "  SUBMIT FAILED $name -- bsub said:" >&2; printf '%s\n' "$res" | sed 's/^/    /' >&2; return 1; fi
  printf '  %-36s %s\n' "$name" "$id"
}

n=0
for part in $PARTS; do
  case "$part" in
    cybergfm)
      for mode in $MODES; do
        trw=""; pre=rw_bert; [ "$mode" = temporal ] && { trw="--trw"; pre=trw_bert; }
        for s in $PSEEDS; do
          ck="pretrained/$mode/$DATASET/${pre}_${DATASET}_ps${s}_${SIZE}.pt"
          if [ ! -f "$REPO/$ck" ]; then echo "  no checkpoint $ck -- skipped" >&2; continue; fi
          submit "oov_ft_${mode}_ps${s}" "08:00" "$REPO" \
            "python lp_finetune.py --dataset $DATASET $trw --size $SIZE --device 0 --walk-len $WL --seed 0 --pretrain-tag _ps${s} --tag _ps${s}_s0 --out-dir $RESULTS/lp-$mode/$DATASET --save-scores" \
            && n=$((n + 1))
        done
      done ;;
    argus)
      for s in $PSEEDS; do
        submit "oov_argus_s${s}" "24:00" "$REPO/prior_works" \
          "python argus_test_lanl_ts.py --device 0 --seed $s --out-dir ../$RESULTS/argus --save-scores" \
          && n=$((n + 1))
      done ;;
    *) echo "unknown part '$part'" >&2; exit 1 ;;
  esac
done
echo
echo "$n job(s) $([ "$DRY" = 1 ] && echo listed || echo submitted); logs in $LOGDIR/"
echo "then: python oov_breakdown.py --root $RESULTS"
