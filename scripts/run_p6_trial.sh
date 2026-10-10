#!/bin/bash
# One P6 autoresearch trial as an afterok chain (submit from anywhere; runs from the repo root):
#   train (GPU) -> evaluate val (GPU) + periodicity (CPU, --max-windows 100) -> trial_score (CPU)
#
# Usage:
#   bash scripts/run_p6_trial.sh [--dry-run] [--queue] <name> <train args...>
#   bash scripts/run_p6_trial.sh [--dry-run] --eval-only <run dir>   # score a finished run
#
# Example (seqconv two-stage from the mlp base):
#   bash scripts/run_p6_trial.sh seqconv_w50_2stage --init-from data/p6/runs/k562_base/best \
#       --head seqconv --frame-weight 50 --steps 10000
#
# Output: data/p6/runs/ar_<name>/ (train records, eval/, periodicity/, trial_score.json) and one
# row appended to autoresearch/p6/results.tsv. Train args default to --arm condition:K562
# --eval-every 1000 --eval-windows 100 (later args win); --windows-dir, --expression and --out
# are always set here. The val ceiling (data/p6/eval/ceiling_val.tsv, uniform_val.tsv) must
# exist; it is linked into the run's eval/ so evaluate never recomputes it.
#
# GPU concurrency (at most 2 of the user's GPU jobs at a time). GPU jobs run in two job-name
# lanes, ${USER}_tisiago_p6ar_lane{0,1}, with --dependency=singleton, so each lane runs one job
# at a time; a trial's evaluate waits on its train in the same lane, so a trial holds one GPU.
#   default : refuse unless the user has <= 1 GPU job running or eligible to start (held jobs
#             and jobs pending on a dependency are not counted) and a lane is idle.
#   --queue : always submit, to the lane with fewer jobs. Each GPU job outside the lanes that is
#             running/eligible at submit time is assigned to a lane (afterany), so lanes wait for
#             those jobs (e.g. the 30k baselines) instead of stacking on top of them.
#
# Overrides (environment): P6_TRIAL_CEILING, P6_TRIAL_REF_PERIODICITY, P6_TRIAL_BASE,
# P6_TRIAL_RESULTS, P6_TRIAL_WINDOWS_DIR, P6_TRIAL_EXPRESSION.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
cd "$REPO_ROOT"

usage() { sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 2; }

DRY_RUN=0
QUEUE=0
EVAL_ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --queue) QUEUE=1; shift ;;
        --eval-only) [[ $# -ge 2 ]] || usage; EVAL_ONLY="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) break ;;
    esac
done

WINDOWS_DIR="${P6_TRIAL_WINDOWS_DIR:-data/p6/windows_new}"
EXPRESSION="${P6_TRIAL_EXPRESSION:-data/p6/expression/expression.new.parquet}"
CEILING="${P6_TRIAL_CEILING:-data/p6/eval/ceiling_val.tsv}"
UNIFORM="$(dirname "$CEILING")/$(basename "$CEILING" | sed 's/^ceiling_/uniform_/')"
REF_PERIODICITY="${P6_TRIAL_REF_PERIODICITY:-data/p6/periodicity/reference_condition_k562/regions_val_max100_replicate.tsv}"
BASE="${P6_TRIAL_BASE:-data/p6/runs/k562_base/trial_score.json}"
RESULTS="${P6_TRIAL_RESULTS:-autoresearch/p6/results.tsv}"
GPU_PARTITION="$TISIAGO_AGFT_PARTITION"
CPU_PARTITION="${TISIAGO_CPU_PARTITION:?set TISIAGO_CPU_PARTITION in .env}"
LANE_PREFIX="${USER}_tisiago_p6ar_lane"

die() { echo "run_p6_trial: $*" >&2; exit 1; }

if [[ -n "$EVAL_ONLY" ]]; then
    [[ $# -eq 0 ]] || die "--eval-only takes no train args"
    RUN="${EVAL_ONLY%/}"
    NAME="$(basename "$RUN")"
    finished="$(python3 -c 'import json,sys; print(bool(json.load(open(sys.argv[1])).get("best")))' \
        "$RUN/run.json" 2>/dev/null || echo False)"
    if [[ "$finished" != True || ! -d "$RUN/best" ]]; then
        msg="$RUN is not a finished run (run.json without best, or no best/)"
        if [[ $DRY_RUN -eq 1 ]]; then echo "# would refuse: $msg" >&2; else die "$msg"; fi
    fi
    ARM="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["args"]["arm"])' \
        "$RUN/run.json" 2>/dev/null || echo condition:K562)"
    TRAIN_ARGS=()
else
    [[ $# -ge 1 ]] || usage
    NAME="$1"; shift
    [[ "$NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || die "trial name must match [A-Za-z0-9_.-]+"
    RUN="data/p6/runs/ar_${NAME}"
    for record in log.jsonl run.json best last; do
        [[ ! -e "$RUN/$record" ]] || die "$RUN already holds $record; pick another name"
    done
    TRAIN_ARGS=(--arm condition:K562 --eval-every 1000 --eval-windows 100 "$@"
        --windows-dir "$WINDOWS_DIR" --expression "$EXPRESSION" --out "$RUN")
    ARM="condition:K562"
    for ((i = 0; i < ${#TRAIN_ARGS[@]} - 1; i++)); do
        [[ "${TRAIN_ARGS[$i]}" == --arm ]] && ARM="${TRAIN_ARGS[$((i + 1))]}"
    done
fi

for f in "$CEILING" "$UNIFORM" "$REF_PERIODICITY"; do
    [[ -f "$f" ]] || [[ $DRY_RUN -eq 1 ]] || die "missing $f (compute the val references first)"
    [[ -f "$f" ]] || echo "# warning: $f does not exist yet" >&2
done
BASE_ARGS=()
if [[ -f "$BASE" && "$(realpath -m "$RUN")" != "$(realpath -m "$(dirname "$BASE")")" ]]; then
    BASE_ARGS=(--base "$BASE")
else
    echo "# note: no base score ($BASE); gate_fit will be NA for this trial" >&2
fi

# --- GPU lanes ---------------------------------------------------------------------------------
declare -a OUTSIDE=()
declare -A LANE_JOBS=([0]=0 [1]=0)
ACTIVE=0
while IFS='|' read -r jobid name state tres reason; do
    [[ "$tres" == *gpu* ]] || continue
    if [[ "$name" == "${LANE_PREFIX}"[01] ]]; then
        LANE_JOBS[${name: -1}]=$((LANE_JOBS[${name: -1}] + 1))
    fi
    [[ "$state" == PENDING && ( "$reason" == Dependency || "$reason" == JobHeld* ) ]] && continue
    ACTIVE=$((ACTIVE + 1))
    [[ "$name" == "${LANE_PREFIX}"[01] ]] || OUTSIDE+=("$jobid")
done < <(squeue -u "$USER" -h -t R,PD -o '%i|%j|%T|%b|%r' 2>/dev/null || true)

LANE=0
[[ ${LANE_JOBS[1]} -lt ${LANE_JOBS[0]} ]] && LANE=1
LANE_NAME="${LANE_PREFIX}${LANE}"
GPU_DEP="singleton"
echo "# GPU jobs running/eligible: $ACTIVE; lane jobs: lane0=${LANE_JOBS[0]} lane1=${LANE_JOBS[1]}"
if [[ $QUEUE -eq 0 ]]; then
    if [[ $ACTIVE -ge 2 || ${LANE_JOBS[$LANE]} -gt 0 ]]; then
        msg="GPU busy ($ACTIVE active, lane$LANE has ${LANE_JOBS[$LANE]} jobs); use --queue"
        if [[ $DRY_RUN -eq 1 ]]; then echo "# would refuse: $msg"; else die "$msg"; fi
    fi
elif [[ ${LANE_JOBS[$LANE]} -eq 0 && ${#OUTSIDE[@]} -gt $LANE ]]; then
    GPU_DEP="singleton,afterany:${OUTSIDE[$LANE]}"
fi
echo "# lane: $LANE_NAME ($GPU_DEP)"

# --- submission --------------------------------------------------------------------------------
submit() {  # submit <placeholder> <sbatch args...>; echoes the job id (placeholder on dry run)
    local placeholder="$1"; shift
    echo "sbatch $(printf '%q ' "$@")" >&2
    if [[ $DRY_RUN -eq 1 ]]; then echo "$placeholder"; else sbatch --parsable "$@"; fi
}

COMMON=(--kill-on-invalid-dep=yes)
GPU=(--partition="$GPU_PARTITION" --job-name="$LANE_NAME" "${COMMON[@]}")
CPU=(--partition="$CPU_PARTITION" "${COMMON[@]}")

if [[ $DRY_RUN -eq 0 ]]; then
    mkdir -p "$RUN/eval"
    for f in "$CEILING" "$UNIFORM"; do ln -sfn "$(realpath "$f")" "$RUN/eval/$(basename "$f")"; done
fi

DEP_TRAIN=""
if [[ ${#TRAIN_ARGS[@]} -gt 0 ]]; then
    TRAIN=$(submit '<train>' "${GPU[@]}" --comment="ar_${NAME} train" \
        --dependency="$GPU_DEP" scripts/run_p6_train.sh "${TRAIN_ARGS[@]}")
    DEP_TRAIN="afterok:${TRAIN}"
    GPU_DEP="${DEP_TRAIN},singleton"
fi
DATA=(--windows-dir "$WINDOWS_DIR" --expression "$EXPRESSION")
EVAL=$(submit '<evaluate>' "${GPU[@]}" --comment="ar_${NAME} evaluate" \
    --dependency="$GPU_DEP" scripts/run_p6_evaluate.sh --checkpoint "$RUN/best" --arm "$ARM" \
    --split val "${DATA[@]}" --out-dir "$RUN/eval")
PERIOD=$(submit '<periodicity>' "${CPU[@]}" --job-name="${USER}_tisiago_p6ar_periodicity" \
    ${DEP_TRAIN:+--dependency="$DEP_TRAIN"} scripts/run_p6_periodicity.sh \
    --checkpoint "$RUN/best" --arm "$ARM" --split val --max-windows 100 "${DATA[@]}" \
    --out-dir "$RUN" --run-name periodicity)
SCORE_CMD=(cd "$REPO_ROOT" '&&' "$TISIAGO_CPU_PYTHON" -m tisiago.seq2func.trial_score
    --run "$RUN" --ceiling "$CEILING" --reference-periodicity "$REF_PERIODICITY"
    "${BASE_ARGS[@]}" --out "$RESULTS")
SCORE=$(submit '<score>' "${CPU[@]}" --job-name="${USER}_tisiago_p6ar_score" \
    --dependency="afterok:${EVAL}:${PERIOD}" --cpus-per-task=1 --mem=8G --time=1:00:00 \
    --output="logs/%x_%j.out" --error="logs/%x_%j.err" --wrap="${SCORE_CMD[*]}")

echo "trial $(basename "$RUN"): train=${TRAIN:-none} evaluate=$EVAL periodicity=$PERIOD score=$SCORE"
echo "run dir: $RUN"
