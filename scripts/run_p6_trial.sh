#!/bin/bash
# One P6 autoresearch trial as an afterok chain (submit from anywhere; runs from the repo root):
#   train (GPU) -> evaluate val (GPU) + periodicity (CPU, --max-windows 100) -> trial_score (CPU)
#
# Usage:
#   bash scripts/run_p6_trial.sh [options] <name> <train args...>
#   bash scripts/run_p6_trial.sh [options] --eval-only <run dir>   # score a finished run
# Options: --dry-run (print sbatch lines), --queue, --after <jobid>[,...] (afterok on the first
# job of the chain; repeatable), --ignore-jobs <jobid>[,...] (GPU jobs not to count).
#
# Example (seqconv two-stage from the mlp base, after its train job):
#   bash scripts/run_p6_trial.sh --queue --after 11125770 seqconv_w50_2stage \
#       --init-from data/p6/runs/k562_base/best --head seqconv --frame-weight 50 --steps 10000
#
# Output: data/p6/runs/ar_<name>/ (train records, eval/, periodicity/, trial_score.json) and one
# row appended to autoresearch/p6/results.tsv. Train args default to --arm condition:K562
# --eval-every 1000 --eval-windows 100 (later args win); --windows-dir, --expression and --out
# are always set here. The val ceiling (data/p6/eval/ceiling_val.tsv, uniform_val.tsv) must
# exist; it is linked into the run's eval/ so evaluate never recomputes it. train.py refuses an
# --init-from run that has not finished, so chain on the source's train job with --after.
#
# GPU concurrency (at most 2 GPU jobs running). Counted: every GPU job of the user (tres gpu or
# an nvidia partition) in any state, minus --ignore-jobs. A failed squeue aborts. Trial GPU jobs
# run in two job-name lanes, ${USER}_tisiago_p6ar_lane{0,1}, with --dependency=singleton (one job
# per lane at a time); a trial's evaluate waits on its train in the same lane, so a trial uses
# one GPU at a time. The trial goes to the lane with fewer jobs (ties: lane0).
#   default : refuse if the count is >= 2 or the chosen lane is not empty.
#   --queue : always submit. The chain's first job waits (afterany) on every job already in its
#             lane (any state), so each lane is strictly serial without relying on singleton
#             (singleton does not block on a pending predecessor; it stays as an extra guard),
#             and on the GPU jobs outside the lanes sorted by job id: lane i on the i-th,
#             i+2-th, ... (lane0 on the 1st, 3rd, ...; lane1 on the 2nd, 4th, ...), so trials
#             never run on top of them. Jobs submitted outside the lanes later are not gated.
#
# Overrides (environment): P6_TRIAL_CEILING, P6_TRIAL_REF_PERIODICITY, P6_TRIAL_BASE,
# P6_TRIAL_RESULTS, P6_TRIAL_WINDOWS_DIR, P6_TRIAL_EXPRESSION, P6_TRIAL_RUNS_DIR.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
cd "$REPO_ROOT"

usage() { sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 2; }

DRY_RUN=0
QUEUE=0
EVAL_ONLY=""
AFTER=""
IGNORE_JOBS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --queue) QUEUE=1; shift ;;
        --eval-only) [[ $# -ge 2 ]] || usage; EVAL_ONLY="$2"; shift 2 ;;
        --after) [[ $# -ge 2 ]] || usage; AFTER="${AFTER:+$AFTER,}$2"; shift 2 ;;
        --ignore-jobs) [[ $# -ge 2 ]] || usage; IGNORE_JOBS="${IGNORE_JOBS:+$IGNORE_JOBS,}$2"; shift 2 ;;
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
    if [[ -z "$AFTER" && ( "$finished" != True || ! -d "$RUN/best" ) ]]; then
        msg="$RUN is not a finished run (run.json without best); pass --after <its train job>"
        if [[ $DRY_RUN -eq 1 ]]; then echo "# would refuse: $msg" >&2; else die "$msg"; fi
    fi
    ARM="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["args"]["arm"])' \
        "$RUN/run.json" 2>/dev/null || echo condition:K562)"
    TRAIN_ARGS=()
else
    [[ $# -ge 1 ]] || usage
    NAME="$1"; shift
    [[ "$NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || die "trial name must match [A-Za-z0-9_.-]+"
    RUN="${P6_TRIAL_RUNS_DIR:-data/p6/runs}/ar_${NAME}"
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
# Every GPU job of the user counts, in any state (running, pending, held, waiting on a
# dependency), except the ids passed with --ignore-jobs. A failed squeue aborts.
declare -A IGNORE=()
for id in ${IGNORE_JOBS//,/ }; do IGNORE[$id]=1; done
SQUEUE_OUT="$(squeue -u "$USER" -h -o '%i|%j|%P|%b')" || die "squeue failed; refusing to guess"
declare -a OUTSIDE=()
declare -A LANE_JOBS=([0]=0 [1]=0)
declare -A LANE_IDS=([0]="" [1]="")
COUNT=0
while IFS='|' read -r jobid name partition tres; do
    [[ -n "$jobid" ]] || continue
    [[ "$tres" == *gpu* || "$partition" == *nvidia* ]] || continue
    [[ -z "${IGNORE[$jobid]:-}" ]] || continue
    COUNT=$((COUNT + 1))
    if [[ "$name" == "${LANE_PREFIX}"[01] ]]; then
        LANE_JOBS[${name: -1}]=$((LANE_JOBS[${name: -1}] + 1))
        LANE_IDS[${name: -1}]+=" $jobid"
    else
        OUTSIDE+=("$jobid")
    fi
done <<< "$SQUEUE_OUT"
mapfile -t OUTSIDE < <(printf '%s\n' "${OUTSIDE[@]}" | grep . | sort -n)

LANE=0
[[ ${LANE_JOBS[1]} -lt ${LANE_JOBS[0]} ]] && LANE=1
LANE_NAME="${LANE_PREFIX}${LANE}"
FIRST_DEP=(singleton)
echo "# GPU jobs (all states): $COUNT; lane0=${LANE_JOBS[0]} lane1=${LANE_JOBS[1]};" \
    "outside lanes (by id): ${OUTSIDE[*]:-none}; ignored: ${IGNORE_JOBS:-none}"
if [[ $QUEUE -eq 0 ]]; then
    if [[ $COUNT -ge 2 || ${LANE_JOBS[$LANE]} -gt 0 ]]; then
        msg="GPU busy ($COUNT GPU jobs, lane$LANE has ${LANE_JOBS[$LANE]}); use --queue"
        if [[ $DRY_RUN -eq 1 ]]; then echo "# would refuse: $msg"; else die "$msg"; fi
    fi
else
    read -ra lane_ids <<< "${LANE_IDS[$LANE]}"
    for id in $(printf '%s\n' "${lane_ids[@]}" | grep . | sort -n); do
        FIRST_DEP+=("afterany:$id")
    done
    for ((i = LANE; i < ${#OUTSIDE[@]}; i += 2)); do FIRST_DEP+=("afterany:${OUTSIDE[$i]}"); done
fi
for id in ${AFTER//,/ }; do FIRST_DEP+=("afterok:$id"); done
FIRST_DEP_STR="$(IFS=,; echo "${FIRST_DEP[*]}")"
echo "# lane: $LANE_NAME; first job waits on: $FIRST_DEP_STR"

# --- submission --------------------------------------------------------------------------------
SUBMITTED=()
submit() {  # submit <var> <placeholder> <sbatch args...>; sets <var> to the job id
    local var="$1" placeholder="$2" id
    shift 2
    echo "sbatch $(printf '%q ' "$@")" >&2
    if [[ $DRY_RUN -eq 1 ]]; then
        id="$placeholder"
    elif ! id="$(sbatch --parsable "$@")"; then
        echo "run_p6_trial: sbatch failed; already submitted: ${SUBMITTED[*]:-none}" >&2
        if [[ ${#SUBMITTED[@]} -gt 0 ]]; then
            scancel "${SUBMITTED[@]}" && echo "run_p6_trial: cancelled ${SUBMITTED[*]}" >&2
        fi
        exit 1
    fi
    id="${id%%;*}"
    SUBMITTED+=("$id")
    printf -v "$var" '%s' "$id"
}

COMMON=(--kill-on-invalid-dep=yes)
GPU=(--partition="$GPU_PARTITION" --job-name="$LANE_NAME" "${COMMON[@]}")
CPU=(--partition="$CPU_PARTITION" "${COMMON[@]}")

if [[ $DRY_RUN -eq 0 ]]; then
    mkdir -p "$RUN/eval"
    for f in "$CEILING" "$UNIFORM"; do ln -sfn "$(realpath "$f")" "$RUN/eval/$(basename "$f")"; done
fi

TRAIN=""
if [[ ${#TRAIN_ARGS[@]} -gt 0 ]]; then
    submit TRAIN '<train>' "${GPU[@]}" --comment="ar_${NAME} train" \
        --dependency="$FIRST_DEP_STR" scripts/run_p6_train.sh "${TRAIN_ARGS[@]}"
    EVAL_DEP="afterok:${TRAIN},singleton"
    PERIOD_DEP="afterok:${TRAIN}"
else
    EVAL_DEP="$FIRST_DEP_STR"
    PERIOD_DEP="$(IFS=,; for id in ${AFTER//,/ }; do echo -n "afterok:$id,"; done)"
    PERIOD_DEP="${PERIOD_DEP%,}"
fi
DATA=(--windows-dir "$WINDOWS_DIR" --expression "$EXPRESSION")
submit EVAL '<evaluate>' "${GPU[@]}" --comment="ar_${NAME} evaluate" \
    --dependency="$EVAL_DEP" scripts/run_p6_evaluate.sh --checkpoint "$RUN/best" --arm "$ARM" \
    --split val "${DATA[@]}" --out-dir "$RUN/eval"
submit PERIOD '<periodicity>' "${CPU[@]}" --job-name="${USER}_tisiago_p6ar_periodicity" \
    ${PERIOD_DEP:+--dependency="$PERIOD_DEP"} scripts/run_p6_periodicity.sh \
    --checkpoint "$RUN/best" --arm "$ARM" --split val --max-windows 100 "${DATA[@]}" \
    --out-dir "$RUN" --run-name periodicity
SCORE_CMD=(cd "$REPO_ROOT" '&&' "$TISIAGO_CPU_PYTHON" -m tisiago.seq2func.trial_score
    --run "$RUN" --ceiling "$CEILING" --reference-periodicity "$REF_PERIODICITY"
    "${BASE_ARGS[@]}" --out "$RESULTS")
submit SCORE '<score>' "${CPU[@]}" --job-name="${USER}_tisiago_p6ar_score" \
    --dependency="afterok:${EVAL}:${PERIOD}" --cpus-per-task=1 --mem=8G --time=1:00:00 \
    --output="logs/%x_%j.out" --error="logs/%x_%j.err" --wrap="${SCORE_CMD[*]}"

echo "trial $(basename "$RUN"): train=${TRAIN:-none} evaluate=$EVAL periodicity=$PERIOD score=$SCORE"
echo "run dir: $RUN"
