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
# rnacond arms (--arm rnacond:<group> --head rnacond) submit train -> evaluate only: evaluate
# writes eval/<arm>_<condition>_val.tsv for every condition; periodicity and trial_score take
# per-condition arms and are skipped.
#
# Output: data/p6/runs/ar_<name>/ (train records, eval/, periodicity/, trial_score.json) and one
# row appended to autoresearch/p6/results.tsv. Train args default to --arm condition:K562
# --eval-every 1000 (later args win; --eval-windows: train.py's default, 100, or 20 per
# condition for rnacond arms); --windows-dir, --expression and --out
# are always set here. The val ceiling (data/p6/eval/ceiling_val.tsv, uniform_val.tsv) must
# exist; it is linked into the run's eval/ so evaluate never recomputes it. train.py refuses an
# --init-from run that has not finished, so chain on the source's train job with --after.
#
# GPU lanes. P6_TRIAL_LANES is a space-separated list of partitions, one per lane (default: two
# lanes on $TISIAGO_AGFT_PARTITION), e.g. "nvidia-A6000-20 nvidia-A6000-20 nvidia-L40S-24". Lane i
# runs on entry i with job name ${USER}_tisiago_p6ar_lane<i> and --dependency=singleton (one job
# per lane at a time); a trial's evaluate waits on its train in the same lane, so a trial uses one
# GPU at a time. All GPU sbatch calls of a trial use the lane's partition. The trial goes to the
# lane with the fewest existing lane jobs (ties: lowest index). Counted: GPU jobs of the user
# (tres gpu or an nvidia partition) in any state, minus --ignore-jobs, but only those in the chosen
# lane's partition; GPU jobs in other partitions never gate. A failed squeue aborts.
#   default : refuse if the chosen lane is not empty or its partition already holds as many GPU
#             jobs as it has lanes.
#   --queue : always submit. The chain's first job waits (afterany) on every job already in its
#             lane (any state), so each lane is strictly serial without relying on singleton
#             (singleton does not block on a pending predecessor; it stays as an extra guard),
#             and on the GPU jobs outside the lanes in its partition sorted by job id: with k
#             lanes on the partition, the j-th of them (0-based) waits on the j-th, j+k-th, ...
#             so trials never run on top of them. Jobs submitted outside the lanes later are not
#             gated.
#
# Overrides (environment): P6_TRIAL_CEILING, P6_TRIAL_REF_PERIODICITY, P6_TRIAL_BASE,
# P6_TRIAL_RESULTS, P6_TRIAL_WINDOWS_DIR, P6_TRIAL_EXPRESSION, P6_TRIAL_RUNS_DIR, P6_TRIAL_LANES.

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
read -ra LANES <<< "${P6_TRIAL_LANES:-$TISIAGO_AGFT_PARTITION $TISIAGO_AGFT_PARTITION}"
[[ ${#LANES[@]} -gt 0 ]] || { echo "run_p6_trial: P6_TRIAL_LANES is empty" >&2; exit 1; }
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
    TRAIN_ARGS=(--arm condition:K562 --eval-every 1000 "$@"
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
NLANES=${#LANES[@]}
declare -A LANE_JOBS=() LANE_IDS=() PART_COUNT=() PART_LANES=() PART_OUTSIDE=()
for ((i = 0; i < NLANES; i++)); do
    LANE_JOBS[$i]=0; LANE_IDS[$i]=""
    PART_LANES[${LANES[$i]}]=$(( ${PART_LANES[${LANES[$i]}]:-0} + 1 ))
done
while IFS='|' read -r jobid name partition tres; do
    [[ -n "$jobid" ]] || continue
    [[ "$tres" == *gpu* || "$partition" == *nvidia* ]] || continue
    [[ -z "${IGNORE[$jobid]:-}" ]] || continue
    PART_COUNT[$partition]=$(( ${PART_COUNT[$partition]:-0} + 1 ))
    idx="${name#"$LANE_PREFIX"}"
    if [[ "$name" == "$LANE_PREFIX"* && "$idx" =~ ^[0-9]+$ && $idx -lt $NLANES ]]; then
        LANE_JOBS[$idx]=$((LANE_JOBS[$idx] + 1))
        LANE_IDS[$idx]+=" $jobid"
    else
        PART_OUTSIDE[$partition]+="$jobid"$'\n'
    fi
done <<< "$SQUEUE_OUT"

LANE=0
for ((i = 1; i < NLANES; i++)); do
    [[ ${LANE_JOBS[$i]} -lt ${LANE_JOBS[$LANE]} ]] && LANE=$i
done
LANE_NAME="${LANE_PREFIX}${LANE}"
GPU_PARTITION="${LANES[$LANE]}"
mapfile -t OUTSIDE < <(printf '%s' "${PART_OUTSIDE[$GPU_PARTITION]:-}" | grep . | sort -n || true)
LANE_SUMMARY=""
for ((i = 0; i < NLANES; i++)); do LANE_SUMMARY+="lane$i(${LANES[$i]})=${LANE_JOBS[$i]} "; done
COUNT=${PART_COUNT[$GPU_PARTITION]:-0}
FIRST_DEP=(singleton)
echo "# GPU jobs in $GPU_PARTITION (all states): $COUNT; $LANE_SUMMARY;" \
    "outside lanes in $GPU_PARTITION (by id): ${OUTSIDE[*]:-none}; ignored: ${IGNORE_JOBS:-none}"
if [[ $QUEUE -eq 0 ]]; then
    if [[ $COUNT -ge ${PART_LANES[$GPU_PARTITION]} || ${LANE_JOBS[$LANE]} -gt 0 ]]; then
        msg="GPU busy ($COUNT GPU jobs in $GPU_PARTITION, lane$LANE has ${LANE_JOBS[$LANE]}); use --queue"
        if [[ $DRY_RUN -eq 1 ]]; then echo "# would refuse: $msg"; else die "$msg"; fi
    fi
else
    read -ra lane_ids <<< "${LANE_IDS[$LANE]}"
    for id in $(printf '%s\n' "${lane_ids[@]}" | grep . | sort -n); do
        FIRST_DEP+=("afterany:$id")
    done
    K=${PART_LANES[$GPU_PARTITION]}
    RANK=0
    for ((i = 0; i < LANE; i++)); do [[ "${LANES[$i]}" == "$GPU_PARTITION" ]] && RANK=$((RANK + 1)); done
    for ((i = RANK; i < ${#OUTSIDE[@]}; i += K)); do FIRST_DEP+=("afterany:${OUTSIDE[$i]}"); done
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
if [[ "$ARM" == rnacond:* ]]; then
    echo "# $ARM: periodicity and trial_score skipped (per-condition arms only)" >&2
    echo "trial $(basename "$RUN"): train=${TRAIN:-none} evaluate=$EVAL"
    echo "run dir: $RUN"
    exit 0
fi
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
