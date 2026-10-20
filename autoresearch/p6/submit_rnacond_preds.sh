#!/bin/bash
# Float val predictions of one rnacond run for every condition, then a scorecard vs the reference.
# Usage (repo root): bash autoresearch/p6/submit_rnacond_preds.sh <run name> [--after <jobid>]
#   <run name>: data/p6/runs/<run name> (e.g. ar_rnacond_k562_f); held-out group K562.
# Prediction jobs: A6000/L40S, staggered 20 s; at most 6 of the user's prediction jobs queued
# (refuses otherwise). Scorecard: CPU partition, afterok on all six, sources rnacond (reference
# ar_rnacond_k562_v2) and the run, out data/p6/scorecard/val_t3_<run name>.
set -euo pipefail
source scripts/_common.sh
RUN="$1"; shift
AFTER=""
[[ "${1:-}" == --after ]] && AFTER="--dependency=afterok:$2"
DIR="data/p6/runs/$RUN"
[[ -d "$DIR" ]] || { echo "no $DIR" >&2; exit 1; }
queued=$(squeue -h -u "$USER" -n "${USER}_tisiago_p6_predrnac" | wc -l)
[[ $queued -eq 0 ]] || { echo "$queued prediction jobs already queued; wait" >&2; exit 1; }
IDS=()
for C in K562 HeLa U2OS RPE1_Async RPE1_Que RPE1_Sen; do
    id=$(sbatch --parsable --partition=nvidia-A6000-20,nvidia-L40S-24 \
        --job-name="${USER}_tisiago_p6_predrnac" ${AFTER:+"$AFTER"} \
        scripts/run_p6_predict_tracks.sh --checkpoint "$DIR/best" --arm rnacond:K562 \
        --condition "$C" --split val --mode float)
    IDS+=("$id")
    echo "predict $C: $id"
    sleep 20
done
DEP=$(IFS=:; echo "${IDS[*]}")
sc=$(sbatch --parsable --partition="$TISIAGO_CPU_PARTITION" --dependency="afterok:$DEP" \
    scripts/run_p6_scorecard.sh --split val --heldout K562 --out-dir "data/p6/scorecard/val_t3_$RUN" \
    --source "rnacond=data/p6/runs/ar_rnacond_k562_v2/predicted_{split}_float" \
    --source "$RUN=$DIR/predicted_{split}_float")
echo "scorecard: $sc"
