#!/bin/bash
#SBATCH --job-name=mdiberna_tisiago_p6_track_tis_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Usage (from the repo root), after scripts/run_p6_predict_tracks.sh --mode float for every
# condition (default output <run>/predicted_<split>_float):
#   sbatch --partition="$TISIAGO_CPU_PARTITION" scripts/run_p6_track_tis_eval.sh --split val
#   sbatch --partition="$TISIAGO_CPU_PARTITION" scripts/run_p6_track_tis_eval.sh \
#       --split test --i-am-final
# Writes data/p6/track_tis_eval/<split>/report.{json,md}. A later --predicted overrides the
# default runs below.

set -euo pipefail

SELF="${BASH_SOURCE[0]}"
[[ -n "${SLURM_JOB_ID:-}" ]] && SELF="$(scontrol show job "$SLURM_JOB_ID" \
    | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"
source "$(cd "$(dirname "$SELF")" && pwd)/_common.sh"
PYTHON="${PYTHON_BIN:-$TISIAGO_CPU_PYTHON}"
require_python "$PYTHON"

export TMPDIR="$REPO_ROOT/.cache/tmp"
mkdir -p "$TMPDIR"

RUNS=data/p6/runs
PREDICTED=(
    "K562=$RUNS/ar_resid_chx_w200_30k/predicted_{split}_float"
    "HeLa=$RUNS/ar_hela_resid_chx_w200_30k/predicted_{split}_float"
    "U2OS=$RUNS/ar_u2os_resid_chx_w200_30k/predicted_{split}_float"
    "RPE1_Async=$RUNS/ar_rpe1_async_resid_chx_w200_30k/predicted_{split}_float"
    "RPE1_Que=$RUNS/ar_rpe1_que_resid_chx_w200_30k/predicted_{split}_float"
    "RPE1_Sen=$RUNS/ar_rpe1_sen_resid_chx_w200_30k/predicted_{split}_float"
)

cd "$REPO_ROOT"
"$PYTHON" -m tisiago.seq2func.track_tis_eval --predicted "${PREDICTED[@]}" "$@"
