#!/bin/bash
#SBATCH --job-name=mdiberna_tisiago_p6_scorecard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Usage (from the repo root), after scripts/run_p6_predict_tracks.sh --mode float for every
# source (default output <run>/predicted_<split>_float):
#   sbatch --partition="$TISIAGO_CPU_PARTITION" scripts/run_p6_scorecard.sh --split val \
#       --heldout HeLa --out-dir data/p6/scorecard/val_rnacond_hela \
#       --source rnacond=data/p6/runs/ar_rnacond_hela/predicted_{split}_float \
#       --source base:HeLa=data/p6/runs/ar_hela_base/predicted_{split}_float
# Writes <out-dir>/scorecard.{tsv,md,json} and plots/ (tisiago.seq2func.scorecard).

set -euo pipefail

SELF="${BASH_SOURCE[0]}"
[[ -n "${SLURM_JOB_ID:-}" ]] && SELF="$(scontrol show job "$SLURM_JOB_ID" \
    | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"
source "$(cd "$(dirname "$SELF")" && pwd)/_common.sh"
PYTHON="${PYTHON_BIN:-$TISIAGO_CPU_PYTHON}"
require_python "$PYTHON"

export TMPDIR="$REPO_ROOT/.cache/tmp"
export MPLCONFIGDIR="$REPO_ROOT/.cache/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

cd "$REPO_ROOT"
"$PYTHON" -m tisiago.seq2func.scorecard "$@"
