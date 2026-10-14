#!/bin/bash
#SBATCH --job-name=mdiberna_tisiago_p6_predicted_calls
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Usage (from the repo root), after scripts/run_p6_predict_tracks.sh:
#   sbatch --partition=20 scripts/run_p6_predicted_calls.sh --condition K562 \
#       --predicted-dir data/p6/runs/ar_resid_chx_w200_30k/predicted_val

set -euo pipefail

SELF="${BASH_SOURCE[0]}"
[[ -n "${SLURM_JOB_ID:-}" ]] && SELF="$(scontrol show job "$SLURM_JOB_ID" \
    | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"
source "$(cd "$(dirname "$SELF")" && pwd)/_common.sh"
PYTHON="${PYTHON_BIN:-$TISIAGO_CPU_PYTHON}"
require_python "$PYTHON"

export TMPDIR="$REPO_ROOT/.cache/tmp"
mkdir -p "$TMPDIR"

cd "$REPO_ROOT"
"$PYTHON" -m tisiago.seq2func.predicted_calls --nproc "${SLURM_CPUS_PER_TASK:-1}" "$@"
