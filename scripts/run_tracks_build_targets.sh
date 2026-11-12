#!/bin/bash
#SBATCH --job-name=mdiberna_tisiago_tracks_build_targets
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Usage (from the repo root):
#   A=$(sbatch --parsable --partition=20 --array=0-35%12 scripts/run_tracks_build_targets.sh)
#   sbatch --partition=20 --dependency=afterok:$A scripts/run_tracks_build_targets.sh pool

set -euo pipefail

SELF="${BASH_SOURCE[0]}"
[[ -n "${SLURM_JOB_ID:-}" ]] && SELF="$(scontrol show job "$SLURM_JOB_ID" \
    | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"
source "$(cd "$(dirname "$SELF")" && pwd)/_common.sh"
PYTHON="${PYTHON_BIN:-$TISIAGO_CPU_PYTHON}"
require_python "$PYTHON"

cd "$REPO_ROOT"
if [[ "${1:-}" == "pool" ]]; then
    "$PYTHON" -m tisiago.tracks.track_io --out-dir data/tracks/tracks --pool
else
    "$PYTHON" -m tisiago.tracks.track_io --out-dir data/tracks/tracks --index "$SLURM_ARRAY_TASK_ID"
fi
