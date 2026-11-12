#!/bin/bash
#SBATCH --job-name=mdiberna_tisiago_tracks_qc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Usage (from the repo root):
#   sbatch --partition=20 scripts/run_tracks_qc.sh

set -euo pipefail

SELF="${BASH_SOURCE[0]}"
[[ -n "${SLURM_JOB_ID:-}" ]] && SELF="$(scontrol show job "$SLURM_JOB_ID" \
    | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"
source "$(cd "$(dirname "$SELF")" && pwd)/_common.sh"
PYTHON="${PYTHON_BIN:-$TISIAGO_CPU_PYTHON}"
require_python "$PYTHON"

cd "$REPO_ROOT"
"$PYTHON" -m tisiago.tracks.qc --gtf "${TISIAGO_GTF:?Set TISIAGO_GTF in .env}"
