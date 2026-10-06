#!/bin/bash
#SBATCH --job-name=mdiberna_tisiago_p6_roundtrip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Usage: sbatch --partition=20 --array=0-5 scripts/run_p6_roundtrip.sh

set -euo pipefail

SELF="${BASH_SOURCE[0]}"
[[ -n "${SLURM_JOB_ID:-}" ]] && SELF="$(scontrol show job "$SLURM_JOB_ID" \
    | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"
source "$(cd "$(dirname "$SELF")" && pwd)/_common.sh"
CONDITIONS=(HeLa K562 U2OS RPE1_Async RPE1_Que RPE1_Sen)
PYTHON="${PYTHON_BIN:-$TISIAGO_CPU_PYTHON}"
require_python "$PYTHON"

cd "$REPO_ROOT"
"$PYTHON" -m tisiago.seq2func.roundtrip --condition "${CONDITIONS[$SLURM_ARRAY_TASK_ID]}" \
    --nproc "${SLURM_CPUS_PER_TASK:-1}"
