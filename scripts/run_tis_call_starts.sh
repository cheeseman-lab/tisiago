#!/bin/bash
#SBATCH --job-name=mdiberna_tisiago_call_starts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Discrete start-site calls from the frozen multi-seed ensemble head (tisiago.call_starts).
# Fails unless the calls reproduce the ensemble metrics logged by representation_eval.
#
# Usage:
#   sbatch --partition=20 scripts/run_tis_call_starts.sh [ARM] [SPLIT] [STORE] [RUN_DIR]

set -euo pipefail

SELF="${BASH_SOURCE[0]}"
[[ -n "${SLURM_JOB_ID:-}" ]] && SELF="$(scontrol show job "$SLURM_JOB_ID" \
    | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"
source "$(cd "$(dirname "$SELF")" && pwd)/_common.sh"
ARM="${1:-ag_w8k}"
SPLIT="${2:-test}"
STORE="${3:-${TISIAGO_DENSE_STORE:-$REPO_ROOT/data/dense_exp_store}}"
RUN_DIR="${4:-$REPO_ROOT/data/dense_txp_representation}"
PYTHON="${PYTHON_BIN:-$TISIAGO_CPU_PYTHON}"
require_python "$PYTHON"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "$REPO_ROOT"
"$PYTHON" -m tisiago.call_starts \
    --store "$STORE" \
    --run-dir "$RUN_DIR" \
    --arm "$ARM" \
    --split "$SPLIT" \
    --out "$REPO_ROOT/data/calls/${ARM}_${SPLIT}_calls.parquet"
