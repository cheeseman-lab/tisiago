#!/bin/bash
#SBATCH --job-name=mdiberna_tisiago_p6_periodicity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Usage (from the repo root; CPU by default):
#   sbatch --partition="$TISIAGO_CPU_PARTITION" scripts/run_p6_periodicity.sh \
#       --checkpoint data/p6/runs/condition_hela/best --max-windows 100
#   # replicate reference (raw rep2 vs rep1; no model) for the same windows:
#   sbatch --partition="$TISIAGO_CPU_PARTITION" scripts/run_p6_periodicity.sh \
#       --reference replicate --arm condition:HeLa --max-windows 100
#   # on a GPU node instead:
#   sbatch --partition="$TISIAGO_AGFT_PARTITION" --gres=gpu:1 scripts/run_p6_periodicity.sh \
#       --checkpoint data/p6/runs/condition_hela/best --device gpu

set -euo pipefail

SELF="${BASH_SOURCE[0]}"
[[ -n "${SLURM_JOB_ID:-}" ]] && SELF="$(scontrol show job "$SLURM_JOB_ID" \
    | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"
source "$(cd "$(dirname "$SELF")" && pwd)/_common.sh"
PYTHON="${PYTHON_BIN:-$TISIAGO_AGFT_PYTHON}"
require_python "$PYTHON"

export TMPDIR="$REPO_ROOT/.cache/tmp"
export XDG_CACHE_HOME="$REPO_ROOT/.cache"
export JAX_COMPILATION_CACHE_DIR="$REPO_ROOT/.cache/jax"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
mkdir -p "$TMPDIR" "$JAX_COMPILATION_CACHE_DIR"

cd "$REPO_ROOT"
"$PYTHON" -m tisiago.seq2func.periodicity "$@"
