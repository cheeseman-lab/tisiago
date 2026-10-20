#!/bin/bash
#SBATCH --job-name=mdiberna_tisiago_p6fast_loop
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Fast-loop configs on the transcript cache (tisiago.seq2func.fastloop), one results row each.
# Usage (from the repo root):
#   sbatch --partition=nvidia-A100-20 scripts/run_p6_fastloop.sh \
#       configs/p6fast/ref_rnacond.yaml configs/p6fast/t3_foc_lw5.yaml \
#       [--results data/p6/fastloop/results.tsv] [--runs-dir data/p6/fastloop/runs]
#   sbatch --partition=nvidia-A100-20 scripts/run_p6_fastloop.sh --reference

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
"$PYTHON" -m tisiago.seq2func.fastloop "$@"
