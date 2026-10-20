#!/bin/bash
#SBATCH --job-name=mdiberna_tisiago_p6fast_txcache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Transcript-space embedding cache for the P6 fast loop (tisiago.seq2func.txcache).
# Usage (from the repo root):
#   python -m tisiago.seq2func.txcache index                                   # CPU, ~1 min
#   sbatch --partition=20 --gres=none --cpus-per-task=8 --mem=64G \
#       scripts/run_p6_txcache.sh tracks --workers 8                           # CPU
#   sbatch --partition=nvidia-A100-20 scripts/run_p6_txcache.sh embed \
#       --shards val_chr7                                                      # GPU (fits PCA)
#   sbatch --partition=nvidia-A100-20 scripts/run_p6_txcache.sh embed          # remaining shards
# Defaults: --name ar_rnacond_k562_v2, --checkpoint data/p6/runs/ar_rnacond_k562_v2/best.

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
"$PYTHON" -m tisiago.seq2func.txcache "$@"
