#!/bin/bash
#SBATCH --job-name=mdiberna_tisiago_p6_txtrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# AlphaGenome fine-tuned on spliced-transcript input + transcript-space head
# (tisiago.seq2func.txtrain, P6 step 2). Usage (from the repo root):
#   sbatch --partition=nvidia-A100-20 scripts/run_p6_txtrain.sh \
#       --cache data/p6/txcache/fc2d_k562 --init data/p6/runs/ar_rnacond_k562_fc2d_l2w300/best \
#       --heldout K562 --out data/p6/txtrain/k562 --export-dir data/p6/txtrain/export
#   # A6000 / L40S: add --max-bucket 16384. Rerun the same command to resume from <out>/last.

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
"$PYTHON" -m tisiago.seq2func.txtrain "$@"
