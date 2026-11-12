#!/bin/bash
#SBATCH --job-name=mdiberna_tisiago_tracks_evaluate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Usage (from the repo root):
#   sbatch --partition="$TISIAGO_AGFT_PARTITION" scripts/run_tracks_evaluate.sh \
#       --checkpoint data/tracks/runs/tracks.genomic.free.pos.ho-HeLa.s0/best \
#       --arm rnacond:HeLa --split val
#   # held-out chromosomes, only for the final numbers:
#   sbatch --partition="$TISIAGO_AGFT_PARTITION" scripts/run_tracks_evaluate.sh \
#       --checkpoint data/tracks/runs/tracks.genomic.free.pos.ho-HeLa.s0/best \
#       --arm rnacond:HeLa --split test --i-am-final
#   # replicate ceiling only (CPU; drop the GPU request):
#   sbatch --partition=20 --gres=none scripts/run_tracks_evaluate.sh --ceiling-only --split val

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
"$PYTHON" -m tisiago.tracks.evaluate "$@"
