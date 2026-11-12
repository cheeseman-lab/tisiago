#!/bin/bash
#SBATCH --job-name=mdiberna_tisiago_tracks_train_spliced
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
# (tisiago.tracks.train_spliced, stage ``spliced``). Usage (from the repo root):
#   sbatch --partition=nvidia-A100-20 scripts/run_tracks_train_spliced.sh \
#       --cache data/tracks/txcache/genomic_fc2d_k562 --init data/tracks/runs/tracks.genomic.free.pos.ho-K562.s0/best \
#       --heldout K562 --out data/tracks/txtrain/k562 --export-dir data/tracks/txtrain/export
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
"$PYTHON" -m tisiago.tracks.train_spliced "$@"
