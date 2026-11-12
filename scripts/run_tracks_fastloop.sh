#!/bin/bash
#SBATCH --job-name=mdiberna_tisiago_tracks_fastloop
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Fast-loop configs on the transcript cache (tisiago.tracks.fastloop), one results row each.
# Usage (from the repo root):
#   sbatch --partition=nvidia-A100-20 scripts/run_tracks_fastloop.sh \
#       configs/tracks/spliced_flux_pos.yaml my_variant.yaml \
#       [--results data/tracks/fastloop/results.tsv] [--runs-dir data/tracks/fastloop/runs]
#   sbatch --partition=nvidia-A100-20 scripts/run_tracks_fastloop.sh --reference
#   # final, once per fold: a saved head (--export-dir run) on a test cache (txcache --test)
#   sbatch --partition=nvidia-A100-20 scripts/run_tracks_fastloop.sh \
#       --predict data/tracks/fastloop/export/<name> --cache data/tracks/txcache/<x>_test \
#       --split test --i-am-final --export-dir data/tracks/fastloop/export_test
#       # -> data/tracks/fastloop/results_test.tsv, runs_test/<name>-<hash>-test.json

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
"$PYTHON" -m tisiago.tracks.fastloop "$@"
