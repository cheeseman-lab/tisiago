#!/bin/bash
#SBATCH --job-name=mdiberna_tisiago_agft_setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Build the alphagenome-ft env from requirements-agft.lock.
# Usage: sbatch --partition=20 scripts/run_p6_agft_env_setup.sh [.venv/agft]

set -euo pipefail

SELF="${BASH_SOURCE[0]}"
[[ -n "${SLURM_JOB_ID:-}" ]] && SELF="$(scontrol show job "$SLURM_JOB_ID" \
    | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"
source "$(cd "$(dirname "$SELF")" && pwd)/_common.sh"
cd "$REPO_ROOT"
ENV_PREFIX="${1:-$REPO_ROOT/.venv/agft}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.cache/uv}"
export TMPDIR="$REPO_ROOT/.cache/tmp"
mkdir -p "$TMPDIR"
uv venv --python 3.11 "$ENV_PREFIX"
uv pip install --python "$ENV_PREFIX/bin/python" -r requirements-agft.lock
uv pip install --python "$ENV_PREFIX/bin/python" --no-deps -e .
"$ENV_PREFIX/bin/python" -c "import alphagenome_ft, jax, tisiago; print('agft', alphagenome_ft.__version__, 'jax', jax.__version__)"
