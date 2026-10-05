#!/bin/bash
#SBATCH --job-name=tisiago_scan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Phase 2 dense scan: embed the all-codon scan manifest (every codon in held-out
# transcripts, from tisiago.enumerate_codons) through the SAME extract.py path as
# the curated run. Reuses the window/position decoupling: one forward pass per tile
# slices every enumerated position, so GPU cost ~ the curated run.
#
# Usage:
#   sbatch --array=0-19%3 --partition="$TISIAGO_AG_PARTITION" \
#       scripts/run_tis_scan.sh TILE_SPEC CONFIG N_SHARDS [MANIFEST] [OUT_DIR]
#
#   TILE_SPEC : ag16k | ag131k | evo2_8k | evo2_8k_s4k | evo2_4k
#               (headline set = ag16k + evo2_8k; latter two are speed ablations)
#   N_SHARDS  : must equal the array size
# Environments and TISIAGO_GENOME come from .env (see scripts/_common.sh);
# PYTHON_BIN overrides the per-backend default.
#
# MEMORY: extraction preallocates one fp16 matrix per requested key and fills it
# batch-wise. Peak host RAM is therefore close to final shard size plus model/manifest
# overhead; the blk28-only config (4 keys) is the production Evo2 configuration.
#
# Scope / size: the scan store is large — ~89 GB at 5632 dims (fp16) for test+val,
# ~57 GB for test only. Calibration uses the CURATED val, so the scan only needs
# TEST transcripts; regenerate test-only with
#   python -m tisiago.enumerate_codons --splits test --out data/scan_manifest.parquet
# and/or restrict to the headline specs (ag16k + evo2_8k) to bound the footprint.

set -euo pipefail

SELF="${BASH_SOURCE[0]}"
[[ -n "${SLURM_JOB_ID:-}" ]] && SELF="$(scontrol show job "$SLURM_JOB_ID" \
    | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"
source "$(cd "$(dirname "$SELF")" && pwd)/_common.sh"
TILE_SPEC="$1"
CONFIG="$2"
N_SHARDS="$3"
MANIFEST="${4:-$REPO_ROOT/data/scan_manifest.parquet}"
OUT_DIR="${5:-$REPO_ROOT/data/scan_parts}"

GENOME="${TISIAGO_GENOME:?Set TISIAGO_GENOME to an indexed reference FASTA}"

if [[ "$TILE_SPEC" == ag* ]]; then
    DEFAULT_PYTHON="$TISIAGO_AG_PYTHON"
else
    DEFAULT_PYTHON="$TISIAGO_EVO_PYTHON"
fi
PYTHON="${PYTHON_BIN:-$DEFAULT_PYTHON}"
[[ -x "$PYTHON" ]] || {
    echo "Python not found at $PYTHON; check TISIAGO_AG_PYTHON/TISIAGO_EVO_PYTHON in .env" >&2
    exit 2
}
"$PYTHON" -c "import gruyerenome, pyfaidx" || {
    echo "This environment lacks gruyerenome/pyfaidx" >&2
    exit 2
}

cd "$REPO_ROOT"
"$PYTHON" -m tisiago.extract \
    --manifest "$MANIFEST" \
    --tile-spec "$TILE_SPEC" \
    --config "$CONFIG" \
    --genome "$GENOME" \
    --out-dir "$OUT_DIR" \
    --shard-id "${SLURM_ARRAY_TASK_ID}" \
    --n-shards "$N_SHARDS" \
    --resume
