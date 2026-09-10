#!/bin/bash
#SBATCH --job-name=tisiago_txp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#
# Exon-spliced, transcript-oriented Evo2 extraction as a SLURM array job.
# This is a separate experimental representation (evo2/TXP), not a drop-in
# replacement for the established genomic evo2/W8k features.
#
# Usage:
#   sbatch --array=0-19%2 --partition="$TISIAGO_EVO_PARTITION" \
#       scripts/run_tis_extract_transcript.sh CONFIG N_SHARDS \
#       [MANIFEST] [OUT_DIR] [BATCH_SIZE] [BUCKET_SIZE] [HEAD_ARTIFACT ...]
# Set TISIAGO_GENOME and TISIAGO_GTF to the matching indexed reference files.
# Supplying head artifacts switches output from full vectors to additive TXP logits.

set -euo pipefail

REPO_ROOT="${TISIAGO_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
CONFIG="$1"
N_SHARDS="$2"
MANIFEST="${3:-$REPO_ROOT/data/manifest.parquet}"
OUT_DIR="${4:-$REPO_ROOT/data/txp_parts}"
BATCH_SIZE="${5:-0}"
BUCKET_SIZE="${6:-0}"
HEAD_ARTIFACTS=("${@:7}")

GENOME="${TISIAGO_GENOME:?Set TISIAGO_GENOME to an indexed reference FASTA}"
GTF="${TISIAGO_GTF:?Set TISIAGO_GTF to the matching transcript annotation}"

PYTHON="${PYTHON_BIN:-$REPO_ROOT/.venv/evo2-next/bin/python}"
[[ -x "$PYTHON" ]] || {
    echo "Python not found at $PYTHON; set PYTHON_BIN to the uv-managed Evo2 environment" >&2
    exit 2
}
"$PYTHON" -c "import gruyerenome, pyfaidx" || {
    echo "Install extraction dependencies with uv before submitting this job" >&2
    exit 2
}

HEAD_ARGS=()
for artifact in "${HEAD_ARTIFACTS[@]}"; do
    HEAD_ARGS+=(--head-artifact "$artifact")
done

cd "$REPO_ROOT"
"$PYTHON" -m tisiago.extract_transcript \
    --manifest "$MANIFEST" \
    --config "$CONFIG" \
    --genome "$GENOME" \
    --gtf "$GTF" \
    --out-dir "$OUT_DIR" \
    --shard-id "${SLURM_ARRAY_TASK_ID}" \
    --n-shards "$N_SHARDS" \
    --batch-size "$BATCH_SIZE" \
    --bucket-size "$BUCKET_SIZE" \
    "${HEAD_ARGS[@]}" \
    --resume
