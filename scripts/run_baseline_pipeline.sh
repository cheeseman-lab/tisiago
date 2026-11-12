#!/bin/bash
# Orchestrate the baseline embedding store: two per-candidate extraction arrays
# (AlphaGenome 16k + 131k) running in parallel, then a single assemble job once both
# complete.
#
# Assumes data/manifest.parquet already exists (produced by swissisoform).
#
# Usage: bash scripts/run_baseline_pipeline.sh [N_SHARDS]
# Required (from .env): TISIAGO_GENOME and ALPHAGENOME_WEIGHTS_PATH.
# Optional scheduler settings: TISIAGO_AG_PARTITION and TISIAGO_AG_CONCURRENCY.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
cd "$REPO_ROOT"

N="${1:-20}"
LAST=$((N - 1))
: "${TISIAGO_GENOME:?Set TISIAGO_GENOME to an indexed reference FASTA}"
: "${ALPHAGENOME_WEIGHTS_PATH:?Set ALPHAGENOME_WEIGHTS_PATH to the AlphaGenome checkpoint directory}"
AG_PYTHON="$TISIAGO_AG_PYTHON"
CPU_PYTHON="$TISIAGO_CPU_PYTHON"
for executable in "$AG_PYTHON" "$CPU_PYTHON"; do
    [[ -x "$executable" ]] || {
        echo "Python not found at $executable; check TISIAGO_*_PYTHON in .env" >&2
        exit 2
    }
done

AG_SCHEDULER=()
if [[ -n "${TISIAGO_AG_PARTITION:-}" ]]; then
    AG_SCHEDULER+=(--partition="$TISIAGO_AG_PARTITION")
fi
AG_CONCURRENCY="${TISIAGO_AG_CONCURRENCY:-3}"

A=$(sbatch --parsable --array="0-${LAST}%${AG_CONCURRENCY}" "${AG_SCHEDULER[@]}" \
    --export=ALL,TISIAGO_REPO="$REPO_ROOT",PYTHON_BIN="$AG_PYTHON" \
    scripts/run_baseline_extract.sh ag16k configs/tis_alphagenome_16k.yaml "$N")
echo "ag16k   array job: $A"

B=$(sbatch --parsable --array="0-${LAST}%${AG_CONCURRENCY}" "${AG_SCHEDULER[@]}" \
    --export=ALL,TISIAGO_REPO="$REPO_ROOT",PYTHON_BIN="$AG_PYTHON" \
    scripts/run_baseline_extract.sh ag131k configs/tis_alphagenome_131k.yaml "$N")
echo "ag131k  array job: $B"

D=$(sbatch --parsable --dependency="afterok:${A}:${B}" \
    --export=ALL,TISIAGO_REPO="$REPO_ROOT",PYTHON_BIN="$CPU_PYTHON" \
    scripts/run_baseline_assemble.sh)
echo "assemble job (after all extraction): $D"
echo "Monitor: squeue -u \$USER"
