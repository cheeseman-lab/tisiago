#!/bin/bash
# Orchestrate the full TIS embedding store: three per-candidate extraction arrays
# (AlphaGenome 16k + 131k on A6000, Evo-2 8k on A100) running in parallel, then a
# single assemble job once all three complete.
#
# Assumes data/manifest.parquet already exists (produced by swissisoform).
#
# Usage: bash scripts/run_tis_pipeline.sh [N_SHARDS]

set -euo pipefail
cd "$(dirname "$0")/.."

N="${1:-20}"
LAST=$((N - 1))

A=$(sbatch --parsable --array=0-${LAST}%3 --partition=nvidia-A6000-20 \
    scripts/run_tis_extract.sh ag16k  configs/tis_alphagenome_16k.yaml  alphagenome "$N")
echo "ag16k   array job: $A"

B=$(sbatch --parsable --array=0-${LAST}%3 --partition=nvidia-A6000-20 \
    scripts/run_tis_extract.sh ag131k configs/tis_alphagenome_131k.yaml alphagenome "$N")
echo "ag131k  array job: $B"

C=$(sbatch --parsable --array=0-${LAST}%2 --partition=nvidia-A100-20 \
    scripts/run_tis_extract.sh evo2_8k configs/tis_evo2_8k.yaml evo2 "$N")
echo "evo2_8k array job: $C"

D=$(sbatch --parsable --dependency=afterok:${A}:${B}:${C} scripts/run_tis_assemble.sh)
echo "assemble job (after all extraction): $D"
echo "Monitor: squeue -u \$USER"
