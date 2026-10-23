#!/bin/bash
#SBATCH --job-name=mdiberna_p6nc_build
#SBATCH --partition=20
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# Per-codon table for the near-cognate diagnosis (val chr7, K562). Submit from the repo root.
set -euo pipefail
set -a; source .env; set +a
export TMPDIR=$PWD/.cache/tmp
"$TISIAGO_CPU_PYTHON" autoresearch/p6nc/build_sites.py --genome "$TISIAGO_GENOME" --out data/p6nc
