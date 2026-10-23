#!/bin/bash
#SBATCH --job-name=mdiberna_tisiago_p6fast_seq
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Usage (repo root): sbatch autoresearch/p6fast/run_build_seq.sh CACHE [CACHE ...]
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source .env; set +a
.venv/agft/bin/python autoresearch/p6fast/build_seq.py --genome "$TISIAGO_GENOME" "$@"
