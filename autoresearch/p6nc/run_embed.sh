#!/bin/bash
#SBATCH --job-name=mdiberna_p6nc_embed
#SBATCH --partition=nvidia-A100-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# Base + final fine-tuned AlphaGenome 1-bp embeddings at every codon of data/p6nc/codons.parquet.
set -euo pipefail
set -a; source .env; set +a
export TMPDIR=$PWD/.cache/tmp
"$TISIAGO_AGFT_PYTHON" autoresearch/p6nc/embed_codons.py --out data/p6nc
