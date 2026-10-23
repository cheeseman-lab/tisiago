#!/bin/bash
#SBATCH --job-name=mdiberna_p6nc_probe
#SBATCH --partition=20
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# Q1 logistic probes (val chr7, K562). Submit from the repo root; extra args go to probe.py.
set -euo pipefail
set -a; source .env; set +a
export TMPDIR=$PWD/.cache/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$TISIAGO_CPU_PYTHON" autoresearch/p6nc/probe.py --data data/p6nc "$@"
