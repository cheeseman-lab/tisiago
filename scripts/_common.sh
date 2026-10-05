# Shared header for tisiago scripts. Source it right after `set -euo pipefail`
# (sbatch runs a spooled copy, so resolve the original path via scontrol):
#
#   SELF="${BASH_SOURCE[0]}"
#   [[ -n "${SLURM_JOB_ID:-}" ]] && SELF="$(scontrol show job "$SLURM_JOB_ID" \
#       | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"
#   source "$(cd "$(dirname "$SELF")" && pwd)/_common.sh"
#
# Sets REPO_ROOT (robust to sbatch spooling the script and to submitting from
# another directory), loads REPO_ROOT/.env, and fills the per-stage Python
# defaults (TISIAGO_{CPU,AG,EVO}_PYTHON). Values already in the environment win.

if [[ -z "${TISIAGO_REPO:-}" ]]; then
    TISIAGO_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
REPO_ROOT="$TISIAGO_REPO"
export TISIAGO_REPO REPO_ROOT

if [[ -f "$REPO_ROOT/.env" ]]; then
    while IFS='=' read -r key value; do
        [[ -z "$key" || "$key" == \#* ]] && continue
        [[ -n "${!key:-}" ]] || export "$key=$value"
    done < "$REPO_ROOT/.env"
fi

: "${TISIAGO_EVO_PYTHON:=$REPO_ROOT/.venv/evo2-next/bin/python}"
: "${TISIAGO_AG_PYTHON:=$(command -v python)}"
: "${TISIAGO_CPU_PYTHON:=$(command -v python)}"
export TISIAGO_EVO_PYTHON TISIAGO_AG_PYTHON TISIAGO_CPU_PYTHON

mkdir -p "$REPO_ROOT/logs"

require_python() {
    [[ -x "$1" ]] || {
        echo "Python not found at $1; check TISIAGO_*_PYTHON in $REPO_ROOT/.env" >&2
        exit 2
    }
}
