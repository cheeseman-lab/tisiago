#!/bin/bash
#SBATCH --job-name=tisiago_evo2_setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Build the isolated Evo2 extraction environment with uv and validate its CUDA
# imports on the allocated GPU. The shared conda Evo2 environment is never
# modified. Run from the tisiago repository root:
#
#   sbatch --partition=YOUR_GPU_PARTITION \
#     scripts/run_tis_evo2_env_setup.sh [.venv/evo2-next]

set -euo pipefail

# Slurm copies submitted scripts into /var/spool. Prefer its recorded submit
# directory so editable installs still resolve to the real checkout.
SELF="${BASH_SOURCE[0]}"
[[ -n "${SLURM_JOB_ID:-}" ]] && SELF="$(scontrol show job "$SLURM_JOB_ID" \
    | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"
source "$(cd "$(dirname "$SELF")" && pwd)/_common.sh"
REPO="$REPO_ROOT"
cd "$REPO"
ENV_PREFIX="${1:-$REPO/.venv/evo2-next}"
if [[ "$ENV_PREFIX" != /* ]]; then
    ENV_PREFIX="$REPO/$ENV_PREFIX"
fi
BASE_PYTHON="${BASE_PYTHON:-python3.11}"
GRUYERENOME="${GRUYERENOME:-$REPO/../gruyerenome}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO/.cache/uv}"
PYTHON_BIN="$ENV_PREFIX/bin/python"
command -v uv >/dev/null || {
    echo "uv is required; install it from https://docs.astral.sh/uv/" >&2
    exit 2
}
command -v "$BASE_PYTHON" >/dev/null || {
    echo "Base Python not found: $BASE_PYTHON (set BASE_PYTHON to Python 3.11+)" >&2
    exit 2
}
[[ -f "$GRUYERENOME/pyproject.toml" ]] || {
    echo "gruyerenome checkout not found at $GRUYERENOME; set GRUYERENOME" >&2
    exit 2
}

# The wheel channel is configurable for the local CUDA driver. The pinned package
# versions must continue to match gruyerenome's declared Evo2 extra.
TORCH_VERSION="2.7.0"
EVO2_VERSION="0.6.0"
VTX_VERSION="1.1.0"
FLASH_ATTN_VERSION="2.8.0.post2"
TORCH_BACKEND="${TISIAGO_TORCH_BACKEND:-cu126}"

export PYTHONNOUSERSITE=1 PIP_USER=0

if [[ ! -x "$PYTHON_BIN" ]]; then
    uv venv --python "$BASE_PYTHON" "$ENV_PREFIX"
fi

echo "=== [1/5] Torch $TORCH_VERSION ($TORCH_BACKEND) ==="
uv pip install \
    --python "$PYTHON_BIN" \
    --cache-dir "$UV_CACHE_DIR" \
    --link-mode copy \
    --torch-backend "$TORCH_BACKEND" \
    "torch==$TORCH_VERSION" packaging setuptools wheel

echo "=== [2/5] FlashAttention $FLASH_ATTN_VERSION for this GPU ==="
# Source-build when an importable matching wheel is unavailable. Infer the
# architecture and CUDA toolkit from the allocated node unless explicitly set.
# Skip this expensive step when an already-installed extension imports cleanly.
if "$PYTHON_BIN" -c "import flash_attn; assert flash_attn.__version__.split('+')[0] == '$FLASH_ATTN_VERSION'" >/dev/null 2>&1; then
    echo "FlashAttention import already passes; keeping the installed build"
else
    if [[ -z "${CUDA_HOME:-}" ]]; then
        NVCC="$(command -v nvcc || true)"
        [[ -n "$NVCC" ]] || {
            echo "nvcc not found; set CUDA_HOME to a CUDA toolkit" >&2
            exit 2
        }
        CUDA_HOME="$(cd "$(dirname "$NVCC")/.." && pwd)"
    fi
    export CUDA_HOME
    export FLASH_ATTENTION_FORCE_BUILD=TRUE
    if [[ -z "${FLASH_ATTN_CUDA_ARCHS:-}" ]]; then
        FLASH_ATTN_CUDA_ARCHS="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')"
    fi
    export FLASH_ATTN_CUDA_ARCHS
    export MAX_JOBS="${MAX_JOBS:-3}"
    export NVCC_THREADS="${NVCC_THREADS:-2}"
    [[ -x "$CUDA_HOME/bin/nvcc" ]] || {
        echo "CUDA compiler not found at $CUDA_HOME/bin/nvcc" >&2
        exit 1
    }
    uv pip install \
        --python "$PYTHON_BIN" \
        --cache-dir "$UV_CACHE_DIR" \
        --link-mode copy \
        --reinstall-package flash-attn \
        --no-binary flash-attn \
        --no-build-isolation-package flash-attn \
        "flash-attn==$FLASH_ATTN_VERSION"
fi

echo "=== [3/5] validated dependency lock ==="
uv pip install \
    --python "$PYTHON_BIN" \
    --cache-dir "$UV_CACHE_DIR" \
    --link-mode copy \
    --torch-backend "$TORCH_BACKEND" \
    -r "$REPO/requirements-evo2.lock"

echo "=== [4/5] editable project source (without re-resolving the lock) ==="
# gruyerenome owns the Evo2 forward pass; tisiago owns orchestration,
# extraction, evaluation, and the integration tests.
uv pip install \
    --python "$PYTHON_BIN" \
    --cache-dir "$UV_CACHE_DIR" \
    --link-mode copy \
    --no-deps \
    -e "$GRUYERENOME" \
    -e "$REPO"

echo "=== [5/5] dependency and GPU import checks ==="
uv pip check --python "$PYTHON_BIN"
"$PYTHON_BIN" -c 'import flash_attn, torch; assert torch.cuda.is_available(); print("torch", torch.__version__, "cuda", torch.version.cuda); print("flash-attn", flash_attn.__version__); print("gpu", torch.cuda.get_device_name(0))'
"$PYTHON_BIN" -c "import importlib.metadata as m; assert m.version('evo2') == '$EVO2_VERSION'; assert m.version('vtx') == '$VTX_VERSION'; print(*[f'{name}=={m.version(name)}' for name in ('evo2', 'vtx', 'gruyerenome', 'tisiago', 'numpy')], sep='\n')"

echo "BUILD-OK: PYTHON_BIN=$PYTHON_BIN"
