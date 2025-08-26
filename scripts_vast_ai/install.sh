#!/usr/bin/env bash

set -euo pipefail

# --- Timezone ---
# Ensure tzdata exists if zoneinfo is missing (some base images)
if [ ! -e /usr/share/zoneinfo/America/Montreal ]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y --no-install-recommends tzdata
fi
ln -snf /usr/share/zoneinfo/America/Montreal /etc/localtime
echo "America/Montreal" > /etc/timezone

# --- Python version check (Unsloth compatibility) ---
echo "🐍 Checking Python version compatibility..."
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Detected Python version: $PYTHON_VERSION"

case "$PYTHON_VERSION" in
  "3.10"|"3.11"|"3.12")
    echo "✅ Python $PYTHON_VERSION is compatible with Unsloth"
    ;;
  *)
    echo "❌ ERROR: Python $PYTHON_VERSION is not supported by Unsloth"
    echo "Unsloth requires Python 3.10, 3.11, or 3.12"
    echo "Please use a compatible Python version or update your base image"
    exit 1
    ;;
esac

# --- GPU compatibility check ---
echo "🎮 Checking GPU compatibility..."
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_INFO=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -n "$GPU_INFO" ]; then
    GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
    COMPUTE_CAP=$(echo "$GPU_INFO" | cut -d',' -f2 | xargs)
    echo "Detected GPU: $GPU_NAME (Compute Capability: $COMPUTE_CAP)"
    
    # Check if compute capability is >= 7.0 (minimum for Unsloth)
    if [ "$(echo "$COMPUTE_CAP >= 7.0" | bc -l 2>/dev/null || echo "0")" = "1" ]; then
      echo "✅ GPU is compatible with Unsloth (requires compute capability >= 7.0)"
    else
      echo "⚠️  WARNING: GPU compute capability $COMPUTE_CAP may not be optimal for Unsloth"
      echo "Unsloth works best with compute capability >= 7.0 (V100, T4, RTX 20/30/40/50 series)"
    fi
  else
    echo "⚠️  Could not detect GPU compute capability"
  fi
else
  echo "⚠️  nvidia-smi not found - GPU compatibility cannot be verified"
fi

# --- Config (edit these later) ---
OPENAI_KEY="fake_key"
VENV_DIR="/data/venvs/"
REPO_DIR="/data/collabllm"

# --- Basics & deps ---
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends curl ca-certificates git python3 python3-venv python3-pip emacs-nox bc

# --- Install uv (Python package manager) ---
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
  if ! grep -q 'HOME/.local/bin' "${HOME}/.bashrc" 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "${HOME}/.bashrc"
  fi
fi

# --- Python venv via uv ---
mkdir -p "$(dirname "$VENV_DIR")"
if [ ! -d "$VENV_DIR" ]; then
  uv venv "$VENV_DIR"
fi
# shellcheck source=/dev/null
. "${VENV_DIR}/bin/activate"

# --- Install packages ---
# Bootstrap pip (uv venvs may not include pip by default)
python -m ensurepip --upgrade || true
python -m pip install --upgrade pip

# Install stable PyTorch stack for compatibility with Unsloth + xFormers
echo "🔥 Installing stable PyTorch 2.4.0 with CUDA 12.1..."
# uv pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install Unsloth first (it will handle xFormers compatibility automatically)
echo "🦙 Installing Unsloth for CUDA 12.1 and PyTorch 2.4.0..."
uv pip install --no-cache-dir --upgrade pip
# uv pip install --no-cache-dir "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
# Ampere devices (A100, H100, RTX 3090) special case
# uv pip install "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
export UNSLOTH_FORCE_ATTN=sdpa
uv pip install --no-cache-dir "unsloth @ git+https://github.com/unslothai/unsloth.git"

# Install collabllm and other required packages
echo "📦 Installing utilities..."
uv pip install --no-cache-dir nvidia-ml-py3
uv pip install --no-cache-dir -U "huggingface_hub[cli]"

# Install my local collabllm repo
uv pip install --no-cache-dir -e ..

# uninstall torchao to avoid conflict with unsloth
uv pip uninstall torchao

# --- Set OpenAI key as env var (dummy; replace later) ---
export OPENAI_API_KEY="${OPENAI_KEY}"

# Persist so SSH/Jupyter shells can see it (per Vast.ai doc)
# (Vast notes that envs set at launch are only visible to onstart/entrypoint,
# so export to /etc/environment for interactive shells.)
if ! grep -q '^OPENAI_API_KEY=' /etc/environment 2>/dev/null; then
  echo "OPENAI_API_KEY=${OPENAI_KEY}" >> /etc/environment
fi
# If you prefer the blanket approach Vast suggests:
# env | grep _ >> /etc/environment

# --- Print Vast container label if present (handy for logging) ---
echo "VAST_CONTAINERLABEL: ${VAST_CONTAINERLABEL:-<unset>}"

# --- Show nvidia-smi (driver & GPUs visibility) ---
echo "===== nvidia-smi ====="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || echo "[WARN] nvidia-smi returned non-zero status."
else
  echo "[WARN] nvidia-smi not found in PATH."
fi

# --- Emacs (terminal) usage hint ---
cat <<'EMACS_HELP'
Emacs (terminal-only) is installed.
Use without windowing:
  emacs -nw /etc/environment
Shortcuts: Save = C-x C-s, Exit = C-x C-c
EMACS_HELP

# --- Validate from Python that the OPENAI_API_KEY is readable ---
echo "===== Validate OPENAI_API_KEY via Python ====="
python - <<'PYCODE'
import os
print("OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY") or "<MISSING>")
PYCODE

# --- Validate GPU access via NVML; optionally report CUDA via torch/cupy/jax if available ---
echo "===== Validate GPU from Python (NVML + optional CUDA libs) ====="
python - <<'PYCODE'
import os, sys

def section(title):
    print("\n--- " + title + " ---")

# NVML check (no PyTorch/JAX/CuPy required)
section("NVML (nvidia-ml-py3)")
try:
    import pynvml as nvml
    nvml.nvmlInit()
    count = nvml.nvmlDeviceGetCount()
    print(f"GPU count: {count}")
    for i in range(count):
        h = nvml.nvmlDeviceGetHandleByIndex(i)
        name = nvml.nvmlDeviceGetName(h).decode("utf-8")
        mem = nvml.nvmlDeviceGetMemoryInfo(h)
        print(f"[{i}] {name} | Mem: {mem.total/1024**3:.2f} GiB total, {mem.free/1024**3:.2f} GiB free")
    nvml.nvmlShutdown()
except Exception as e:
    print("NVML check failed:", repr(e))

# PyTorch CUDA + xFormers + Unsloth compatibility check
section("PyTorch CUDA + xFormers + Unsloth")
try:
    import torch
    import xformers
    import unsloth
    print("torch.version:", torch.__version__)
    print("xformers.version:", xformers.__version__)
    print("unsloth.version:", unsloth.__version__)
    print("torch.version.cuda:", getattr(torch.version, "cuda", None))
    print("torch.cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch.cuda.device_count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"CUDA device {i}:", torch.cuda.get_device_name(i))
    print("✅ All ML packages compatible and working!")
except Exception as e:
    print("❌ ML package compatibility issue:", repr(e))
PYCODE

# --- Hugging Face cache setup ---
export HF_HOME="/data/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$TRANSFORMERS_CACHE"
# Persist for future shells
if ! grep -q 'HF_HOME=' /etc/environment 2>/dev/null; then
  {
    echo "HF_HOME=$HF_HOME"
    echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
    echo "HF_HUB_ENABLE_HF_TRANSFER=1"
  } >> /etc/environment
fi

echo "Setup complete."
