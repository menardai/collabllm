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

# --- Config (edit these later) ---
OPENAI_KEY="fake_key"
VENV_DIR="/data/venvs/"
REPO_DIR="/data/collabllm"

# --- Basics & deps ---
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends curl ca-certificates git python3 python3-venv python3-pip emacs-nox

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
uv pip install --upgrade pip
# Install required packages
uv pip install --no-cache-dir nvidia-ml-py3

# Install Unsloth
uv pip install --no-cache-dir unsloth

# Install collabllm
uv pip install --no-cache-dir ..
uv pip install --no-cache-dir hf_transfer

# Install DeepSpeed
#uv pip install --no-cache-dir deepspeed

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

# Optional: torch CUDA
section("PyTorch CUDA (optional)")
try:
    import torch
    print("torch.version:", torch.__version__)
    print("torch.version.cuda:", getattr(torch.version, "cuda", None))
    print("torch.cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch.cuda.device_count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"CUDA device {i}:", torch.cuda.get_device_name(i))
except Exception as e:
    print("PyTorch not available or failed to query:", repr(e))
PYCODE

echo "Setup complete."
