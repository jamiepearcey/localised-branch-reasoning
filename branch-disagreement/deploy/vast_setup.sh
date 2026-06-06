#!/usr/bin/env bash
# Remote one-time setup: venv, GPU deps, and model prefetch.
# Idempotent — safe to re-run after a redeploy.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

echo "Setting up environment on $(remote_target) ..."

remote_ssh "REMOTE_DIR='${REMOTE_DIR}' MODEL='${MODEL:-}' NLI_MODEL='${NLI_MODEL:-}' \
            HF_TOKEN='${HF_TOKEN:-}' bash -s" <<'REMOTE'
set -euo pipefail
cd "${REMOTE_DIR}/prototype"

echo "== python =="
python3 --version

# Bare images lack venv/pip/tmux; PyTorch images already have them. Install
# prerequisites idempotently either way.
if ! python3 -m venv --help >/dev/null 2>&1 || ! command -v tmux >/dev/null 2>&1; then
  echo "== installing OS prerequisites (python3-venv, pip, tmux) =="
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y python3-venv python3-pip tmux git
fi

# Create a venv. --system-site-packages lets us reuse a base image's torch when
# present (PyTorch templates); on a bare image vLLM installs its own matched
# torch into the venv.
if [ ! -d "${REMOTE_DIR}/.venv" ]; then
  python3 -m venv --system-site-packages "${REMOTE_DIR}/.venv"
fi
# shellcheck disable=SC1091
. "${REMOTE_DIR}/.venv/bin/activate"

pip install --upgrade pip wheel
# Pin a CUDA 12.8 stack. cu128 wheels run on driver >= 570 (this box) AND on
# newer driver-590/CUDA-13 boxes (backward compatible), so this is the safe
# universal choice. Auto-pulled vLLM uses CUDA-13 torch, which fails on older
# drivers — hence the explicit pins. Pre-install torch so vLLM reuses it.
echo "== installing torch 2.7.1 (CUDA 12.8) =="
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
echo "== installing vLLM 0.10.1.1 + deps (this can take a few minutes) =="
# Pin transformers <5: vLLM 0.10 is built for transformers 4.x. transformers 5.x
# removed slow-tokenizer attrs (all_special_tokens_extended) vLLM relies on, and
# the --system-site-packages venv can otherwise expose a base-image 5.x.
pip install vllm==0.10.1.1 "transformers>=4.55,<5" "huggingface_hub<1" \
  datasets accelerate

if [ -n "${HF_TOKEN}" ]; then
  python3 - <<PY
from huggingface_hub import login
login(token="${HF_TOKEN}")
print("hf login ok")
PY
fi

echo "== prefetching models (optional; speeds the first run) =="
python3 - <<PY || echo "prefetch skipped/failed (run will download on first use)"
from huggingface_hub import snapshot_download
import os
for repo in [os.environ.get("MODEL",""), os.environ.get("NLI_MODEL","")]:
    if repo:
        print("prefetch", repo)
        snapshot_download(repo)
PY

echo "== smoke import check =="
python3 -c "import vllm, transformers, datasets; print('vllm', vllm.__version__)"
echo "SETUP OK"
REMOTE

echo "Remote setup complete."
