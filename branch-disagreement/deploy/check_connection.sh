#!/usr/bin/env bash
# Verify SSH works and the GPU is visible on the rented box.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

echo "Connecting to $(remote_target) :$SSH_PORT ..."
remote_ssh 'echo "ok: $(hostname)"; echo "python: $(python3 --version 2>&1)"; \
  if command -v nvidia-smi >/dev/null 2>&1; then \
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; \
  else echo "WARNING: nvidia-smi not found — is this a GPU instance?"; fi'
echo "Connection OK."
