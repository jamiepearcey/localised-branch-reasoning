#!/usr/bin/env bash
# Rsync the prototype code to the remote box. Excludes run artifacts and caches.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

echo "Creating ${REMOTE_DIR} on $(remote_target) ..."
remote_ssh "mkdir -p ${REMOTE_DIR}"

echo "Syncing prototype/ ..."
remote_rsync \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.venv/' \
  --exclude 'reports/*' \
  --exclude 'data/cache/' \
  "$PROJECT_ROOT/prototype/" \
  "$(remote_target):${REMOTE_DIR}/prototype/"

echo "Deploy complete -> ${REMOTE_DIR}/prototype"
