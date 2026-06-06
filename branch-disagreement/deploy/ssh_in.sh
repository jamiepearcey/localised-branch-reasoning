#!/usr/bin/env bash
# Open an interactive shell on the box (e.g. to `tmux attach -t bdis`).
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
exec ssh -t "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
  "cd ${REMOTE_DIR} 2>/dev/null; exec \$SHELL -l"
