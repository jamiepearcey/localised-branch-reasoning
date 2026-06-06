#!/usr/bin/env bash
# Show whether the run is still going and tail its log.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

LINES="${1:-40}"
remote_ssh "REMOTE_DIR='${REMOTE_DIR}' LINES='${LINES}' bash -s" <<'REMOTE'
set -euo pipefail
if tmux has-session -t bdis 2>/dev/null; then
  echo "STATUS: running (tmux session 'bdis' active)"
else
  echo "STATUS: no active 'bdis' session (finished or not started)"
fi
LOG="${REMOTE_DIR}/prototype/run.log"
if [ -f "$LOG" ]; then
  echo "---- last ${LINES} log lines ----"
  tail -n "${LINES}" "$LOG"
else
  echo "no run.log yet at $LOG"
fi
REMOTE
