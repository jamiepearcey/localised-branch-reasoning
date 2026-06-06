#!/usr/bin/env bash
# Pull the reports/ directory back from the box into prototype/reports/.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

DEST="$PROJECT_ROOT/prototype/reports/"
mkdir -p "$DEST"
echo "Fetching reports from $(remote_target) ..."
remote_rsync \
  "$(remote_target):${REMOTE_DIR}/prototype/reports/" \
  "$DEST"
echo "Results in: $DEST"
ls -1 "$DEST" | sed 's/^/  /'
