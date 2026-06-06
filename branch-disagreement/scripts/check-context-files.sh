#!/bin/sh
set -eu

required_files="
AGENTS.md
CODEX.md
CLAUDE.md
.context/project-brief.md
.context/current-state.md
.context/invariants.md
docs/index.md
docs/architecture/system-overview.md
docs/decisions/ADR-0001-project-memory-structure.md
docs/decisions/ADR-0002-experiment-design.md
docs/workflows/agent-instructions.md
docs/workflows/testing.md
docs/workflows/benchmarking.md
docs/workflows/deployment.md
docs/tasks/backlog.md
docs/tasks/current.md
docs/tasks/task-template.md
"

missing=0
for file in $required_files; do
  if [ ! -f "$file" ]; then
    echo "Missing required context file: $file" >&2
    missing=1
  fi
done

exit $missing
