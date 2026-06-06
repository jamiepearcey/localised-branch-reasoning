# ADR-0001: Adopt Persistent Project Memory Structure

## Status

Accepted

## Context

This project needs durable context that survives across CLI, headless, and
multi-agent coding sessions. The implementation directory alone is not enough to
preserve current state, invariants, active work, and architectural intent. The
structure mirrors the sibling `localised-reasoning` project so agents move
between them without relearning conventions.

## Decision

Adopt a version-controlled project memory structure at the project root
consisting of:

- `AGENTS.md`, `CODEX.md`, and `CLAUDE.md` for agent instructions
- `.context/` for project brief, current state, and invariants
- `docs/` for architecture notes, workflows, decisions, and tasks
- `scripts/check-context-files.sh` as a lightweight presence check

## Consequences

- Agents have a stable entry point before editing.
- Project context evolves independently of the inner implementation.
- Architectural intent and active work are easier to preserve across sessions.
- These files must be maintained as part of normal project hygiene.
