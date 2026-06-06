# Agent Instructions

## Required startup path

- Read `docs/index.md`, `.context/project-brief.md`, `.context/current-state.md`, and `.context/invariants.md` before making changes.
- Prefer small, reviewable diffs.
- Avoid architecture rewrites without an ADR.
- Update `.context/current-state.md` when meaningful project state changes.
- Add or update an ADR for major architectural decisions.
- Use `docs/tasks/current.md` as the active work queue.
- Run relevant tests or explain why they were not run.
- Never remove context files without explicit instruction.

## Local expectations

- Treat the project root as the long-lived memory layer.
- Treat `prototype/` as the code layer.
- Keep the GPU path optional: do not add `torch` / `vllm` / `transformers` /
  `numpy` imports to pure-logic modules (datasets, scoring, metrics, proxy).
- Update context and task files when the project state materially changes.
- Record major structural or architectural changes as ADRs.
