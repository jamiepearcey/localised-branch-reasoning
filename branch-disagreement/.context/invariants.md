# Invariants

These rules should not be broken without an explicit architectural decision.

- The GPU path is optional. Pure logic (datasets, scoring, metrics, proxy runner)
  must import and be testable without `torch`, `vllm`, `transformers`, or even
  `numpy`. Heavy dependencies live behind the vLLM runner only.
- Always report detection **quality and cost together**. Never present an AUROC
  number without the token / latency budget that produced it.
- Do not claim to beat semantic entropy on detection quality. The thesis is
  *equivalent detection at lower cost*. State it that way.
- Short-answer correctness scoring is deterministic and normalization-based.
  Document the normalization; do not silently change it between runs.
- Be explicit about prototype behaviour versus proven result. Proxy-engine
  numbers are process checks, not quality claims, and must be labelled as such.
- Every baseline (token logprob, lexical agreement, semantic entropy) must be run
  on the *same* generations as the branch-disagreement score, so comparisons are
  paired.

## General agent rules

- Prefer small, reviewable diffs.
- Avoid architecture rewrites without an ADR.
- Update `.context/current-state.md` when meaningful project state changes.
- Add or update an ADR for major architectural decisions.
- Use `docs/tasks/current.md` as the active work queue.
- Run relevant tests or explain why they were not run.
- Never remove context files without explicit instruction.

## Open items

- Add deeper invariants here as the experimental method hardens (e.g. fixed
  decoding temperature, branch count, and dataset slice for headline numbers).
