# ADR-0002: Experiment Design for the Hallucination-Detection Claim

## Status

Accepted

## Context

The sibling `localised-reasoning` project produced an interesting concept but
weak proof, partly because experiments were bound to a slow local Metal path and
used short multiple-choice tasks that do not exercise the architecture's intended
use. This project exists to prove (or refute) one specific, well-scoped claim
cleanly and cheaply.

## Decision

1. **Single headline claim.** Branch disagreement over prefix-shared branches
   predicts answer correctness with AUROC competitive with full semantic entropy
   and clearly above token-probability confidence, at lower token cost.

2. **Metric.** AUROC of each score versus a binary "answer is wrong" label, with
   a bootstrap confidence interval, and a DeLong test comparing the
   branch-disagreement detector against each baseline on the same questions.

3. **Baselines, paired.** All baselines are computed from the *same* generations:
   mean token logprob, lexical self-consistency agreement, and full semantic
   entropy. No baseline gets its own separate sampling pass.

4. **Dataset.** PopQA as primary (short factual answers with a popularity axis,
   so the detector can be shown to fire hardest on obscure facts). TriviaQA as a
   secondary check. Deterministic normalization-based correctness scoring.

5. **Cost axis.** Record total generated tokens and wall-clock per question for
   each method. The result is a two-axis frontier (AUROC vs cost), not a single
   number.

6. **Hardware independence of the claim.** The conceptual claim is validated with
   any open-weights model via vLLM on a rented GPU; it is deliberately decoupled
   from the Metal KV-fork efficiency engineering, which is a separate systems
   question.

## Consequences

- The project can produce a defensible result for a small compute budget.
- A negative result (disagreement does not beat naive confidence) is still a
  valid, publishable-shaped outcome and must be reported honestly.
- Efficiency claims about KV forking are explicitly out of scope here; vLLM
  prefix caching stands in for the shared-prefix mechanism.
