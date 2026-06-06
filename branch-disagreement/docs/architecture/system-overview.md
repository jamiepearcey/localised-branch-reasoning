# System Overview

## Summary

A hardware-independent experiment measuring whether disagreement across
prefix-shared reasoning branches predicts model hallucination, and at what
compute cost relative to baselines.

## Pipeline

```text
dataset (PopQA / TriviaQA)
   -> normalize to EvalQuestion {id, question, gold_answers, metadata}
   -> for each question:
        build shared prefix (system + few-shot + question)
        sample N branches from the shared prefix
          - self-consistency mode: N samples, same prompt, temperature > 0
          - localised mode: N branches, each with a distinct approach marker
   -> extract a short answer per branch
   -> cluster branch answers by meaning (NLI bidirectional entailment;
      exact-match fallback when no NLI model is available)
   -> compute scores per question:
        * branch_disagreement  (cluster-based; the method under test)
        * semantic_entropy      (Kuhn et al; the quality reference)
        * lexical_agreement     (naive self-consistency baseline)
        * mean_token_logprob    (naive confidence baseline)
   -> label each question correct / incorrect (deterministic normalization)
   -> metrics: AUROC of each score vs error, bootstrap CI, DeLong comparison
   -> report: AUROC table + AUROC-vs-token-cost frontier
```

## Major directories

- `prototype/src/branch_disagreement/`: library code.
- `prototype/scripts/`: entry points (`run_experiment.py`, `smoke_test.py`).
- `prototype/tests/`: pure-stdlib unit tests.
- `deploy/`: SSH toolkit for the vast.ai GPU box.

## Runner abstraction

`model_runner.py` defines a `ModelRunner` protocol with two implementations:

- `ProxyRunner`: dependency-free, deterministic, CPU-only. Generates synthetic
  branch answers with controllable disagreement so the full pipeline and the
  scoring/metric code can be exercised without a GPU. Process check only.
- `VLLMRunner`: loads a real model under vLLM, relies on automatic prefix caching
  to make the shared prefix near-free across branches, and records token counts
  and wall-clock for the cost axis.

## Distinctive point

The detection signal is established prior art; the contribution is producing it
cheaply via prefix-shared branches and reporting the quality/cost trade-off
explicitly.

## Notes

- Expand this file as subsystems harden. Add per-subsystem pages under this
  folder when they become substantial.
