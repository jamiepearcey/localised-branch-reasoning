# Project Brief

## What this project is

A hardware-independent experiment that tests whether **disagreement across
multiple reasoning branches predicts model hallucination**, and whether those
branches can be produced cheaply by sharing a cached prefix (the "localised
reasoning" idea from the sibling `localised-reasoning` project).

## What problem it solves

Hallucination detection. When a model is asked something it has not internalised,
independently sampled reasoning paths tend to disagree. This project measures how
well that disagreement separates correct answers from wrong ones, and at what
compute cost relative to existing methods.

## What is innovative or distinctive

The detection idea itself is grounded in published work (semantic entropy,
Farquhar/Kuhn/Gal, *Nature* 2024). The distinctive angle here is **cost**:
producing the multiple paths via KV-prefix-shared branches (localised reasoning)
rather than N fully independent samples, and reporting detection quality
(AUROC) jointly with compute (tokens / latency). The claim under test is
"equivalent detection at lower cost," not "better detection."

## Primary hypothesis (the thing to prove)

A branch-disagreement score computed over prefix-shared branches achieves AUROC
for predicting answer correctness that is competitive with full semantic entropy
and clearly above naive token-probability confidence, at materially lower token
cost.
