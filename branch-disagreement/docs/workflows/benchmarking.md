# Benchmarking Workflow

## Expectations

- Benchmark performance-sensitive work when the task changes sampling paths,
  clustering, or scoring hot paths.
- Record benchmark commands and notable results when they influence decisions.

## Project notes

- The headline result is itself a benchmark: AUROC of each detector against a
  correctness label, plus the token / latency cost that produced it.
- Always report the cost axis alongside AUROC (see `.context/invariants.md`).
- Record the model, dataset slice, branch count, and decoding temperature for any
  headline number so it is reproducible.
