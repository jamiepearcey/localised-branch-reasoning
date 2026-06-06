# branch-disagreement

Can **disagreement across multiple reasoning branches tell us when a model is
hallucinating** — and can those branches be produced cheaply by sharing a cached
prefix?

This project is the rigorous, hardware-independent proof of "Idea 2" from the
sibling [`localised-reasoning`](../localised-reasoning) project. It is
deliberately decoupled from the Apple-Silicon KV-fork engineering so the concept
can be validated on a rented GPU in an afternoon.

## The claim under test

> A branch-disagreement score computed over prefix-shared branches predicts
> answer correctness with AUROC competitive with full **semantic entropy**
> (Farquhar/Kuhn/Gal, *Nature* 2024) and clearly above naive token-probability
> confidence, at materially **lower token cost**.

The detection idea is established prior art. The contribution is *cost*: getting
the multiple paths via prefix-shared branches rather than N independent samples,
and reporting quality (AUROC) jointly with compute (tokens / latency). See
[ADR-0002](docs/decisions/ADR-0002-experiment-design.md).

## How it works

```text
PopQA / TriviaQA  ->  sample N prefix-shared branches per question
                  ->  cluster branch answers by meaning (NLI entailment)
                  ->  score: branch_disagreement, semantic_entropy,
                            lexical_agreement, mean_token_logprob
                  ->  AUROC of each score vs "answer is wrong" (+ DeLong test)
                  ->  report AUROC-vs-token-cost frontier
```

## Folder guide

- `prototype/`: the experiment harness (datasets, runners, scoring, metrics, tests).
- `deploy/`: vast.ai SSH toolkit (deploy, remote setup, run, fetch results).
- `notes/`, `papers/`, `experiments/`: research notes, references, run write-ups.

## Quick start (local, no GPU)

The full pipeline runs on CPU in **proxy mode** — a synthetic model with
controllable disagreement — so the scoring and metric code can be exercised with
no heavy dependencies:

```bash
cd prototype
python3 -m unittest discover -s tests          # pure-stdlib unit tests
PYTHONPATH=src python3 scripts/smoke_test.py    # end-to-end proxy run
```

Proxy numbers are a **process check, not a quality claim**.

## Real run (vast.ai GPU)

See [docs/workflows/deployment.md](docs/workflows/deployment.md). In short:

```bash
cp deploy/config.env.example deploy/config.env   # fill in SSH + model
cd deploy
./check_connection.sh && ./deploy.sh && ./vast_setup.sh && ./run_remote.sh
./fetch_results.sh
```

## Status

Harness complete and green in proxy mode. No real GPU run yet — that is the next
task (`docs/tasks/current.md`).
