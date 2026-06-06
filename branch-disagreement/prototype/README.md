# branch-disagreement prototype

The experiment harness. Pure-logic modules run on the standard library; the
real-model path lives behind the vLLM runner and is only used on the GPU box.

## Layout

```text
src/branch_disagreement/
  config.py        ExperimentConfig
  normalize.py     short-answer normalization + correctness matching
  datasets.py      EvalQuestion schema; sample / PopQA / TriviaQA loaders
  model_runner.py  ProxyRunner (CPU) and VLLMRunner (GPU), shared-prefix branches
  clustering.py    ExactMatchClusterer (CPU) and NLIClusterer (GPU)
  scoring.py       branch_disagreement + 3 baselines, all from the same branches
  metrics.py       pure-stdlib AUROC, bootstrap CI, DeLong test
  pipeline.py      generate -> cluster -> score -> label -> metrics
  report.py        CSV/JSON output + console summary
scripts/
  run_experiment.py   CLI entry point (proxy or vllm)
  smoke_test.py       CPU end-to-end proxy check
tests/                pure-stdlib unit tests
```

## Run locally (no GPU)

```bash
python3 -m unittest discover -s tests
PYTHONPATH=src python3 scripts/smoke_test.py
PYTHONPATH=src python3 scripts/run_experiment.py --engine proxy --output-prefix proxy_smoke
```

## Run on the GPU box

```bash
pip install -e ".[gpu]"        # done by deploy/vast_setup.sh
PYTHONPATH=src python3 scripts/run_experiment.py \
  --engine vllm --model Qwen/Qwen2.5-14B-Instruct \
  --dataset popqa --limit 500 --n-branches 8 \
  --branch-mode self_consistency --output-prefix popqa_qwen14b_sc
```

Outputs land in `reports/<prefix>_summary.csv`, `_detail.csv`, and `.json`.
The summary CSV is the headline: AUROC per detector + DeLong p vs the
branch-disagreement detector, alongside the per-question token cost.
