#!/usr/bin/env python3
"""Run the branch-disagreement experiment.

Proxy (CPU, no deps):
    PYTHONPATH=src python3 scripts/run_experiment.py --engine proxy

Real model (GPU box):
    PYTHONPATH=src python3 scripts/run_experiment.py \
        --engine vllm --model Qwen/Qwen2.5-14B-Instruct \
        --dataset popqa --limit 500 --n-branches 8 --branch-mode self_consistency \
        --output-prefix popqa_qwen14b_sc
"""

import argparse
import sys

from branch_disagreement.clustering import get_clusterer
from branch_disagreement.config import ExperimentConfig
from branch_disagreement.model_runner import build_runner
from branch_disagreement.pipeline import run_experiment
from branch_disagreement.report import print_summary, write_all


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="branch-disagreement experiment")
    p.add_argument("--engine", choices=["proxy", "vllm"], default="proxy")
    p.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--nli-model", default="microsoft/deberta-large-mnli")
    p.add_argument("--dataset", default="sample", choices=["sample", "popqa", "triviaqa"])
    p.add_argument("--split", default="", help="dataset split ('' = per-dataset default)")
    p.add_argument("--response-mode", choices=["short", "long"], default="short")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--n-branches", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--branch-mode", choices=["self_consistency", "localised"],
                   default="self_consistency")
    p.add_argument("--entropy-weighting", choices=["count", "likelihood"], default="count")
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--output-prefix", default="branch_disagreement")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = ExperimentConfig(
        engine=args.engine,
        model=args.model,
        nli_model=args.nli_model,
        dataset=args.dataset,
        split=args.split,
        response_mode=args.response_mode,
        limit=args.limit,
        n_branches=args.n_branches,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        branch_mode=args.branch_mode,
        entropy_weighting=args.entropy_weighting,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        reports_dir=args.reports_dir,
        output_prefix=args.output_prefix,
    )
    runner = build_runner(config)
    clusterer = get_clusterer(config.engine, config.nli_model)
    result = run_experiment(config, runner, clusterer)
    print_summary(result)
    paths = write_all(result, config.reports_dir, config.output_prefix)
    print("wrote:")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
