#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from localised_reasoning.benchmark_datasets import benchmark_info, load_benchmark_questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a normalized benchmark question CSV.")
    parser.add_argument("--source", choices=["mmlu-pro"], default="mmlu-pro")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--categories", default="")
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument("--output-csv", type=Path, default=Path("data/benchmarks/mmlu_pro_500.csv"))
    parser.add_argument("--info", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.info:
        info = benchmark_info(args.source, split=args.split)
        print(f"source={info.source}")
        print(f"split={info.split}")
        print(f"row_count={info.row_count}")
        print(f"sampled_categories={', '.join(info.categories)}")

    categories = [item.strip() for item in args.categories.split(",") if item.strip()]
    questions = load_benchmark_questions(
        source=args.source,
        split=args.split,
        limit=args.limit,
        offset=args.offset,
        categories=categories,
        shuffle_seed=args.shuffle_seed,
    )
    rows = [
        {
            "question_id": question.question_id,
            "category": question.category,
            "question": question.question,
            "expected_answer": question.expected_answer,
        }
        for question in questions
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"wrote_csv={args.output_csv}")
    print(f"question_count={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
