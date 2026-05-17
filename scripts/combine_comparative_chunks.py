#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from localised_reasoning.comparative_eval import (
    build_category_summary,
    build_summary,
    export_comparative_eval_excel,
)
from localised_reasoning.qa_scenarios import export_qa_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine comparative benchmark chunk CSVs.")
    parser.add_argument("--detail-csv", action="append", type=Path, required=True)
    parser.add_argument("--branch-csv", action="append", type=Path, required=True)
    parser.add_argument("--reasoning-csv", action="append", type=Path, required=True)
    parser.add_argument("--output-xlsx", type=Path, required=True)
    parser.add_argument("--output-detail-csv", type=Path, required=True)
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-category-summary-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    detail_df = _concat_csvs(args.detail_csv)
    branch_df = _concat_csvs(args.branch_csv)
    reasoning_df = _concat_csvs(args.reasoning_csv)
    summary_df = build_summary(detail_df)
    category_summary_df = build_category_summary(detail_df)

    export_qa_csv(detail_df, args.output_detail_csv)
    export_qa_csv(summary_df, args.output_summary_csv)
    export_qa_csv(category_summary_df, args.output_category_summary_csv)
    output = export_comparative_eval_excel(
        detail_df=detail_df,
        summary_df=summary_df,
        branch_df=branch_df,
        reasoning_df=reasoning_df,
        output_path=args.output_xlsx,
    )
    print(f"wrote_xlsx={output}")
    print(summary_df.to_string(index=False))
    print("\nby_category")
    print(category_summary_df.to_string(index=False))
    return 0


def _concat_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    raise SystemExit(main())
