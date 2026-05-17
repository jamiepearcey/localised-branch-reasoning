#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from localised_reasoning.qa_scenarios import (
    ReasoningBudget,
    SyntheticScenarioEngine,
    build_qa_dataframe,
    build_reasoning_dataframe,
    default_scenarios,
    default_trap_questions,
    export_qa_csv,
    export_qa_excel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scenario-branched Q&A evaluation from Python.")
    parser.add_argument("--limit", type=int, default=5, help="Number of built-in trap questions to run.")
    parser.add_argument("--output-xlsx", type=Path, default=Path("reports/qa_scenario_branches.xlsx"))
    parser.add_argument("--output-csv", type=Path, default=Path("reports/qa_scenario_branches.csv"))
    parser.add_argument("--reasoning-output-csv", type=Path, default=Path("reports/qa_reasoning_baseline.csv"))
    parser.add_argument("--branch-max-new-tokens", type=int, default=96)
    parser.add_argument("--judge-max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--no-xlsx",
        action="store_true",
        help="Skip Excel export and write only CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    questions = default_trap_questions()[: args.limit]
    scenarios = default_scenarios()
    engine = SyntheticScenarioEngine()
    budget = ReasoningBudget(
        branch_count=len(scenarios),
        branch_max_new_tokens=args.branch_max_new_tokens,
        judge_max_new_tokens=args.judge_max_new_tokens,
    )

    df = build_qa_dataframe(engine=engine, questions=questions, scenarios=scenarios)
    reasoning_df = build_reasoning_dataframe(engine=engine, questions=questions, budget=budget)
    csv_path = export_qa_csv(df, args.output_csv)
    reasoning_csv_path = export_qa_csv(reasoning_df, args.reasoning_output_csv)
    print(f"wrote_csv={csv_path}")
    print(f"wrote_reasoning_csv={reasoning_csv_path}")

    if not args.no_xlsx:
        xlsx_path = export_qa_excel(df, args.output_xlsx, reasoning_df=reasoning_df)
        print(f"wrote_xlsx={xlsx_path}")

    print(f"reasoning_budget_tokens={budget.total_tokens} ({budget.formula})")
    print(df.to_string(index=False))
    print(reasoning_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
