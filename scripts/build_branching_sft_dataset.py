#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from localised_reasoning.branch_training import (  # noqa: E402
    format_branching_sft_text,
    parse_planned_branch_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert planned-branch demo reports into SFT JSONL seed data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=sorted((ROOT / "reports").glob("planned_branch_kv_demo*.txt")),
        help="One or more planned_branch_kv_demo report files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "branching_sft_seed.jsonl",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for path in args.input:
            record = parse_planned_branch_report(path.read_text(encoding="utf-8"))
            payload = {
                "text": format_branching_sft_text(record),
                "metadata": {
                    "source": str(path),
                    "decision_point": record.decision_point,
                    "factors": list(record.factors),
                    "branch_count": len(record.branches),
                },
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote_examples={written}")
    print(f"output={args.output}")
    if written == 0:
        raise SystemExit("no examples written")


if __name__ == "__main__":
    main()
