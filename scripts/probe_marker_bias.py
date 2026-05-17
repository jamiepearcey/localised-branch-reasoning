#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from localised_reasoning.ollama_branching import (  # noqa: E402
    BUILD_BRANCH,
    DEFAULT_MODEL,
    run_marker_bias_probe,
)


DEFAULT_CONTEXT = """\
We are designing a localized reasoning inference runtime. In one compact
paragraph, state the next implementation validation needed for KV-cache forking.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run no-marker, inert-user-text, and trusted-control marker-bias probes."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--context", default=DEFAULT_CONTEXT)
    parser.add_argument("--num-predict", type=int, default=140)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--seeds",
        default="21",
        help="Comma-separated deterministic seeds, for example 21,22,23.",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Write a machine-readable run summary to this path.",
    )
    args = parser.parse_args()

    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    failed = 0
    records: list[dict[str, object]] = []

    for seed in seeds:
        result = run_marker_bias_probe(
            model=args.model,
            shared_context=args.context,
            branch=BUILD_BRANCH,
            num_predict=args.num_predict,
            temperature=args.temperature,
            seed=seed,
        )

        failed += 0 if result.passed else 1
        records.append(
            {
                "seed": seed,
                "passed": result.passed,
                "issues": list(result.issues),
                "inert_similarity": result.inert_similarity,
                "trusted_similarity": result.trusted_similarity,
                "outputs": {
                    "no_marker": result.no_marker_output,
                    "inert_user_text": result.inert_user_text_output,
                    "trusted_runtime_control": result.trusted_runtime_control_output,
                },
            }
        )
        print(f"# seed: {seed}")
        print("## no-marker condition")
        print(result.no_marker_output)
        print()
        print("## inert-user-text condition")
        print(result.inert_user_text_output)
        print()
        print("## trusted-runtime-control condition")
        print(result.trusted_runtime_control_output)
        print()
        print(f"inert_similarity={result.inert_similarity:.2f}")
        print(f"trusted_similarity={result.trusted_similarity:.2f}")
        print(f"assessment={'PASS' if result.passed else 'FAIL'}")
        for issue in result.issues:
            print(f"- {issue}")
        print()

    passed = len(seeds) - failed
    print(f"summary={passed}/{len(seeds)} probes passed")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "context": args.context,
                    "num_predict": args.num_predict,
                    "temperature": args.temperature,
                    "seeds": seeds,
                    "passed_probes": passed,
                    "failed_probes": failed,
                    "total_probes": len(seeds),
                    "records": records,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.strict and failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
