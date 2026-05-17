#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from localised_reasoning.ollama_branching import (  # noqa: E402
    BRANCHES,
    BUILD_BRANCH,
    DEFAULT_MODEL,
    REVIEW_BRANCH,
    assess_branch_pair,
    assess_branch_output,
    generate_branch,
)


CASES = {
    "kv_fork": """\
We are designing a localized reasoning inference runtime. The runtime should
fork a shared prefix KV cache, inject hidden text continuation markers into
each branch, and let branches continue independently. Explain the next
engineering step.
""",
    "marker_risk": """\
Within the current text-marker-only prototype, we need to reduce the risk that
ordinary marker text carries learned semantics and biases weaker models. We are
not training special tokens, inspecting activations, inspecting hidden states,
or disabling shared prefix blocks. Decide the immediate black-box validation
step using only generated outputs and cache metadata. The no-marker condition
and inert-user-text condition should remain semantically equivalent; the
trusted-runtime-control condition should follow the active branch instruction.
""",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate branch-marker continuations across contexts and seeds."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-predict", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument(
        "--seeds",
        default="11",
        help="Comma-separated deterministic seeds, for example 11,12,13.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any branch or pair assessment fails.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Write a machine-readable run summary to this path.",
    )
    args = parser.parse_args()

    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    total_checks = 0
    failed_checks = 0
    records: list[dict[str, object]] = []

    for seed in seeds:
        print(f"# seed: {seed}")
        seed_record: dict[str, object] = {"seed": seed, "cases": []}
        case_records: list[dict[str, object]] = []
        for case_name, context in CASES.items():
            print(f"## case: {case_name}")
            outputs: dict[str, str] = {}
            branch_records: list[dict[str, object]] = []
            for branch in BRANCHES:
                output = generate_branch(
                    model=args.model,
                    shared_context=context,
                    branch=branch,
                    num_predict=args.num_predict,
                    temperature=args.temperature,
                    seed=seed,
                )
                outputs[branch.name] = output
                assessment = assess_branch_output(branch, output)
                total_checks += 1
                failed_checks += 0 if assessment.passed else 1
                branch_records.append(
                    {
                        "branch": branch.name,
                        "label": branch.label,
                        "passed": assessment.passed,
                        "issues": list(assessment.issues),
                        "output": output,
                    }
                )
                print(f"### {branch.label}")
                print(output)
                print(f"assessment={'PASS' if assessment.passed else 'FAIL'}")
                for issue in assessment.issues:
                    print(f"- {issue}")
                print()
            pair_assessment = assess_branch_pair(
                outputs[BUILD_BRANCH.name],
                outputs[REVIEW_BRANCH.name],
            )
            total_checks += 1
            failed_checks += 0 if pair_assessment.passed else 1
            case_records.append(
                {
                    "case": case_name,
                    "branches": branch_records,
                    "pair_assessment": {
                        "passed": pair_assessment.passed,
                        "issues": list(pair_assessment.issues),
                        "lexical_overlap": pair_assessment.lexical_overlap,
                    },
                }
            )
            print(f"pair_assessment={'PASS' if pair_assessment.passed else 'FAIL'}")
            print(f"lexical_overlap={pair_assessment.lexical_overlap:.2f}")
            for issue in pair_assessment.issues:
                print(f"- {issue}")
            print()
        seed_record["cases"] = case_records
        records.append(seed_record)
    passed_checks = total_checks - failed_checks
    print(f"summary={passed_checks}/{total_checks} checks passed")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "num_predict": args.num_predict,
                    "temperature": args.temperature,
                    "seeds": seeds,
                    "passed_checks": passed_checks,
                    "failed_checks": failed_checks,
                    "total_checks": total_checks,
                    "records": records,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.strict and failed_checks:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
